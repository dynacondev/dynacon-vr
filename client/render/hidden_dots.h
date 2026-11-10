/*
 * WiVRn VR streaming
 * Copyright (C) 2026  Guillaume Meunier <guillaume.meunier@centraliens.net>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

// Hidden-object dot overlay (3d-passthrough-hidden): a fixed world-size dot
// grid wrapping the mesh, drawn over the revealed camera feed in the config
// color. Real camera-facing billboard quads, instanced from per-dot
// center+normal lists baked once per (mesh, pitch) by the feather hidden
// renderer (which keeps owning upload + eviction; dots just borrow).
// Screen-aligned by construction, so every dot is round from any angle
// and the quad extends a full radius past the center, so triangle edges
// never slice dots. Backfaces collapse in the vertex shader (facing test
// on the baked normal). Three pipelines, three passes: a D16 mesh-depth
// prepass plus a D16 sphere-field prepass, then the dots sample both
// stored depths at their CENTERS — whole circles live or die, never
// half-chopped by geometry. Trivial circle fragment with a 1px AA band
// and coverage fade at distance.
//
// Composited as ONE transparent projection layer (standard alpha blend,
// above the flat passthrough, below gizmos/GUI) so the maxLayerCount
// discipline holds. Own swapchain, no depth. Render thread only.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "vk/allocation.h"
#include "xr/swapchain.h"

namespace xr
{
class instance;
class session;
} // namespace xr

class hidden_dots_renderer
{
public:
	// Dot radius in lattice cells (diameter = 2x this x pitch): the
	// single size knob, packed into push view_params.w per draw and read
	// by both shaders (quad size + circle edge), so they can't disagree.
	// Floor note: radii far below ~0.08 cells go subpixel at room range
	// (2m, 5cm pitch) and fade by coverage — invisible, not broken.
	static constexpr float dot_radius_cells = 0.05f;
	// One silhouette draw: per-eye clip transforms (placement already
	// baked in), dot color + pitch, per-eye scissor rects (dots pixel
	// space, computed stream-side from the same cull corners), per-eye
	// view params (viewport W/H px + proj[1][1] for the world→px radius
	// + dot radius cells in w), the baked dot instance buffer borrowed
	// from the feather hidden renderer (per-dot center + face normal;
	// count 0 = skip), and the mesh triangle buffers for the depth
	// prepass (occluded dots must not show through).
	struct instance_draw
	{
		std::array<glm::mat4, 2> mvp;
		glm::vec3 color{1.0f, 0.16f, 0.10f};
		float pitch = 0.05f;
		std::array<vk::Rect2D, 2> scissor;
		std::array<glm::vec3, 2> view_params; // x: view W px, y: view H px, z: proj[1][1]
		vk::Buffer dot_buffer;
		uint32_t dot_count = 0;
		vk::Buffer vertex_buffer;
		vk::Buffer index_buffer;
		uint32_t index_count = 0;
	};

	explicit hidden_dots_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device);

	// Shared driver pipeline cache (one per device, same object the
	// feather mask renderer uses). Held by each renderer, so the cache
	// outlives its users but dies with the last one. Render-thread only.
	static std::shared_ptr<vk::raii::PipelineCache> pipeline_cache(vk::raii::Device & device);
	// Force a first build now (e.g. at first video swapchain, before any
	// map exists) instead of paying driver compiles on the first visible
	// frame. A throwaway renderer populates the cache above.
	static void warm_pipelines(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device, vk::Format format);

	// (Re)create swapchain + targets + pipeline for format/size.
	// waitIdle inside on rebuild; no-op when unchanged.
	void ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h);

	bool ready() const
	{
		return (bool)swapchain_;
	}

	// Acquire + wait (100ms); -1 on timeout (pairing-safe: the untouched
	// acquisition is released inside, skip the frame).
	int acquire();

	// Record draws into both eye layers of the acquired image (union of
	// the draws' scissor rects per eye; render-area CLEAR stays full).
	// No-op (false) when draws is empty. extents size the per-eye
	// viewports (the submitted layer rects). spheres are proximity
	// boundary MVPs (one per sphere, both eyes; model = translate(center)
	// * scale(radius), stream-built so the renderer stays timebase-free):
	// drawn first into the prepass with a min-accumulating LESS test, so
	// the stored field is the furthest exit = inside ANY sphere. Returns
	// false without recording (caller must release the acquisition to
	// keep pairing).
	bool record(vk::raii::CommandBuffer & cmd, size_t image_index,
	            const std::array<vk::Extent2D, 2> & extents,
	            const std::vector<instance_draw> & draws,
	            const std::vector<std::array<glm::mat4, 2>> & spheres);

	// Release the acquired image (after submit, before endFrame — the
	// compositor takes the last released image).
	void release();

	xr::swapchain & swapchain()
	{
		return swapchain_;
	}

private:
	void init_pipelines(vk::Format format);

	vk::raii::Device * device = nullptr;
	vk::raii::PhysicalDevice * physical_device = nullptr;
	std::shared_ptr<vk::raii::PipelineCache> pipe_cache;

	vk::raii::PipelineLayout pipeline_layout{nullptr};
	// Color-only renderpass (CLEAR/STORE/GENERAL) for the dots pass;
	// depth-only renderpass (CLEAR/STORE, ends shader-readable) for the
	// prepasses. The sphere prepass reuses the same depth RP object over
	// its own image (near-clear for LESS); only clear values and
	// framebuffers differ.
	vk::raii::RenderPass color_renderpass{nullptr};
	// Shared by the mesh prepass (GREATER over a far-clear) and the
	// proximity sphere prepass (LESS over a near-clear): same D16
	// CLEAR/STORE/shader-readable shape, only the clear values and the
	// framebuffers differ. Separate IMAGES (the fields must not mix).
	vk::raii::RenderPass depth_renderpass{nullptr};
	vk::raii::Pipeline pipeline{nullptr};
	// Depth-fill pipeline (mesh triangles into the depth attachment,
	// color writes off): the dots sample its output, they don't
	// depth-test against it. Shared layout (subset of push).
	vk::raii::Pipeline fill_pipeline{nullptr};
	// Proximity sphere pipeline (unit-sphere verts into the sphere depth
	// image): cull-front keeps the inside/exit wall from any viewpoint;
	// LESS min-accumulates the furthest exit across spheres. Same stages
	// (dotfill), layout and renderpass as the mesh fill above.
	vk::raii::Pipeline sphere_pipeline{nullptr};
	// Depth target (D16, half-res, 2 layers + one single-layer view per
	// eye), recreated with the swapchain in ensure(). Sampled by the
	// dots pass (nearest: exact texel, no cross-edge blending).
	image_allocation depth_;
	std::array<vk::raii::ImageView, 2> depth_views{nullptr, nullptr};
	// Proximity sphere depth (D16, half-res, same shape): min-accumulated
	// furthest-exit field, sampled alongside the mesh depth (binding 1).
	image_allocation sphere_depth_;
	std::array<vk::raii::ImageView, 2> sphere_views{nullptr, nullptr};
	// One descriptor set per eye over the single-layer depth view above
	// (rewritten on rebuild, not per frame). Nearest sampler.
	vk::raii::DescriptorSetLayout descriptor_layout{nullptr};
	vk::raii::DescriptorPool descriptor_pool{nullptr};
	std::vector<vk::raii::DescriptorSet> descriptor_sets;
	vk::raii::Sampler depth_sampler{nullptr};
	// Static billboard quad (6 corners, -1..1, non-indexed), uploaded
	// once with the pipeline. Per-dot data arrives per draw as
	// instanced attributes from the borrowed feather buffer.
	buffer_allocation quad_;
	// Unit-sphere verts (non-indexed, outward-CCW) for the proximity
	// prepass, uploaded once. Expanded per frame by the sphere MVPs.
	buffer_allocation sphere_verts_;
	uint32_t sphere_vert_count = 0;

	xr::swapchain swapchain_;
	vk::Format format_ = vk::Format::eUndefined;
	vk::Extent2D extent_{0, 0};

	struct target
	{
		vk::raii::ImageView view{nullptr};
		vk::raii::Framebuffer framebuffer{nullptr}; // color pass
		vk::raii::Framebuffer depth_framebuffer{nullptr}; // mesh prepass
		vk::raii::Framebuffer sphere_framebuffer{nullptr}; // sphere prepass
	};
	// [image][eye], parallel to the swapchain image list.
	std::vector<std::array<target, 2>> targets;
};
