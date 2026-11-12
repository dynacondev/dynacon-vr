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
// on the baked normal). Two pipelines, two passes: a D16 mesh-depth
// prepass, then the dots sample it plus the mask-rendered proximity
// sphere field (one sphere raster per frame, owned by the feather mask
// hidden record) at their CENTERS — whole circles live or die, never
// half-chopped by geometry. Trivial circle fragment with a 1px AA band
// and coverage fade at distance.
//
// Composited through the union-merge layer (not its own layer): the
// merged mask image carries premultiplied dot RGB holed out of its
// alpha, so the flat passthrough passes dots through where the
// camera would otherwise cover them. Own swapchain, no depth. Render
// thread only.

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
	// waitIdle inside on rebuild; no-op when unchanged. Returns true
	// when anything was rebuilt (swapchain, pipelines, depth targets):
	// the feather mask hidden record builds framebuffers against the
	// sphere views below and must drop them (reset_targets) on rebuild.
	bool ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h);

	bool ready() const
	{
		return (bool)swapchain_;
	}

	// Borrowed submitted-image array view (both layers) for the
	// merge pass: the just-recorded image's multiview color view,
	// sampled (not rendered) by the merge. Built for every recorded
	// image alongside the framebuffers, so any image recorded this
	// frame has one. Null when unknown (retired image or never
	// recorded): the caller must skip the input loudly, never
	// silently. Same borrowed lifetime as the mesh buffers above
	// (dies with ensure() rebuilds — merge uses it same-frame only,
	// so no reset coupling needed).
	vk::ImageView submitted_view_array(vk::Image image) const;

	// Acquire + wait (100ms); -1 on timeout (pairing-safe: the untouched
	// acquisition is released inside, skip the frame).
	int acquire();

	// Record draws into both eye layers of the acquired image (union of
	// the draws' scissor rects per eye; render-area CLEAR stays full).
	// No-op (false) when draws is empty. extents size the per-eye
	// viewports (the submitted layer rects). The proximity sphere field
	// is NOT rasterized here: the feather mask hidden record renders it
	// into sphere_depth_ below (same command buffer, earlier) and this
	// pass only samples it — one sphere raster per frame, not two.
	// Callers must only record after the mask record ran this frame
	// (fresh field guaranteed); on mask skip frames the image holds
	// last frame's field and dots must sit out. Returns false without
	// recording (caller must release the acquisition to keep pairing).
	bool record(vk::raii::CommandBuffer & cmd, size_t image_index,
	            const std::array<vk::Extent2D, 2> & extents,
	            const std::vector<instance_draw> & draws);

	// Release the acquired image (after submit, before endFrame — the
	// compositor takes the last released image).
	void release();

	xr::swapchain & swapchain()
	{
		return swapchain_;
	}

	// Shared proximity sphere field (P4: written by the feather mask
	// hidden record, sampled by the dots pass). Views die with this
	// renderer — borrowers must reset first (see ensure()).
	VkImage sphere_image() const
	{
		return sphere_depth_;
	}
	// Array view over both sphere layers for the mask's multiview
	// sphere framebuffer (same borrowed lifetime as the image).
	vk::ImageView sphere_view_array() const
	{
		return *sphere_view_array_;
	}

private:
	void init_pipelines(vk::Format format);

	vk::raii::Device * device = nullptr;
	vk::raii::PhysicalDevice * physical_device = nullptr;
	std::shared_ptr<vk::raii::PipelineCache> pipe_cache;

	vk::raii::PipelineLayout pipeline_layout{nullptr};
	// Color-only renderpass (CLEAR/STORE/OPTIMAL — submitted image,
	// nothing app-side re-reads it) for the dots pass; depth-only
	// renderpass (CLEAR/STORE, ends shader-readable) for the mesh
	// prepass. Both multiview (viewMask=both).
	vk::raii::RenderPass color_renderpass{nullptr};
	// Shared by the mesh prepass (GREATER over a far-clear): D16
	// CLEAR/STORE/shader-readable depth for the dots pass to sample.
	vk::raii::RenderPass depth_renderpass{nullptr};
	vk::raii::Pipeline pipeline{nullptr};
	// Depth-fill pipeline (mesh triangles into the depth attachment,
	// color writes off): the dots sample its output, they don't
	// depth-test against it. Shared layout (subset of push).
	vk::raii::Pipeline fill_pipeline{nullptr};
	// Depth target (D16, quarter-res, 2 layers, one array view): written
	// by the mesh prepass, sampled by the dots pass (nearest: exact
	// texel, no cross-edge blending).
	image_allocation depth_;
	vk::raii::ImageView depth_view_array{nullptr};
	// Proximity sphere depth (D16, quarter-res, 2 layers, one array
	// view): min-accumulated furthest-exit field, sampled alongside the
	// mesh depth (binding 1).
	// SHARED with the feather mask hidden record: the mask renders the
	// spheres into this image (same command buffer, earlier pass) and
	// transitions it to shader-readable; this renderer never writes it.
	// Borrowed-framebuffer discipline (mask builds its sphere FB against
	// the array view): ensure() reports rebuilds so the mask can drop
	// its FBs first — views die with this renderer.
	image_allocation sphere_depth_;
	// Array view over both sphere layers: sampled by the dots pass
	// (binding 1, indexed by gl_ViewIndex) and lent to the mask's
	// multiview sphere framebuffer (see sphere_view_array()).
	vk::raii::ImageView sphere_view_array_{nullptr};
	// One descriptor set over the two array views above (rewritten on
	// rebuild, not per frame). Nearest sampler.
	vk::raii::DescriptorSetLayout descriptor_layout{nullptr};
	vk::raii::DescriptorPool descriptor_pool{nullptr};
	std::vector<vk::raii::DescriptorSet> descriptor_sets;
	vk::raii::Sampler depth_sampler{nullptr};
	// Static billboard quad (6 corners, -1..1, non-indexed), uploaded
	// once with the pipeline. Per-dot data arrives per draw as
	// instanced attributes from the borrowed feather buffer.
	buffer_allocation quad_;

	xr::swapchain swapchain_;
	vk::Format format_ = vk::Format::eUndefined;
	vk::Extent2D extent_{0, 0};

	struct target
	{
		vk::raii::ImageView view{nullptr}; // color array view (both eyes)
		vk::raii::Framebuffer framebuffer{nullptr}; // color MV pass
		vk::raii::Framebuffer depth_framebuffer{nullptr}; // mesh prepass MV
	};
	// [image], parallel to the swapchain image list (one multiview
	// target per image — both eyes broadcast in one pass).
	std::vector<target> targets;
};
