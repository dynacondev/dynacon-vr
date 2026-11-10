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

// Hidden-object warning outline (3d-passthrough-hidden): a flat guardian-
// style rim around the revealed camera feed, following the mesh silhouette.
//
// Inverted-hull in one renderpass: an occluder prepass draws the original
// mesh with color writes masked off (depth only), then the rim pass draws
// the shell expanded ~2cm about the model-space AABB center with front
// faces culled, depth-tested against the prepass. Interior fails, the rim
// ring passes. Both shaders are trivial (MVP + flat color, no varyings,
// no derivatives): the triplanar grid this replaces cost ~80 ALU per pixel
// over the full silhouette, this costs ~nothing per pixel.
//
// Composited as ONE transparent projection layer (above the flat
// passthrough, below gizmos/GUI) so the maxLayerCount discipline holds.
// Own swapchain + own D16 depth (never sampled, cleared every frame).
// Render thread only.
//
// Notes: open single-sided meshes viewed from behind write no occluder
// depth (back culled), so the rim draws as a solid blob there — acceptable
// warning behavior, and closed volumes (the complaint: far-side lines
// showing through) resolve correctly. Overlapping hidden objects resolve
// nearest-wins via depth regardless of draw order.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "vk/allocation.h"
#include "xr/swapchain.h"
#include "render/passthrough_mesh.h"

namespace xr
{
class instance;
class session;
} // namespace xr

class hidden_outline_renderer
{
public:
	// One silhouette draw: which uploaded mesh, per-eye clip transforms
	// (placement already baked in), model-space shell center for the
	// vertex-stage expansion, and the flat outline color.
	struct instance_draw
	{
		std::string mesh;
		std::array<glm::mat4, 2> mvp;
		glm::vec3 color{128.f / 255.f, 212.f / 255.f, 1.0f};
		glm::vec3 obj_center{0};
	};

	explicit hidden_outline_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device);

	// Shared driver pipeline cache (one per device, same object the
	// feather mask renderer uses): every renderer compiles distinct
	// pipelines, but sharing keeps one cache object alive across them.
	// Held by each renderer, so the cache outlives its users but dies
	// with the last one. Render-thread only.
	static std::shared_ptr<vk::raii::PipelineCache> pipeline_cache(vk::raii::Device & device);
	// Force a first build now (e.g. at first video swapchain, before any
	// map exists) instead of paying driver compiles on the first visible
	// frame. A throwaway renderer populates the cache above.
	static void warm_pipelines(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device, vk::Format format);

	// (Re)create swapchain + targets + pipelines for format/size.
	// waitIdle inside on rebuild; no-op when unchanged.
	void ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h);

	bool ready() const
	{
		return (bool)swapchain_;
	}

	// Acquire + wait (100ms); -1 on timeout (pairing-safe: the untouched
	// acquisition is released inside, skip the frame).
	int acquire();

	// Record draws into both eye layers of the acquired image.
	// No-op when draws is empty. extents size the per-eye viewports
	// (the submitted layer rects). Both passes record inside one
	// renderpass per eye (occluder prepass, then rim).
	void record(vk::raii::CommandBuffer & cmd, size_t image_index,
	            const std::array<vk::Extent2D, 2> & extents,
	            const std::vector<instance_draw> & draws);

	// Release the acquired image (after submit, before endFrame — the
	// compositor takes the last released image).
	void release();

	xr::swapchain & swapchain()
	{
		return swapchain_;
	}

	// (Re)upload one object's mesh buffers under its key. Synchronous;
	// call when idle, e.g. on map change, not per frame. Empty soup
	// erases the entry.
	void set_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup);
	void remove_mesh(const std::string & key);
	// Model-space AABB for the frustum skip and the shell center
	// (nullopt = unknown key: skip the draw). Computed once at upload.
	std::optional<std::pair<glm::vec3, glm::vec3>> mesh_bounds(const std::string & key) const;

private:
	void init_pipelines(vk::Format format);

	vk::raii::Device * device = nullptr;
	vk::raii::PhysicalDevice * physical_device = nullptr;
	std::shared_ptr<vk::raii::PipelineCache> pipe_cache;

	vk::raii::PipelineLayout pipeline_layout{nullptr};
	vk::raii::RenderPass renderpass{nullptr};
	vk::raii::Pipeline pipeline_occluder{nullptr};
	vk::raii::Pipeline pipeline_rim{nullptr};

	xr::swapchain swapchain_;
	vk::Format format_ = vk::Format::eUndefined;
	vk::Extent2D extent_{0, 0};

	struct mesh_buffers
	{
		buffer_allocation vertex_buffer;
		buffer_allocation index_buffer;
		uint32_t index_count = 0;
		glm::vec3 aabb_min{0}, aabb_max{0};
	};
	std::map<std::string, mesh_buffers> meshes;

	// Depth-test-only D16 (never sampled, device-local). Shared across
	// swapchain images: a single frame records at a time.
	vk::raii::Image depth_image{nullptr};
	vk::raii::DeviceMemory depth_memory{nullptr};
	std::array<vk::raii::ImageView, 2> depth_views{nullptr, nullptr};

	struct target
	{
		vk::raii::ImageView view{nullptr};
		vk::raii::Framebuffer framebuffer{nullptr};
	};
	// [image][eye], parallel to the swapchain image list.
	std::vector<std::array<target, 2>> targets;
};
