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

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "vk/allocation.h"
#include "xr/swapchain.h"
#include "decoder/decoder.h"

// Fullscreen environment-depth visualizer: samples the runtime's D16
// depth array texture and false-colors it green (nearZ) to red (farZ).
// Composited as ONE transparent projection layer (standard alpha blend,
// above passthrough/video, below the GUI) so maxLayerCount discipline
// holds. Own swapchain, CLEAR-on-UNDEFINED every frame. The shader
// samples the depth image directly per eye layer: no reprojection, no
// history, no smoothing — a pure debug view. Render thread only.
namespace xr
{
class instance;
class session;
} // namespace xr

// Debug visualization selector (Passthrough tab, session-scoped,
// render thread only). None = off. Sensor = existing runtime D16
// false-color. Vr_depth = VR depth stream (SBS, green->red over
// 0..2.046m, far past 1020 black). Layer = composited common depth
// layer of placed passthrough objects before the verdict (same
// 0..2.046m ramp, far >2.041m saturates red, absent FAR black).
// Verdict = final VR-vs-passthrough comparison (red = VR wins incl.
// far-vs-far, green = passthrough, no-window black). Raster_vs_sensor
// validates the composited layer against the live sensor depth
// (green = agree, red ramp = |layer - sensor| over 0..2m, no-data
// black). All modes share the global opacity below.
enum class debug_viz_mode
{
	none = 0,
	sensor,
	vr_depth,
	layer,
	verdict,
	raster_vs_sensor,
};

inline const char *
debug_viz_mode_name(debug_viz_mode m)
{
	switch (m)
	{
		case debug_viz_mode::none:
			return "None";
		case debug_viz_mode::sensor:
			return "Depth Sensor";
		case debug_viz_mode::vr_depth:
			return "VR Depth";
		case debug_viz_mode::layer:
			return "Composited Depth";
		case debug_viz_mode::verdict:
			return "VR vs Passthrough";
		case debug_viz_mode::raster_vs_sensor:
			return "Raster vs Sensor";
	}
	return "None";
}

class depth_overlay_renderer
{
public:
	// Trimmed UBO mirror of depth_viz.frag params (std140): near/far
	// planes in meters plus the global opacity. 16 bytes, no pads.
	struct frame_params
	{
		float near_z = 0.1f;
		float far_z = 10.f;
		float opacity = 0.3f;
		float pad = 0;
	};

	explicit depth_overlay_renderer(vk::raii::Device & device);

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

	// Record the fullscreen viz into both eye layers of the acquired
	// image, sampling depth_view (runtime D16 2-layer array) with params.
	void record(vk::raii::CommandBuffer & cmd, size_t image_index,
	            const std::array<vk::Extent2D, 2> & extents,
	            vk::ImageView depth_view,
	            const frame_params & params);

	// Composited-layer viz (R16F linear meters, 2-layer array, half
	// merge res): green->red over 0..2.046m to match VR depth, FAR
	// (>=65000) black. Null view records opaque black (no merge yet).
	void record_layer(vk::raii::CommandBuffer & cmd, size_t image_index,
	                  const std::array<vk::Extent2D, 2> & extents,
	                  vk::Image layer_image, vk::ImageView layer_view,
	                  const frame_params & params);

	// Verdict viz (R8 binary + R16F layer for the no-window mask):
	// red = VR wins, green = passthrough, layer FAR black. Null views
	// record opaque black.
	void record_verdict(vk::raii::CommandBuffer & cmd, size_t image_index,
	                    const std::array<vk::Extent2D, 2> & extents,
	                    vk::Image verdict_image, vk::ImageView verdict_view,
	                    vk::Image layer_image, vk::ImageView layer_view,
	                    const frame_params & params);

	// Raster-vs-sensor validation viz (R16F layer + runtime D16 sensor
	// for the same eye UV): green = agree, red ramp = |layer - sensor|
	// over 0..2m, no-data (layer FAR / sensor no-return) black. Null
	// views record opaque black. Sensor needs no barrier (runtime hands
	// it ready to sample, like the sensor path); layer rides the
	// cross-CB read barrier.
	void record_raster_sensor(vk::raii::CommandBuffer & cmd, size_t image_index,
	                          const std::array<vk::Extent2D, 2> & extents,
	                          vk::Image layer_image, vk::ImageView layer_view,
	                          vk::ImageView sensor_view,
	                          const frame_params & params);

	// VR-depth viz (decoder SBS, single image, GENERAL): green->red
	// over 0..2.046m, far past 1020 black. Null view records opaque
	// black. YCbCr views ride the immutable-sampler path (verdict
	// pattern: decoder-shared sampler, waitIdle'd rebuild).
	void record_virt(vk::raii::CommandBuffer & cmd, size_t image_index,
	                 const std::array<vk::Extent2D, 2> & extents,
	                 vk::ImageView virt_view, vk::Sampler virt_sampler,
	                 const wivrn::decoder::ycbcr_definition & ycbcr,
	                 const frame_params & params);

	// Release the acquired image (after submit, before endFrame).
	void release();

	xr::swapchain & swapchain()
	{
		return swapchain_;
	}

private:
	void init_pipelines(vk::Format format);
	void upload_viz_params(const frame_params & params);

	vk::raii::Device * device = nullptr;

	vk::raii::PipelineLayout pipeline_layout{nullptr};
	vk::raii::RenderPass renderpass{nullptr};
	vk::raii::Pipeline pipeline{nullptr}; // sensor viz (D16 false-color)
	vk::raii::Pipeline layer_pipeline{nullptr}; // composited R16F viz
	vk::raii::Pipeline verdict_pipeline{nullptr}; // VR-vs-passthrough viz
	vk::raii::DescriptorSetLayout set_layout{nullptr};
	vk::raii::DescriptorPool descriptor_pool{nullptr};
	vk::raii::DescriptorSet descriptor_set{nullptr};
	vk::raii::Sampler sampler{nullptr};
	buffer_allocation ubo;
	// Verdict-viz state (verdict R8 + layer R16F + UBO): own layout
	// (two image bindings), pool and set; shares pipeline_layout push
	// shape (eye) and renderpass/targets above.
	vk::raii::DescriptorSetLayout verdict_viz_layout{nullptr};
	vk::raii::DescriptorPool verdict_viz_pool{nullptr};
	vk::raii::DescriptorSet verdict_viz_set{nullptr};
	vk::raii::PipelineLayout verdict_viz_pipeline_layout{nullptr};
	// Raster-vs-sensor state (layer R16F + sensor D16 + UBO): own layout
	// (two image bindings), pool, set, layout and pipeline; shares the
	// eye-push shape and renderpass/targets above (verdict-viz pattern).
	vk::raii::DescriptorSetLayout raster_sensor_layout{nullptr};
	vk::raii::DescriptorPool raster_sensor_pool{nullptr};
	vk::raii::DescriptorSet raster_sensor_set{nullptr};
	vk::raii::PipelineLayout raster_sensor_pipeline_layout{nullptr};
	vk::raii::Pipeline raster_sensor_pipeline{nullptr};
	// VR-depth viz state (decoder SBS sampler2D + UBO): own layout +
	// pool + set + pipeline. Converted views bake the decoder's live
	// sampler as immutable (VUID-02738, verdict pattern); plain views
	// keep the mutable layout. waitIdle'd rebuild, rare. Declaration
	// order matters: layout outlives sets (reverse destruction).
	// Non-owning alias: refreshed on every rebuild before any bind;
	// a decoder-side sampler recreation rebuilds before the next bind
	// (handle comparison in ensure_virt_layout).
	vk::Sampler virt_immutable_src{nullptr};
	wivrn::decoder::ycbcr_definition virt_ycbcr_built{};
	void create_virt_state(const wivrn::decoder::ycbcr_definition & def, vk::Sampler decoder_sampler);
	void ensure_virt_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler decoder_sampler);
	vk::raii::DescriptorSetLayout virt_layout{nullptr};
	vk::raii::DescriptorPool virt_pool{nullptr};
	vk::raii::DescriptorSet virt_set{nullptr};
	vk::raii::PipelineLayout virt_pipeline_layout{nullptr};
	vk::raii::Pipeline virt_pipeline{nullptr};

	xr::swapchain swapchain_;
	vk::Format format_ = vk::Format::eUndefined;
	vk::Extent2D extent_{0, 0};

	struct target
	{
		vk::raii::ImageView view{nullptr};
		vk::raii::Framebuffer framebuffer{nullptr};
	};
	// [image][eye], parallel to the swapchain image list.
	std::vector<std::array<target, 2>> targets;
};
