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
// black). Vr_raster overlays the composited raster ramp on the VR
// depth ramp wherever a window is present (shared ramp: same wall,
// different color = the paths disagree by that distance). All modes
// share the global opacity below.
enum class debug_viz_mode
{
	none = 0,
	sensor,
	vr_depth,
	layer,
	verdict,
	raster_vs_sensor,
	vr_raster,
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
		case debug_viz_mode::vr_raster:
			return "VR + Raster";
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

	// VR-depth viz with the raster ramp pasted on top wherever a window
	// is present (shared 0..2.046m ramp, far past 1020 black, no-window
	// pixels keep the VR color). Null virt view records opaque black;
	// null layer falls back to plain VR depth. YCbCr views ride the
	// immutable-sampler path (verdict pattern: handle-generation
	// sampler baked immutable, waitIdle'd rebuild).
	void record_vr_raster(vk::raii::CommandBuffer & cmd, size_t image_index,
	                      const std::array<vk::Extent2D, 2> & extents,
	                      vk::ImageView virt_view, vk::Sampler virt_sampler,
	                      const wivrn::decoder::ycbcr_definition & ycbcr,
	                      vk::Image layer_image, vk::ImageView layer_view,
	                      const frame_params & params);

	// VR-depth viz (decoder SBS, single image, GENERAL): green->red
	// over 0..2.046m, far past 1020 black. Null view records opaque
	// black. YCbCr views ride the immutable-sampler path (verdict
	// pattern: handle-generation sampler baked immutable, waitIdle'd
	// rebuild).
	void record_virt(vk::raii::CommandBuffer & cmd, size_t image_index,
	                 const std::array<vk::Extent2D, 2> & extents,
	                 vk::ImageView virt_view, vk::Sampler virt_sampler,
	                 const wivrn::decoder::ycbcr_definition & ycbcr,
	                 const frame_params & params);

	// Release the acquired image (after submit, before endFrame).
	void release();

	// 6DOF sensor reprojection record (debug viz): forward-warps the
	// native sensor texels (one mesh vertex per texel) into the display
	// eye at predicted time, false-colored like the direct tap
	// (NEAREST throughout: true outcomes, no invented depths).
	// Missing/no-data and streak-discarded pixels stay cleared = far
	// (no hole-filling). Renders into the top-left half-extent quadrant
	// of the acquired image (half res = the future compare working
	// res); the caller submits that quadrant with the predicted eye
	// pose/fov. The warp runs per eye (sensor layer = eye).
	struct reproject_params
	{
		// Display clip-from-world at predicted time, column-major.
		std::array<std::array<float, 16>, 2> disp_vp{};
		// World-from-sensor-eye at capture time (D_cap, full 6DOF),
		// column-major.
		std::array<std::array<float, 16>, 2> sens_mount{};
		// tan(angleLeft, angleRight, angleDown, angleUp) per eye.
		std::array<std::array<float, 4>, 2> sens_tan{};
		// Sensor near, far (meters) per eye.
		std::array<std::array<float, 2>, 2> sens_nf{};
		// Native sensor grid the mesh is keyed on (rebuild on change).
		uint32_t sens_w = 0, sens_h = 0;
	};
	void record_sensor_reproject(vk::raii::CommandBuffer & cmd, size_t image_index,
	                             const std::array<vk::Extent2D, 2> & half_extents,
	                             vk::ImageView sensor_view,
	                             const reproject_params & warp,
	                             const frame_params & params);

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
	// VR+raster state (decoder SBS sampler2D + R16F layer + UBO): own
	// layout + pool + set + pipeline. Converted views bake the paired
	// handle's generation sampler as immutable (VUID-02738, verdict
	// pattern); plain views keep the mutable layout. waitIdle'd
	// rebuild, rare. Declaration order matters: layout outlives sets
	// (reverse destruction). Non-owning alias: refreshed on every
	// rebuild before any bind; a generation change rebuilds before the
	// next bind (handle comparison in ensure_vr_raster_layout).
	vk::Sampler vr_raster_immutable_src{nullptr};
	wivrn::decoder::ycbcr_definition vr_raster_ycbcr_built{};
	void create_vr_raster_state(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler);
	void ensure_vr_raster_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler);
	vk::raii::DescriptorSetLayout vr_raster_layout{nullptr};
	vk::raii::DescriptorPool vr_raster_pool{nullptr};
	vk::raii::DescriptorSet vr_raster_set{nullptr};
	vk::raii::PipelineLayout vr_raster_pipeline_layout{nullptr};
	vk::raii::Pipeline vr_raster_pipeline{nullptr};
	// VR-depth viz state (decoder SBS sampler2D + UBO): own layout +
	// pool + set + pipeline. Converted views bake the paired handle's
	// generation sampler as immutable (VUID-02738, verdict pattern);
	// plain views keep the mutable layout. waitIdle'd rebuild, rare.
	// Declaration order matters: layout outlives sets (reverse
	// destruction). Non-owning alias: refreshed on every rebuild before
	// any bind; a generation change rebuilds before the next bind
	// (handle comparison in ensure_virt_layout).
	vk::Sampler virt_immutable_src{nullptr};
	wivrn::decoder::ycbcr_definition virt_ycbcr_built{};
	void create_virt_state(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler);
	void ensure_virt_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler);
	vk::raii::DescriptorSetLayout virt_layout{nullptr};
	vk::raii::DescriptorPool virt_pool{nullptr};
	vk::raii::DescriptorSet virt_set{nullptr};
	vk::raii::PipelineLayout virt_pipeline_layout{nullptr};
	vk::raii::Pipeline virt_pipeline{nullptr};

	// 6DOF sensor warp state: own set layout (sensor sampler shared with
	// the direct tap, warp UBO for the vertex stage, frame-params UBO
	// for the fragment ramp), dedicated renderpass with a depth
	// attachment (nearest-wins for overlapping splats; framebuffer-sized
	// even though only the half-extent quadrant renders), per-eye
	// warp-UBO upload, and the native-texel grid mesh (rebuilt when the
	// sensor extent changes — in practice never, waitIdle'd like ensure).
	void init_warp();
	void ensure_warp_targets();
	void ensure_warp_mesh(uint32_t sens_w, uint32_t sens_h);
	vk::raii::DescriptorSetLayout warp_layout{nullptr};
	vk::raii::DescriptorPool warp_pool{nullptr};
	// One set per eye: each draw must read its own UBO slot (a single
	// set rewritten between draws would alias, and updating a set after
	// bind is invalid — all writes happen before any bind).
	std::vector<vk::raii::DescriptorSet> warp_sets;
	vk::raii::PipelineLayout warp_pipeline_layout{nullptr};
	vk::raii::RenderPass warp_renderpass{nullptr};
	vk::raii::Pipeline warp_pipeline{nullptr};
	buffer_allocation warp_ubo;
	image_allocation warp_depth_image;
	vk::raii::ImageView warp_depth_view{nullptr};
	uint32_t warp_mesh_w = 0, warp_mesh_h = 0;
	buffer_allocation warp_vertex_buffer;
	buffer_allocation warp_index_buffer;
	uint32_t warp_index_count = 0;

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
	struct warp_target
	{
		vk::raii::Framebuffer framebuffer{nullptr};
	};
	// [image][eye] warp framebuffers: share targets' color views, so
	// declared after targets (dies first). Parallel to targets.
	std::vector<std::array<warp_target, 2>> warp_targets;
};
