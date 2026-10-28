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

// Feathered passthrough window mask.
//
// Per frame (both eyes): rasterize the calibrated mesh silhouette binary
// at half resolution into intermediate A, separable Gaussian blur H into
// B and V upscale into the submitted swapchain image (the sampler's
// bilinear step is the final upscale, free). The blur decouples feather
// width from shape resolution: opaque interiors stay pixel-exact while
// the band is genuinely smooth (robust to the compositor sampling with or
// without filtering). Fixed 5-tap kernel (sigma 2, tails truncated);
// feather-px maps to
// tap spread, recommended range 4-12 (see docs/configuration.md).
//
// Feathering is internal-only: a separable box-min erosion runs on the
// blur input, so the blur band lands inside the silhouette instead of
// straddling it — exact inset for any feather width (two 1D min passes
// compose a (2r+1)^2 box, r = ceil bleed). Tier-0 hard edges skip
// erosion (already exact); reversed cuts erode consistently with fills
// (coverage erosion grows holes, shrinks fills).
//
// Swapchain images only allow COLOR_ATTACHMENT output, which is why the
// silhouette is a raster pass and not a buffer copy.

#include "render/passthrough_mesh.h"

#include <algorithm>
#include <array>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <map>
#include <memory>
#include <openxr/openxr.h>
#include <optional>
#include <string>
#include "vk/allocation.h"
#include <unordered_map>
#include <utility>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

class feather_mask_renderer
{
public:
	feather_mask_renderer(vk::raii::Device & device,
	                      vk::raii::PhysicalDevice & physical_device,
	                      vk::Format format);

	// Shared driver pipeline cache (one per device): every renderer
	// compiles identical pipelines, so the first build populates the
	// cache and the rest hit it (~85ms once per session, not per group).
	// Held by each renderer, so the cache outlives its users but dies
	// with the last one (no shutdown-order hazard). Render-thread only.
	static std::shared_ptr<vk::raii::PipelineCache> pipeline_cache(vk::raii::Device & device);
	// Force a first build now (e.g. at first video swapchain, before any
	// map exists) instead of paying driver compiles on the first visible
	// frame. A throwaway renderer populates the cache above.
	static void warm_pipelines(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device, vk::Format format);

	// (Re)upload one object's mesh buffers under its key. Synchronous
	// (device idle on replace); call when idle, e.g. on map change, not
	// per frame. Empty soup erases the entry.
	void set_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup);
	void remove_mesh(const std::string & key);
	// Model-space AABB for frustum skip (nullopt = unknown key: keep the
	// draw). Computed once at upload; the draws filter in stream.cpp
	// consults it before acquire, so fully-culled groups cost nothing.
	std::optional<std::pair<glm::vec3, glm::vec3>> mesh_bounds(const std::string & key) const;

	// One silhouette draw: which uploaded mesh, per-eye transforms, and
	// the first-acquisition fade opacity (presence 0..1 for both
	// polarities; the renderer pushes 1 - fade as the cut strength for
	// reversed draws). Reversed draws (3d-passthrough-reversed) cut the
	// mask instead of filling it, depth-tested against the positives so a
	// distant negative behind a nearer window never cuts it. Callers must
	// sort positives before reversed (the renderer resolves polarity in
	// two passes, but within-polarity order is draw order as before).
	// Split-union note: depth testing is per-group (per union), so a
	// reversed cut only resolves against the positives in its own union.
	// Cross-union occlusion is accepted as rare (documented tradeoff:
	// reversed meshes upload into every union but test independently).
	struct instance_draw
	{
		std::string mesh;
		std::array<glm::mat4, 2> mvp;
		float opacity = 1;
		bool reverse = false;
	};

	// Tier selection from feather-px, shared with the scene (which sizes
	// member swapchains from it: tier-0 submits full, tier-1 half, tiered
	// quarter). Single source of truth: record() and the scene must agree,
	// or sizing (and the merged tier-1 gate on out == half) misfires.
	static int tier_for_feather(float feather_px)
	{
		float f = feather_px;
		if (f <= 0)
			return 0;
		if (f > 128.f)
			f = 128.f;
		if (f <= 16)
			return 1;
		if (f <= 48)
			return 2;
		if (f <= 96)
			return 4;
		return 8;
	}

	// Per-record host-side cost breakdown (steady_clock, milliseconds).
	// Filled when record() gets a non-null out-pointer; all zeros on
	// early-out paths. Permanent diagnostics: the stream scene sums these
	// per frame into the Statistics plots and a periodic logcat line.
	// GPU cost per group comes from timestamp brackets the caller writes
	// around record() (one group = one blur stack = the GPU unit of
	// interest); CPU sections here identify the host-hot stage.
	struct mask_stage_cpu
	{
		double raster_ms = 0; // Stage 1 silhouettes (or tier-0 direct raster)
		double erode_ms = 0;  // separable box-min erosion (H+V, any radius)
		double blur_ms = 0;   // blur H+V (tier 1) or downsample+H+V (tiered)
		double punch_ms = 0;  // marker-window cutout punch
		double total_ms = 0;  // whole record() body, incl. target setup
		int tier = -1;        // selected tier, -1 = bypass/clear-only/empty
		size_t draws = 0;     // silhouette drawIndexed calls (both eyes)
		float box_frac = 1;   // scissor-box area fraction (0-1) at working
		                      // res; 1 = fullscreen fallback. Diagnostic for
		                      // cull effectiveness (log only, no plot).
		// Depth-volume stages (3d-depth-passthrough objects in this
		// group): furthest-point raster at 320px, virtual-depth SBS->D
		// conversion at 320px (when paired), strict sensor<min(volume,
		// virtual) flatten, upscale+MAX merge into the seed ahead of erode.
		// Zero when the group carries no volume draws.
		double volume_raster_ms = 0;
		double volume_virt_ms = 0;
		double volume_compare_ms = 0;
		double volume_merge_ms = 0;
		float volume_box_frac = 1; // merge-box area fraction of the seed
		// (0-1); drives the renderArea scissor economics, logged.
	};

	// Record silhouettes + blur chain for both eyes into the layers of an
	// acquired swapchain image. No-op when no mesh is set. The image must
	// be unused (UNDEFINED is fine); it is left in GENERAL for the
	// compositor. extent sizes the intermediates (full); out_extent sizes
	// the submitted image and its passes (half for feathered groups: V
	// writes half pixels, the compositor upscales on submit; equal for
	// tier-0). Feather selects the tier as documented; rasterize=false
	// clears only.
	// Cutouts are world-space quads (two triangles each) punched crisp
	// through the finished mask at full alpha (mask 1 = reality), sharing
	// one view-only MVP. Mask path only (the binary projected layer has
	// no alpha control to punch through).
	// Depth-volume layer (3d-depth-passthrough): furthest-point draws in
	// display-eye MVP space plus the runtime D16 sensor view they test
	// against (strict sensor < render, far hidden). Null = no volume in
	// this group: the seed passes through untouched. Non-null with draws
	// and a valid sensor view rasterizes at 320px, flattens to binary R8,
	// and upscales+MAXes into the tier seed AFTER the reversed MIN cuts
	// (the volume overlays everything, including VR cutthroughs) and
	// BEFORE erode/blur, so it feathers exactly like silhouettes.
	// Virtual depth (stream 3, SBS 640x320 luma): when valid, the decoded
	// SBS view converts to a sensor-compatible D canvas (per-eye halves,
	// meters->D with near_z/far_z) and the compare tests
	// sensor < min(volume, virtual) in D space. Null/invalid = legacy
	// sensor < volume behavior for that frame (strict: no stale virtual).
	struct volume_input
	{
		std::vector<instance_draw> draws; // depth objects, eye MVP space
		vk::ImageView sensor_view = nullptr; // runtime D16 2-layer array
		float near_z = 0.1f; // depth frustum the raster must match
		float far_z = 10.f;
		// Virtual depth SBS (decoded stream 3): 640x320 single image,
		// left eye x[0..320), right eye x[320..640), Y = 1cm/LSB with
		// 255 = far. Sampled with the decoder's own sampler (Y'CbCr);
		// green channel (least chroma crosstalk under 709) carries luma.
		vk::ImageView virt_view = nullptr;
		vk::Sampler virt_sampler = nullptr;
		bool virt_valid = false;
		// Rotation reprojection: per-eye homography columns (eye e uses
		// [3*e .. 3*e+2], w unused) mapping eye UV to sensor UV. Identity
		// reproduces the direct tap; computed once per frame on the CPU.
		std::array<glm::vec4, 6> remap_cols{};
		// B2 lateral correction: per-eye (T.xy, a, b) with
		// 1/depth = a*D + b (a, b duplicated per eye for vec4 packing).
		// T maps 1/depth (1/m) to sensor-UV shift; forward/radial +
		// depth-value terms excluded.
		std::array<glm::vec4, 2> remap_corr{};
		// GPU timestamp sink (stream-owned, 5 queries: raster/virt/
		// compare/merge boundaries). Null disables stamping. Reset per
		// record() call, so with several volume groups the pool holds the
		// LAST group only.
		vk::QueryPool timestamp_pool = nullptr;
	};
	void record(vk::raii::CommandBuffer & cmd,
	            vk::Image image,
	            vk::Extent2D extent,
	            const std::vector<instance_draw> & draws,
	            bool rasterize,
	            float feather_px,
	            const std::vector<std::array<glm::vec3, 6>> & cutouts,
	            const std::array<glm::mat4, 2> & cutout_mvp,
	            mask_stage_cpu * cpu_stats,
	            vk::Extent2D out_extent,
	            const volume_input * volume = nullptr);

	bool has_mesh() const
	{
		return not meshes.empty();
	}

	// Drop cached framebuffers/views (swapchain recreated: old image
	// handles are dead and driver handle recycling could otherwise alias
	// a stale framebuffer onto a new image).
	void reset_targets()
	{
		targets.clear();
	}

	// Ensure intermediates for an extent (idempotent, change-gated). Public
	// so the scene can warm targets at map arrival, before first draws;
	// record() calls it too (same gate, no double work).
	void ensure_targets(vk::Extent2D extent);

private:
	vk::raii::Device & device;
	vk::raii::PhysicalDevice & physical_device;
	vk::Format format;
	std::shared_ptr<vk::raii::PipelineCache> pipe_cache;

	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> make_buffer(vk::DeviceSize size,
	                                                                vk::BufferUsageFlags usage,
	                                                                vk::MemoryPropertyFlags properties);

	vk::raii::RenderPass renderpass{nullptr};
	vk::raii::PipelineLayout pipeline_layout{nullptr};
	vk::raii::Pipeline pipeline{nullptr};

	// R8 variants of the raster path for the single-channel blur
	// intermediates (same shaders: mask.frag replicates opacity to all
	// channels, so one body serves R8 (.r) and RGBA8 (.a) targets).
	// Renderpasses differ by format only; layout is shared (format-free).
	vk::raii::RenderPass raster_r8_renderpass{nullptr};
	vk::raii::Pipeline raster_r8_pipeline{nullptr};

	// Depth-tested raster variants for reversed draws
	// (3d-passthrough-reversed): same shaders/layout, plus a D16 depth
	// attachment. Positives write depth (GREATER: the shared projection
	// is reversed-Z, near = 1, far = 0) and overwrite color as
	// before; reversed draws test GREATER without writing depth and MIN-blend
	// (1 - fade) into color, so a negative cuts only where it is nearer
	// than the positives covering that pixel. Two format variants like
	// the plain raster path (RGBA8 submitted images, R8 intermediates).
	// Engaged per record() call only when a reversed draw exists;
	// positive-only frames keep the legacy passes bit-identically.
	vk::raii::RenderPass depth_renderpass{nullptr};
	vk::raii::Pipeline depth_pipeline{nullptr};
	vk::raii::Pipeline reverse_pipeline{nullptr};
	vk::raii::RenderPass raster_r8_depth_renderpass{nullptr};
	vk::raii::Pipeline raster_r8_depth_pipeline{nullptr};
	vk::raii::Pipeline raster_r8_reverse_pipeline{nullptr};

	// LOAD variant of the raster pass for the marker window cutout: same
	// single-attachment shape (framebuffer-compatible with the above), so
	// the finished blurred image can be re-begun and stamped with a crisp
	// full-alpha quad no blur pass touches.
	vk::raii::RenderPass cutout_renderpass{nullptr};

	// Separable Gaussian blur (fullscreen triangle, sampled input).
	// Two format variants: R8 for the intermediates, RGBA8 for the V
	// upscale into swapchain images. Same shaders/layouts (format-free);
	// only renderpass + pipeline objects split. Multiview (viewMask =
	// both eyes): one fullscreen draw broadcasts to both layers.
	vk::raii::RenderPass blur_renderpass_mv{nullptr};
	vk::raii::RenderPass blur_r8_renderpass_mv{nullptr};
	// Merged erode-H/erode-V/blur-H triple (one renderpass, 3
	// subpasses): sampler reads across subpasses keep tile residency
	// without input attachments (neighborhood taps need real
	// sampling). Tier-1 [A,B] and tiered [h,eblur] share it: same
	// formats, same load/store/initial/final pattern (loadOp is not
	// part of renderpass compatibility anyway).
	vk::raii::RenderPass triple_rp{nullptr};
	vk::raii::PipelineLayout blur_layout{nullptr};
	// Fullscreen pipelines (multiview: one draw broadcasts to both
	// eyes). The _mv fragment twins read each view's layer.
	vk::raii::Pipeline blur_pipeline_mv{nullptr};
	vk::raii::Pipeline blur_r8_pipeline_mv{nullptr};
	// Box-downsample pipeline (fullscreen triangle, shared layout: it only
	// reads the push block's src_texel prefix).
	vk::raii::Pipeline downsample_pipeline_mv{nullptr};
	// Separable box-min erosion for internal-only feathering at any
	// width: two 1D min passes (H then V, radius = ceil bleed) compose an
	// exact (2r+1)^2 box. Same fullscreen vertex shader/layout as blur
	// (24-byte push) and R8 standalone shape as the downsample pass, so
	// it records into any single-attachment R8 framebuffer (A/B blur_fbs,
	// eblur blur_fbs) with no new renderpasses or targets.
	// Triple-subpass pipes (subpass index is baked in: erode runs in
	// subpasses 0/1, blur-H in 2). Shared by both triple renderpasses.
	vk::raii::Pipeline triple_erode_0{nullptr};
	vk::raii::Pipeline triple_erode_1{nullptr};
	vk::raii::Pipeline triple_blur_h{nullptr};
	vk::raii::DescriptorSetLayout descriptor_layout{nullptr};
	vk::raii::DescriptorPool descriptor_pool{nullptr};
	// One set per (pass, eye): descriptor updates are host-side writes
	// that complete before submit, so every draw would otherwise read the
	// LAST update (all passes sampling the final image, i.e. unwritten
	// data). Distinct sets make each binding stable across the frame.
	// Index: 0 = blur-H, 1 = volume upscale+MAX merge, 2 = blur-V
	// upscale; 4,6,8 = downsample levels 1..3; 10 = erode-H, 12 = erode-V.
	// One set per pass (not
	// per eye): the multiview passes sample the array view, so eye-1
	// slots stay unused. Erosion needs two sets because its H and V
	// passes sample different images in the same command buffer (one
	// host-side write per set per frame, like the rest).
	std::vector<vk::raii::DescriptorSet> descriptor_sets;
	vk::raii::Sampler sampler{nullptr};
	// Point sampler for the erode passes only: erode taps land on exact
	// texel centers (integer offsets), where LINEAR == NEAREST
	// mathematically but executes the full bilinear path per tap.
	// Blur keeps the linear sampler (its fractional taps deliberately
	// exploit hardware bilinear); downsample keeps it too (its ±0.5
	// offsets would hit nearest tie-breaks).
	vk::raii::Sampler sampler_nearest{nullptr};

	// Blur intermediates (full mask resolution, own images: full usage
	// control, unlike swapchain images). Recreated when the extent changes.
	struct blur_target
	{
		vk::raii::Image image{nullptr};
		vk::raii::DeviceMemory memory{nullptr};
		std::vector<vk::raii::ImageView> views;
		std::vector<vk::raii::Framebuffer> raster_fbs;
		std::vector<vk::raii::Framebuffer> blur_fbs;
		// Multiview sampling + render target: 2D-array view over both
		// layers and one layers=2 framebuffer. The _mv fullscreen
		// passes read via view_array (sampler2DArray indexed by
		// gl_ViewIndex) and render into blur_fb_mv.
		vk::raii::ImageView view_array{nullptr};
		vk::raii::Framebuffer blur_fb_mv{nullptr};
		// Depth-tested raster companions (D16, same extent as the color
		// target above): depth image + views shared by both eyes' layers,
		// plus one color+depth framebuffer per eye for the seed raster.
		// Built for raster sources only (A, down levels); blur-only
		// targets (B, eblur) leave these null.
		vk::raii::Image depth_image{nullptr};
		vk::raii::DeviceMemory depth_memory{nullptr};
		std::vector<vk::raii::ImageView> depth_views;
		std::vector<vk::raii::Framebuffer> raster_depth_fbs;
	};
	blur_target target_a;
	blur_target target_b;
	// Merged-triple framebuffer (one renderpass, both layers): built
	// lazily on first record — the (src, tmp) image pair is fixed per
	// renderer (tier is fixed per group), so no rebuilds after that.
	// Creation-only (never destroys live targets), hence no waitIdle.
	vk::raii::Framebuffer triple_fb{nullptr};
	vk::Image triple_src = VK_NULL_HANDLE;
	vk::Image triple_tmp = VK_NULL_HANDLE;
	// Downsample chain (D) and tier blur workspaces (E) at half, quarter
	// and eighth mask resolution (index 0..2). Tier k blurs at 1/k size
	// with the same fixed kernel, so tap density (and cost) stays flat
	// while feather-px grows; the band stays continuous by exact mapping.
	blur_target down_targets[3];
	blur_target eblur_targets[3];
	vk::Extent2D targets_extent{0, 0};

	// Depth-volume chain (3d-depth-passthrough, fixed 320px working
	// size regardless of mask extent). Raster outputs R16 sensor-D,
	// compare flattens to 4-level coverage R8, presmooth H blurs at 320
	// with V folded into the merge upscale, merge upscales+MAXes into
	// the tier seed. Created once in the constructor (format-fixed);
	// targets are extent-fixed, so no rebuilds ever (unlike the mask
	// intermediates).
	static constexpr uint32_t volume_size = 320;
	static constexpr vk::Format volume_depth_format = vk::Format::eR16Unorm;
	// Presmooth radius (blur-spread units = texels at 320): stairs
	// smaller than ~2x this vanish into a stable gradient. 4 -> ~8px.
	static constexpr float volume_smooth_spread = 1.4f;
	struct volume_push
	{
		glm::mat4 mvp;
		float near_z = 0.1f;
		float far_z = 10.f;
	};
	vk::raii::RenderPass volume_raster_renderpass{nullptr};
	vk::raii::PipelineLayout volume_raster_layout{nullptr};
	vk::raii::Pipeline volume_raster_pipeline{nullptr};
	// Compare (fullscreen _mv, 2 samplers: sensor D16 + volume R16).
	// Dedicated CLEAR renderpass over volume_bin.blur_fb_mv (same R8 MV
	// shape as the blur intermediates; loadOp is not part of render-pass
	// compatibility, so the framebuffer is shared). CLEAR, not DONT_CARE:
	// the merge LINEAR-upscales the whole 320 canvas, and any texel the
	// compare draw did not write must read back deterministic 0, never
	// undefined tile garbage.
	vk::raii::RenderPass compare_renderpass_mv{nullptr};
	vk::raii::DescriptorSetLayout compare_set_layout{nullptr};
	vk::raii::PipelineLayout compare_pipe_layout{nullptr};
	vk::raii::DescriptorPool compare_pool{nullptr};
	vk::raii::DescriptorSet compare_set{nullptr};
	// Compare homography + B2 lateral correction: per-eye 3x3 columns as
	// 6 vec4s plus per-eye corr (T.xy, a, b), 8 vec4s / 128B total,
	// filled per volume record from volume_input and pushed (named
	// members, never an array: push-constant arrays have driver-defined
	// packing). Layout matches mat4 columns exactly.
	struct volume_homography
	{
		glm::vec4 col[6]{};
		glm::vec4 corr[2]{};
	};
	vk::raii::Pipeline compare_pipeline_mv{nullptr};
	// Linear sampler for the compare taps (sensor + volume depth):
	// deliberate coverage-AA. Bilinear ramps the depth across edge texels
	// so the d<v crossing slides sub-texel instead of stepping on the
	// 320-grid; the upscale+blur feather then smooths an already-smooth
	// outline. Transition verdicts stay within ~1 texel + feather.
	vk::raii::Sampler volume_sampler_linear{nullptr};
	// Virtual-depth SBS -> sensor-compatible D canvas (fullscreen MV
	// pass at 320px, meters->D with the volume near/far). Comparison
	// stays in D space (sensor < min(volume, virtual)), so the compare
	// push constants (already at the 128B minimum guarantee) are
	// untouched. Canvas cleared to far (1.0): unpaired frames leave the
	// min() a no-op. Written with plain STORE (every texel covered).
	struct volume_virt_push
	{
		float near_z = 0.1f;
		float far_z = 10.f;
	};
	vk::raii::RenderPass volume_virt_renderpass{nullptr};
	vk::raii::PipelineLayout volume_virt_layout{nullptr};
	vk::raii::Pipeline volume_virt_pipeline{nullptr};
	vk::raii::DescriptorSetLayout volume_virt_set_layout{nullptr};
	vk::raii::DescriptorPool volume_virt_pool{nullptr};
	vk::DescriptorSet volume_virt_set = nullptr;
	// Upscale+MAX merge into the tier seed (LOAD, preserves the
	// silhouette + reversed-cut result, sets 1 where the sensor is
	// inside). Reuses the blur shaders with spread 0 (the linear
	// sampler is the upscale) under MAX blending; framebuffers are the
	// seed's own MV target (loadOp is not part of compatibility).
	vk::raii::RenderPass volume_max_r8_renderpass{nullptr};
	vk::raii::Pipeline volume_max_r8_pipeline{nullptr};
	vk::raii::RenderPass volume_max_rgb8_renderpass{nullptr};
	vk::raii::Pipeline volume_max_rgb8_pipeline{nullptr};
	blur_target volume_depth; // R16 furthest-D canvas (320, 2 layers)
	blur_target volume_virt; // R16 virtual-depth D canvas (320, 2 layers):
	// SBS meters->D conversion output, cleared to far, sampled by the
	// compare pass alongside volume_depth. Own MV framebuffer
	// (volume_virt_fb_mv) under volume_virt_renderpass (CLEAR R16 MV).
	vk::raii::Framebuffer volume_virt_fb_mv{nullptr};
	blur_target volume_bin; // R8 coverage canvas (320, 2 layers):
	// compare output, presmoothed in place before the merge upscale
	blur_target volume_smooth; // R8 presmooth workspace (320, 2 layers):
	// H destination, read directly by the merge upscale (V folded in)
	bool volume_targets_ready = false;
	void ensure_volume_targets();
	void update_source(vk::ImageView view, uint32_t set, bool nearest = false,
	                     vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal);

	// One uploaded mesh per object in the group. Staging is per mesh and
	// shared-read (uploads are rare, map-change only); the pending copy
	// applies at the start of the next record().
	struct mesh_buffers
	{
		vk::raii::Buffer vertex_buffer{nullptr};
		vk::raii::DeviceMemory vertex_memory{nullptr};
		vk::raii::Buffer index_buffer{nullptr};
		vk::raii::DeviceMemory index_memory{nullptr};
		uint32_t index_count = 0;
		vk::raii::Buffer staging_buffer{nullptr};
		vk::raii::DeviceMemory staging_memory{nullptr};
		vk::DeviceSize staging_size = 0;
		std::optional<std::pair<vk::DeviceSize, vk::DeviceSize>> pending_upload;
		glm::vec3 aabb_min{0}, aabb_max{0};
	};
	std::map<std::string, mesh_buffers> meshes;
	void flush_upload(vk::raii::CommandBuffer & cmd, mesh_buffers & mesh);

	// Marker window-cutout quads (two world-space triangles each, batched
	// per record with one upload + firstVertex draws, unmapped to flush).
	// Grown on demand; created lazily on first cutout use.
	buffer_allocation cutout_verts;

	// Framebuffers + views per swapchain image (images cycle; entries for
	// retired swapchains are dropped via reset_targets()).
	struct frame_targets
	{
		vk::Extent2D extent;
		std::array<vk::raii::ImageView, 2> views;
		std::array<vk::raii::Framebuffer, 2> framebuffers;
		// Multiview V-upscale target: array view + layers=2
		// framebuffer under blur_renderpass_mv. Per-eye framebuffers
		// above stay (tier-0 raster, cutout punch).
		vk::raii::ImageView view_array{nullptr};
		vk::raii::Framebuffer framebuffer_mv{nullptr};
		// Depth-tested raster companions for tier-0 direct raster (color +
		// depth, under depth_renderpass): same color views as above plus a
		// cached submitted-size depth view (see output_depth below). Built
		// lazily on first reversed use (has_depth); positive-only targets
		// stay lean.
		std::array<vk::raii::Framebuffer, 2> depth_framebuffers;
		bool has_depth = false;
		frame_targets(vk::Extent2D extent_,
		              std::array<vk::raii::ImageView, 2> views_,
		              std::array<vk::raii::Framebuffer, 2> framebuffers_,
		              std::array<vk::raii::Framebuffer, 2> depth_framebuffers_,
		              bool has_depth_,
		              vk::raii::ImageView view_array_ = nullptr,
		              vk::raii::Framebuffer framebuffer_mv_ = nullptr) :
		        extent(extent_),
		        views(std::move(views_)),
		        framebuffers(std::move(framebuffers_)),
		        view_array(std::move(view_array_)),
		        framebuffer_mv(std::move(framebuffer_mv_)),
		        depth_framebuffers(std::move(depth_framebuffers_)),
		        has_depth(has_depth_)
		{
		}
	};
	std::unordered_map<VkImage, frame_targets> targets;
	// Submitted-size D16 depth views for the tier-0 depth raster above:
	// cached by extent (swapchain-independent, survives reset_targets();
	// only a handful of quantized submit sizes ever exist).
	struct depth_entry
	{
		vk::raii::Image image{nullptr};
		vk::raii::DeviceMemory memory{nullptr};
		std::vector<vk::raii::ImageView> views;
	};
	std::map<std::pair<uint32_t, uint32_t>, depth_entry> output_depth;
	vk::ImageView output_depth_view(vk::Extent2D extent, int eye);
	// D16 depth image + per-eye views at an extent (depth-test-only
	// raster: never sampled, device-local). Backs output_depth above and
	// the intermediate raster companions in ensure_targets.
	depth_entry make_depth_entry(vk::Extent2D extent);
};
