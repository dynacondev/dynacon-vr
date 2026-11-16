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
	// In-place vertex rewrite for dynamic meshes (skinned hands): same
	// vertex/index counts as the uploaded soup (topology is fixed),
	// new positions only. No waitIdle (buffers are never replaced),
	// flushed by the normal per-draw flush_upload path. No-op on
	// unknown key or size mismatch (caller falls back to set_mesh).
	void update_mesh_positions(const std::string & key, const std::vector<XrVector3f> & verts);
	// Model-space AABB for frustum skip (nullopt = unknown key: keep the
	// draw). Computed once at upload; the draws filter in stream.cpp
	// consults it before acquire, so fully-culled groups cost nothing.
	std::optional<std::pair<glm::vec3, glm::vec3>> mesh_bounds(const std::string & key) const;

	// Borrowed submitted-image array view (both layers) for the
	// merge pass: the just-recorded image's multiview color
	// view, sampled (not rendered) by the merge. Built for every
	// recorded image alongside the framebuffers, so any image a group
	// recorded this frame has one. Null when unknown (retired image
	// or never recorded): the caller must skip the input loudly,
	// never silently. Same borrowed lifetime as the mesh buffers
	// above (dies with reset_targets(), waitIdle'd upstream).
	vk::ImageView submitted_view_array(vk::Image image) const;

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
		size_t draws = 0;   // drawIndexed calls: silhouettes (one per MV
		                      // broadcast) plus exact-depth and volume draws
		float box_frac = 1;   // scissor-box area fraction (0-1) at working
		                      // res; 1 = fullscreen fallback. Diagnostic for
		                      // cull effectiveness (log only, no plot).
		// Common-layer depth stages (single 3d-depth-passthrough, tier-1
		// group only): geometry raster (hidden + bbox into the borrowed
		// merge intermediate), warp + hands MIN, rev raster + alpha build,
		// intermediate min-dilate, sensor compare write, layer add.
		// Zero when no depth in group.
		double warp_ms = 0;     // sensor warp + hands MIN (D domain)
		double geom_ms = 0;     // hidden + bbox geometry raster (linear)
		double alpha_ms = 0;    // rev raster + alpha build (win && !rev)
		double dilate_ms = 0;   // intermediate min-dilate H+V (half-res texels)
		double compare_ms = 0;  // sensor compare write (warped D vs dilated layer)
		double add_ms = 0;      // layer add (linearized sensor MIN into the layer)
	};

	// Record silhouettes + blur chain for both eyes into the layers of an
	// acquired swapchain image. No-op when no mesh is set. The image must
	// be unused (UNDEFINED is fine); it is left in COLOR_ATTACHMENT_OPTIMAL
	// for the compositor. extent sizes the intermediates (full); out_extent sizes
	// the submitted image and its passes (half for feathered groups: V
	// writes half pixels, the compositor upscales on submit; equal for
	// tier-0). Feather selects the tier as documented; rasterize=false
	// clears only.
	// Cutouts are world-space quads (two triangles each) punched crisp
	// through the finished mask at full alpha (mask 1 = reality), sharing
	// one view-only MVP.
	// Depth-volume layer (3d-depth-passthrough, tier-1 groups only):
	// the hidden proximity shapes and the volume bounding box raster
	// into the merge-owned common intermediate (R16F linear meters,
	// MIN accumulation) at the working res W; the warped sensor
	// (rotation warp H + time-bias trim, skinned hand depth MINed over
	// it, D domain) is tested against the min-dilated layer natively
	// (bias on the layer only) and writes the seed the group blurs;
	// the linearized sensor depth MINs back into the raw intermediate
	// (ungated: where the test failed sens >= layer, a no-op). Null =
	// no volume in this group: legacy silhouette raster runs instead.
	// GPU timestamps cover the depth chain on the shared pool
	// (stream-owned, 10 queries: base/geom/dilate/compare/add in the
	// merge + this record, verdict/apply/merge in the merge finish).
	struct volume_input
	{
		// Hidden proximity objects, server-MVP space at W (eye frame):
		// nearest linear into the borrowed intermediate (CULL_BACK).
		// Mesh keys must be uploaded (set_mesh); reverse/opacity ignored.
		std::vector<instance_draw> hidden_draws;
		// Volume bounding boxes, one MVP pair per depth volume at W
		// (server MVP mapping the renderer's internal closed unit box
		// [-1,1]^3 onto the volume bounds): far-wall linear into the
		// borrowed intermediate (CULL_FRONT). A dedicated closed box is
		// rasterized, never the object soup: soups are arbitrary glTF
		// (not guaranteed closed), which would break the far-wall trick.
		std::vector<std::array<glm::mat4, 2>> bbox_draws;
		// Borrowed common intermediate (merge-owned, R16F linear meters
		// at W): geometry MINs into it under LOAD after the merge base
		// write earlier in the same command buffer (pipeline barrier
		// between). May be null: geometry_ok is false then. The
		// renderer bakes its own framebuffer on the borrowed image
		// (keyed on the handle: a merge rebuild waitIdles upstream, so
		// a handle change safely rebuilds); reset_targets() drops it.
		vk::Image layer_image = nullptr;
		vk::ImageView layer_view = nullptr;
		vk::Extent2D layer_extent{0, 0};
		// Skinned hand draws, eye MVP space (display/sync frame): MINed over the warped
		// sensor canvas in the same pass (same near/far D encoding).
		// Always rastered: MIN is idempotent with sensor hands.
		std::vector<instance_draw> hand_draws;
		float bias = 0.06f; // sensor-space compare only: layer -= layer*bias
		// Intermediate min-dilate radius in *half-res* texels
		// (intermediate -> dilTemp, H+V box-min): expands the
		// known-geometry layer before the native sensor test, so
		// depth-sensor edge noise just outside a window/hidden
		// silhouette still reads as blocked. 0 = off, 8 = shader tap
		// max. Tuned live from the Passthrough tab (default 2).
		float dilate_texels = 2.f;
		vk::ImageView sensor_view = nullptr; // runtime D16 2-layer array
		// Sensor frame for the warp source: extent of the depth-camera
		// images (from the environment-depth provider).
		vk::Extent2D sensor_extent{0, 0};
		float near_z = 0.1f; // depth frustum the raster must match
		float far_z = 10.f;
		// Rotation warp: per-eye homography columns (eye e uses
		// [3*e .. 3*e+2]) mapping eye UV to sensor UV (old-main
		// rotation method + time-bias trim). Identity reproduces
		// the direct tap. Built once per frame on the CPU.
		std::array<glm::vec4, 6> warp_cols{};
		// GPU timestamp sink (stream-owned, 10 queries, shared with the
		// merge base/finish: this record stamps geom/dilate/compare/add
		// slots). Null disables stamping. Reset by the scene once per
		// frame before record_base(), so with several volume groups
		// the pool holds the LAST group only.
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

	// Hidden proximity record (3d-passthrough-hidden): hard-edged
	// near-field reveal, submit-merged into the single merged layer.
	// Dedicated path beside record()
	// (shared logic untouched): RP1 rasters the proximity shapes
	// (spheres + body cylinders) into a
	// depth field, RP2 rasters the silhouettes with a depth test against
	// it, so only in-radius fragments write mask. Quarter-res working +
	// submit; tier-0 direct raster (no blur/erode/volume/VR ever).
	// spheres/cylinders: one MVP pair per proximity volume (model =
	// translate(center) * scale(radius) for spheres, translate(column
	// center) * scale(r, depth/2, r) for the head-hung body columns),
	// stream-built so this stays timebase-free. Safety theorem baked
	// in: LESS accumulates the furthest proximity exit (max distance),
	// OVER-reveal (front-of-shape imprecision, accepted); GREATER here
	// would under-reveal (unsafe miss) and is forbidden.
	// The proximity field is a feather-owned D16 2-layer image (one
	// raster per frame): RP1 renders into it, RP2 tests the silhouettes
	// against it. Extent always equals the mask working extent (both
	// (re)created from the same quarter-res in the scene prewarm).
	// One draw per silhouette: which uploaded mesh, per-eye transforms,
	// opacity, and the full-res pixel bounds per eye (scaled to the mask
	// extent for a per-draw scissor — two MV begins total here: the
	// proximity field (spheres plus body cylinders) then the silhouettes,
	// one draw per item each).
	struct hidden_draw
	{
		std::string mesh;
		std::array<glm::mat4, 2> mvp;
		float opacity = 1;
		std::array<std::array<float, 4>, 2> px; // full-res x0,y0,x1,y1 per eye
	};
	struct hidden_cpu
	{
		double raster_ms = 0;
		int draws = 0;
	};
	// Ensure the hidden proximity field image for an extent (idempotent,
	// change-gated). Public so the scene can create it in the prewarm
	// (ahead of the mask record that renders into it); waitIdle inside
	// on rebuild. Returns true when (re)created: the sphere
	// framebuffers built in ensure_hidden bake the new views, so the
	// scene must reset_targets() on rebuild.
	bool ensure_hidden_field(vk::Extent2D extent);

	void record_hidden(vk::raii::CommandBuffer & cmd,
	                   vk::Image image,
	                   vk::Extent2D extent,
	                   const std::vector<hidden_draw> & draws,
	                   const std::vector<std::array<glm::mat4, 2>> & spheres,
	                   const std::vector<std::array<glm::mat4, 2>> & cylinders,
	                   hidden_cpu * cpu_stats,
	                   const std::array<vk::Extent2D, 2> & full_extents);

	bool has_mesh() const
	{
		return not meshes.empty();
	}

	// Drop cached framebuffers/views (swapchain recreated: old image
	// handles are dead and driver handle recycling could otherwise alias
	// a stale framebuffer onto a new image). Also drops the borrowed
	// common-layer framebuffer (merge-owned image: the merge rebuild
	// waitIdles upstream, and run_depth re-bakes on handle change).
	void reset_targets()
	{
		targets.clear();
		hidden_targets.clear();
		layer_fb.clear();
		layer_fb_image = VK_NULL_HANDLE;
		layer_fb_extent = {0, 0};
		// Drop the merged-triple framebuffer too: it bakes the (src,
		// tmp, extent) triple above, so a stale entry would reuse a
		// wrong-sized framebuffer. Creation-only (no waitIdle needed).
		triple_fb.clear();
		triple_src = VK_NULL_HANDLE;
		triple_tmp = VK_NULL_HANDLE;
		triple_extent = {0, 0};
	}

	// Ensure intermediates for an extent (idempotent, change-gated). Public
	// so the scene can warm targets at map arrival, before first draws;
	// record() calls it too (same gate, no double work).
	// with_intermediates=false skips (and drops) the blur/downsample
	// workspaces: tier-0 direct raster never touches them (it rasters
	// straight into the submitted images; reversed depth uses
	// output_depth, volume/VR use their own fixed targets). Saves ~17 MB
	// per tier-0 group at 2k masks. The gate keys on (extent, flag) so a
	// tier change rebuilds correctly.
	void ensure_targets(vk::Extent2D extent, bool with_intermediates = true);

private:
	vk::raii::Device & device;
	vk::raii::PhysicalDevice & physical_device;
	vk::Format format;
	std::shared_ptr<vk::raii::PipelineCache> pipe_cache;

	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> make_buffer(vk::DeviceSize size,
	                                                                vk::BufferUsageFlags usage,
	                                                                vk::MemoryPropertyFlags properties);

	vk::raii::RenderPass renderpass_mv{nullptr};
	vk::raii::RenderPass depth_renderpass_mv{nullptr};
	vk::raii::RenderPass raster_r8_renderpass_mv{nullptr};
	vk::raii::RenderPass raster_r8_depth_renderpass_mv{nullptr};
	vk::raii::PipelineLayout silhouette_mv_layout{nullptr};
	vk::raii::Pipeline pipeline_mv{nullptr};
	vk::raii::Pipeline depth_pipeline_mv{nullptr};
	vk::raii::Pipeline reverse_pipeline_mv{nullptr};
	vk::raii::Pipeline raster_r8_pipeline_mv{nullptr};
	vk::raii::Pipeline raster_r8_depth_pipeline_mv{nullptr};
	vk::raii::Pipeline raster_r8_reverse_pipeline_mv{nullptr};

	// All raster paths are multiview now (silhouette, hidden, cutout);
	// no per-eye raster renderpasses or pipelines remain. The marker
	// cutout re-begins the submit framebuffer_mv under
	// punch_load_renderpass (LOAD twin of the V-upscale pass) with
	// pipeline_mv at opacity 1.

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
	// (24-byte push). Erode runs only in triple subpasses 0/1 (blur-H in
	// 2, subpass index baked in); one pipe set shared by both triple use
	// sites (tier-1 [A,B] and tiered [h,eblur]).
	vk::raii::Pipeline triple_erode_0{nullptr};
	vk::raii::Pipeline triple_erode_1{nullptr};
	vk::raii::Pipeline triple_blur_h{nullptr};
	vk::raii::DescriptorSetLayout descriptor_layout{nullptr};
	vk::raii::DescriptorPool descriptor_pool{nullptr};
	// One set per pass: descriptor updates are host-side writes
	// that complete before submit, so every draw would otherwise read the
	// LAST update (all passes sampling the final image, i.e. unwritten
	// data). Distinct sets make each binding stable across the frame.
	// Index: 0 = blur-H, 1 = upscale merges (volume MAX — triple never
	// touches it), 2 = blur-V upscale; 4,5 = downsample levels
	// 2,3; 7 = erode-H, 8 = erode-V; 9 = standalone dilate-H,
	// 10 = standalone dilate-V. (3 and 6 are retired: the VR-cut MIN
	// merge and the removed VR slot — the alpha field needs no merge
	// set of its own.)
	// One set per pass (not
	// per eye): the multiview passes sample the array view, so no
	// per-eye slots exist. Erosion needs two sets because its H and V
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
		// Multiview sampling + render target: 2D-array view over both
		// layers and one layers=2 framebuffer. The _mv fullscreen
		// passes read via view_array (sampler2DArray indexed by
		// gl_ViewIndex) and render into blur_fb_mv.
		vk::raii::ImageView view_array{nullptr};
		vk::raii::Framebuffer blur_fb_mv{nullptr};
		// Multiview raster target: same array view under the raster MV pass.
		vk::raii::Framebuffer raster_fb_mv{nullptr};
		// Depth-tested raster companions (D16, same extent as the color
		// target above): depth image shared by both layers. Built for
		// raster sources only (A, down levels); blur-only targets (B,
		// eblur) leave these null.
		vk::raii::Image depth_image{nullptr};
		vk::raii::DeviceMemory depth_memory{nullptr};
		vk::raii::ImageView depth_view_array{nullptr};
		// Multiview depth companion: one layers=2 color+depth FB under the raster MV depth pass.
		vk::raii::Framebuffer raster_depth_fb_mv{nullptr};
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
	// Triple framebuffer extent (width/height): the cached framebuffer
	// binds a fixed size, so (src, tmp) alone is not a sufficient key.
	vk::Extent2D triple_extent{0, 0};
	// Downsample chain (D) and tier blur workspaces (E) at half, quarter
	// and eighth mask resolution (index 0..2). Tier k blurs at 1/k size
	// with the same fixed kernel, so tap density (and cost) stays flat
	// while feather-px grows; the band stays continuous by exact mapping.
	blur_target down_targets[3];
	blur_target eblur_targets[3];
	vk::Extent2D targets_extent{0, 0};
	// Whether the intermediates above currently exist (tier-0 groups
	// skip them — see ensure_targets; part of the rebuild gate).
	bool targets_has_intermediates = false;

	// Common depth layer (single 3d-depth-passthrough, tier-1 only).
	// The merge owns the R16F linear-meters intermediate at W (half of
	// full): stage-0 MINs window depths (merge record_base), the sensor
	// stage MINs hidden + volume bbox, then revealed sensor depths. MIN
	// is idempotent and order-independent within each stage. The sensor
	// test reads the min-dilated layer natively (bias on the layer
	// only); the exact, bias-free global compare against virt runs
	// merge-side afterwards (record_finish verdict + apply).
	// R16F linear meters, large-finite FAR clear (65504.0): the layer
	// carries meters, never D — no resample semantics ever exist.
	static constexpr vk::Format linear_depth_format = vk::Format::eR16Sfloat;
	// Layer FAR clear (large-finite: compares stay ordered, nothing
	// reads as absent-geometry except the explicit verdict gate).
	static constexpr float layer_far_clear = 65504.f;
	// Multiview geometry raster (geom_depth_mv.vert/frag, 128B MVP-pair
	// push): one draw broadcasts to both layers. Same CLEAR/STORE/
	// OPTIMAL R16F attachment as the warp/exact passes below (shared
	// renderpass, viewMask=both).
	struct geom_depth_push
	{
		std::array<glm::mat4, 2> mvp;
	};
	// Geometry raster: nearest linear (hidden, rev) or far-wall linear
	// (bbox) depending on cull mode; MIN blend over the FAR clear, so
	// the nearest surface wins order-independently. Cull-BACK keeps
	// single-sided windows correct from outside; closed meshes resolve
	// to the front wall either way. The bbox uses CULL_FRONT on the
	// internal closed unit box (see below) to resolve the far wall;
	// rev uses cull-NOTHING (legacy revD coverage: a cut counts from
	// either side).
	vk::raii::RenderPass geom_load_renderpass_mv{nullptr};
	vk::raii::PipelineLayout geom_raster_layout_mv{nullptr};
	vk::raii::Pipeline geom_back_pipeline_mv{nullptr};
	vk::raii::Pipeline geom_front_pipeline_mv{nullptr};
	vk::raii::Pipeline geom_none_pipeline_mv{nullptr};
	// Layer add (fullscreen _mv, 1 sampler: warped-sensor D): linearizes
	// the sample for the MIN-blend LOAD back into the raw intermediate.
	// Own 8B near/far push (no bias: UNGATED by design). Renders under
	// geom_load_renderpass_mv (same R16F LOAD/STORE/OPTIMAL MV shape).
	struct layer_add_push
	{
		float near_z = 0.1f;
		float far_z = 10.f;
	};
	static_assert(sizeof(layer_add_push) == 8);
	vk::raii::DescriptorSetLayout add_set_layout{nullptr};
	vk::raii::PipelineLayout add_pipe_layout{nullptr};
	vk::raii::DescriptorPool add_pool{nullptr};
	vk::raii::DescriptorSet add_set{nullptr};
	vk::raii::Pipeline add_pipeline_mv{nullptr};
	// Exact-depth raster: same shaders as the legacy volume path
	// (volume_mv.vert + volume_raster_mv.frag encode eye depth as
	// sensor-compatible D), nearest-point state (shared
	// exact_raster_pipeline_mv): cull-nothing + MIN blend over a far
	// (1.0) clear, so the closest surface wins order-independently.
	// Serves the hand MIN draws over the warped sensor canvas (same
	// pass, same pipe). Cull-nothing keeps single-sided hands visible
	// from either side; closed meshes still resolve to the front wall.
	// Sensor warp (fullscreen _mv, 1 sampler: warped sensor source).
	// Dedicated R16F CLEAR renderpass? No — reuses
	// volume_raster_renderpass_mv (same R16F CLEAR/STORE/OPTIMAL MV
	// shape). The pass CLEARs to far, draws the warp fullscreen,
	// then MIN-blends the skinned hand draws in the same pass
	// (exact_raster_pipeline_mv is renderpass-compatible).
	// Hands raster push (volume_mv.vert D-encoding, 136B MVP-pair +
	// near/far): same >128B note as the silhouette MV pushes; the ctor
	// throws loudly below when the device reports less.
	struct volume_mv_push
	{
		std::array<glm::mat4, 2> mvp;
		float near_z = 0.1f;
		float far_z = 10.f;
	};
	vk::raii::RenderPass volume_raster_renderpass_mv{nullptr};
	vk::raii::PipelineLayout volume_raster_layout_mv{nullptr};
	vk::raii::Pipeline exact_raster_pipeline_mv{nullptr};
	vk::raii::DescriptorSetLayout warp_set_layout{nullptr};
	vk::raii::PipelineLayout warp_pipe_layout{nullptr};
	vk::raii::DescriptorPool warp_pool{nullptr};
	vk::raii::DescriptorSet warp_set{nullptr};
	// Warp homography: per-eye 3x3 columns as 6 vec4s, 96B total
	// (under the 128B Vulkan minimum: no >128B throw needed),
	// pushed per warp from volume_input and pushed (named members,
	// never an array: push-constant arrays have driver-defined
	// packing). Layout matches mat4 columns exactly.
	struct warp_push
	{
		glm::vec4 col[6]{};
	};
	static_assert(sizeof(warp_push) == 96);
	vk::raii::Pipeline warp_pipeline_mv{nullptr};
	// Alpha build (fullscreen _mv, 2 samplers: sensor-verdict R8 +
	// rev linear R16F, no push). Renders into target_a.blur_fb_mv (the
	// tier-1 seed, R8 MV) under compare_renderpass_mv below (same R8 MV
	// shape; loadOp is not part of render-pass compatibility, so the
	// framebuffer is shared). CLEAR, not DONT_CARE: any texel the
	// alpha draw did not write must read back deterministic 0, never
	// undefined tile garbage. alpha = win && !revPresent (presence cut:
	// no nearer-test; the global virt verdict gates by depth later).
	vk::raii::DescriptorSetLayout alpha_set_layout{nullptr};
	vk::raii::PipelineLayout alpha_pipe_layout{nullptr};
	vk::raii::DescriptorPool alpha_pool{nullptr};
	vk::raii::DescriptorSet alpha_set{nullptr};
	vk::raii::Pipeline alpha_pipeline_mv{nullptr};
	// Reversed-depth scratch (R16F linear meters at W, 2 layers): the
	// group's own reversed draws rasterized nearest (far clear, MIN
	// blend, cull-nothing geom pipe), sampled by the alpha build.
	// Always re-rasterized (CLEAR covers the no-draws case).
	blur_target rev_depth;
	// Standalone min-dilate H/V (intermediate -> dilTemp, adjustable
	// radius in half-res texels): same erode fragment as the triple, one
	// pipeline serving both passes (identical create info, both run at
	// subpass 0) under
	// volume_raster_renderpass_mv (CLEAR R16F MV, fullscreen coverage).
	// Own descriptor sets 9/10 (host writes are last-write-wins per
	// CB, and the triple rewrites 7/8 later in the same CB).
	vk::raii::Pipeline dilate_pipeline{nullptr};
	// Sensor compare (fullscreen _mv, 2 samplers: warped sensor D +
	// dilated layer linear). Dedicated 16B near/far/bias push (bias on
	// the layer only now). Renders into volume_bin under
	// compare_renderpass_mv (same R8 CLEAR/STORE/OPTIMAL MV shape).
	// CLEAR, not DONT_CARE: unwritten texels read back 0.
	vk::raii::RenderPass compare_renderpass_mv{nullptr};
	vk::raii::DescriptorSetLayout compare_set_layout{nullptr};
	vk::raii::PipelineLayout compare_pipe_layout{nullptr};
	vk::raii::DescriptorPool compare_pool{nullptr};
	vk::raii::DescriptorSet compare_set{nullptr};
	// Compare push: near/far/bias, 16B total (named members, never an
	// array). Bias applies to the layer (distance-scaled).
	struct compare_push
	{
		// x = near (m), y = far (m, may be inf), z = bias (0..0.25),
		// w unused.
		glm::vec4 near_far_bias{0.1f, 10.f, 0.06f, 0.f};
	};
	static_assert(sizeof(compare_push) == 16);
	vk::raii::Pipeline compare_pipeline_mv{nullptr};
	// Linear sampler for the warp + add + compare sensor taps (warped
	// sensor D): bilinear upscale of the working-res canvas; the binary
	// verdict is feathered by the shared blur chain below. The layer
	// tap in the compare uses the NEAREST sampler instead (exact min
	// values, no invented intermediates).
	vk::raii::Sampler volume_sampler_linear{nullptr};
	// RGBA8 LOAD twin of the submit pass: serves the debug cutout
	// punch only (re-begins the finished swapchain framebuffer; loadOp
	// is not part of render-pass compatibility). Tier-0 never merges,
	// so no MAX pipe exists here.
	vk::raii::RenderPass punch_load_renderpass{nullptr};
	blur_target sensorW; // R16F warped-sensor canvas (working res, 2 layers):
	// rotation-warped sensor (off-canvas baked to far, D domain) with
	// skinned hand depth MINed over it in the same pass. Own MV
	// framebuffer (raster_fb_mv) under volume_raster_renderpass_mv
	// (CLEAR R16F MV; the hands pipe is renderpass-compatible).
	blur_target dilTemp; // R16F min-dilated layer (working res, 2 layers):
	// intermediate H+V box-min (adjustable radius, half-res texels),
	// sampled by the sensor compare. H renders into blur_fb_mv, V back
	// into raster_fb_mv, both under volume_raster_renderpass_mv (CLEAR
	// R16F MV; fullscreen coverage makes CLEAR deterministic).
	blur_target volume_bin; // R8 verdict canvas (working res, 2 layers):
	// sensor-compare binary output, combined with the rev cut by the
	// alpha build into the seed (no presmooth: the shared blur chain
	// feathers the edge).
	bool volume_targets_ready = false;
	vk::Extent2D depth_working_extent{0, 0}; // display-space chain res (half of full)
	vk::Extent2D depth_sensor_extent{0, 0}; // sensor extent (warp source)
	// Closed unit-box mesh ([-1,1]^3, outward-CCW, non-indexed) baked
	// once for the far-wall bbox raster: the bbox draws carry MVP pairs
	// mapping this box onto the volume bounds (scene-built). Rasterized
	// CULL_FRONT so the far wall wins (dedicated path: object soups are
	// arbitrary glTF, never assumed closed).
	buffer_allocation bbox_verts;
	// Borrowed common-layer framebuffer (baked on the merge-owned image
	// above, keyed by handle): geometry + add re-begin it under LOAD.
	// Dropped in reset_targets() and rebuilt on handle change (the merge
	// rebuild waitIdles upstream, so a stale entry can never alias a
	// live image).
	vk::raii::Framebuffer layer_fb{nullptr};
	vk::Image layer_fb_image = VK_NULL_HANDLE;
	vk::Extent2D layer_fb_extent{0, 0};
	void ensure_depth_targets(vk::Extent2D working, vk::Extent2D sensor);
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
		uint32_t vertex_count = 0;
		vk::raii::Buffer staging_buffer{nullptr};
		vk::raii::DeviceMemory staging_memory{nullptr};
		vk::DeviceSize staging_size = 0;
		struct upload_sizes
		{
			vk::DeviceSize vertex_bytes = 0;
			vk::DeviceSize index_bytes = 0;
		};
		std::optional<upload_sizes> pending_upload;
		glm::vec3 aabb_min{0}, aabb_max{0};
	};
	std::map<std::string, mesh_buffers> meshes;
	void flush_upload(vk::raii::CommandBuffer & cmd, mesh_buffers & mesh);

	// Marker window-cutout quads (two world-space triangles each, batched
	// per record with one upload + firstVertex draws, unmapped to flush).
	// Grown on demand; created lazily on first cutout use.
	buffer_allocation cutout_verts;

	// Hidden proximity path (record_hidden): own renderpasses (sphere
	// depth-only + color-with-depth-LOAD), own pipelines (proximity fill +
	// silhouette-with-proximity-test, same mask_mv.vert/mask_mv.frag modules),
	// own unit-sphere + unit-cylinder verts. Created lazily on first record_hidden.
	vk::raii::RenderPass hidden_sphere_rp{nullptr};
	vk::raii::RenderPass hidden_color_rp{nullptr};
	vk::raii::Pipeline hidden_sphere_pipe{nullptr};
	vk::raii::Pipeline hidden_silhouette_pipe{nullptr};
	// (silhouette_mv_layout above, built in the ctor, serves the hidden
	// pipes too — identical 132B range, one layout for all MV mask pipes).
	buffer_allocation hidden_sphere_verts_;
	uint32_t hidden_sphere_vert_count = 0;
	// Body-column twin of the sphere verts: closed unit cylinder
	// (radius 1, y in [-1,1]), drawn with the same depth-only pipe.
	buffer_allocation hidden_cylinder_verts_;
	uint32_t hidden_cylinder_vert_count = 0;
	// Feather-owned hidden proximity field (D16 2-layer image + array view
	// the proximity framebuffer below bakes against; P4: one proximity raster
	// per frame). (Re)created by ensure_hidden_field (waitIdle'd scene
	// prewarm); reset_targets() on rebuild. output_depth_array_view is
	// declared with output_depth below.
	image_allocation hidden_sphere_depth_;
	vk::raii::ImageView hidden_sphere_view_array_{nullptr};
	vk::Extent2D hidden_sphere_extent_{0, 0};
	// Per-swapchain-image multiview framebuffers for the two hidden
	// passes (color + proximity-depth, layers=2 each); dropped by
	// reset_targets() with the rest (swapchain recreate = waitIdle'd
	// upstream).
	struct hidden_frame_targets
	{
		vk::Extent2D extent{0, 0};
		vk::raii::ImageView view_array{nullptr}; // color, both layers
		vk::raii::Framebuffer framebuffer{nullptr}; // color MV, layers=2
		vk::raii::Framebuffer sphere_framebuffer{nullptr}; // sphere MV, layers=2
	};
	std::unordered_map<VkImage, hidden_frame_targets> hidden_targets;
	void ensure_hidden(vk::Image image, vk::Extent2D extent);
	bool hidden_ready = false; // lazy RPs/pipes/proximity verts (first record_hidden)

	// Framebuffer + view per swapchain image (images cycle; entries for
	// retired swapchains are dropped via reset_targets()).
	struct frame_targets
	{
		vk::Extent2D extent;
		// Multiview submit target: array view (both layers) + layers=2
		// framebuffer under blur_renderpass_mv. All output is multiview
		// (raster, V upscale, cutout punch — the
		// cutout re-begins this framebuffer under
		// punch_load_renderpass).
		vk::raii::ImageView view_array{nullptr};
		vk::raii::Framebuffer framebuffer_mv{nullptr};
		// Depth-tested raster companion for tier-0 direct raster (color +
		// depth): one layers=2 color+depth FB under depth_renderpass_mv,
		// built lazily on first reversed use (has_depth); positive-only
		// targets stay lean.
		vk::raii::Framebuffer depth_framebuffer_mv{nullptr};
		bool has_depth = false;
		frame_targets(vk::Extent2D extent_,
		              bool has_depth_,
		              vk::raii::ImageView view_array_ = nullptr,
		              vk::raii::Framebuffer framebuffer_mv_ = nullptr) :
		        extent(extent_),
		        view_array(std::move(view_array_)),
		        framebuffer_mv(std::move(framebuffer_mv_)),
		        has_depth(has_depth_)
		{
		}
	};
	std::unordered_map<VkImage, frame_targets> targets;
	// Submitted-size D16 depth array view for the tier-0 MV depth raster:
	// cached by extent (swapchain-independent, survives reset_targets();
	// only a handful of quantized submit sizes ever exist).
	struct depth_entry
	{
		vk::raii::Image image{nullptr};
		vk::raii::DeviceMemory memory{nullptr};
		vk::raii::ImageView view_array{nullptr};
	};
	std::map<std::pair<uint32_t, uint32_t>, depth_entry> output_depth;
	vk::ImageView output_depth_array_view(vk::Extent2D extent);
	// D16 depth image + array view at an extent (depth-test-only raster:
	// never sampled, device-local). Backs output_depth above and the
	// intermediate raster companions in ensure_targets.
	depth_entry make_depth_entry(vk::Extent2D extent);
};
