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
	// per frame. Empty soup erases the entry. dot_pitch_m > 0 also bakes
	// the hidden billboard dot lattice (object-space centers + face
	// normals for this pitch) into an extra buffer, uploaded and gated
	// with the mesh; 0 skips it (non-hidden meshes). dot_pitch_m is in
	// OBJECT units: callers pass world_pitch / model_scale so the world
	// spacing matches the configured pitch at any model scale.
	void set_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup, float dot_pitch_m = 0);
	void remove_mesh(const std::string & key);
	// Model-space AABB for frustum skip (nullopt = unknown key: keep the
	// draw). Computed once at upload; the draws filter in stream.cpp
	// consults it before acquire, so fully-culled groups cost nothing.
	std::optional<std::pair<glm::vec3, glm::vec3>> mesh_bounds(const std::string & key) const;

	// Borrowed GPU mesh buffers for the dots overlay (binds these for its
	// own silhouette raster; feather keeps owning upload + eviction).
	// Nullopt when unknown OR an upload is still pending (no flush ran
	// ahead of this borrow — e.g. empty mask draws or a skipped mask
	// record). On flush frames the copy + barrier precede the borrower in
	// the same command buffer, so it binds fresh data immediately.
	struct borrowed_mesh
	{
		vk::Buffer vertex_buffer;
		vk::Buffer index_buffer;
		uint32_t index_count;
	};
	std::optional<borrowed_mesh> borrowed_mesh_buffers(const std::string & key) const;

	// Borrowed billboard dot instances for the dots overlay (per-dot
	// object-space center + face normal, baked at upload for the pitch
	// passed to set_mesh). Same pending-upload gate as the mesh buffers
	// above: nullopt when unknown, pending, or baked with dot_pitch 0.
	// Count 0 (degenerate lattice) also means skip the draw.
	struct borrowed_dots
	{
		vk::Buffer buffer;
		uint32_t count;
	};
	std::optional<borrowed_dots> borrowed_dot_buffers(const std::string & key) const;

	// Borrowed submitted-image array view (both layers) for the
	// union-B merge pass: the just-recorded image's multiview color
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
		// VR-occlusion stages (3d-passthrough windows in this group):
		// nearest-point object raster at depth resolution, eye-space
		// virt<object flatten, upscale+MIN merge into the seed ahead of
		// erode (and ahead of the volume MAX above, so a real reveal
		// there can still win where the sensor is nearer than VR).
		// Zero when the group carries no VR draws (no paired virtual
		// depth).
		double vr_raster_ms = 0;
		double vr_compare_ms = 0;
		double vr_merge_ms = 0;
		float vr_box_frac = 1; // merge-box area fraction of the seed
		// (0-1), same scissor economics as the volume box above.
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
	// one view-only MVP.
	// Depth-volume layer (3d-depth-passthrough): furthest-point draws in
	// display-eye MVP space plus the runtime D16 sensor view they test
	// against (strict sensor < render, far hidden). Null = no volume in
	// this group: the seed passes through untouched. Non-null with draws
	// and a valid sensor view rasterizes at 320px, flattens to binary R8,
	// and upscales+MAXes into the tier seed AFTER the reversed MIN cuts
	// (the volume overlays everything, including VR cutthroughs) and
	// BEFORE erode/blur, so it feathers exactly like silhouettes.
	// Virtual depth (stream 3, half-res SBS luma): when valid, the decoded
	// SBS view converts to a sensor-compatible D canvas (per-eye halves,
	// meters->D with near_z/far_z) and the compare tests
	// sensor < min(volume, virtual) in D space. Null/invalid = legacy
	// sensor < volume behavior for that frame (strict: no stale virtual).
	// The VR-occlusion path below samples its own high-res conversion,
	// so one SBS decode feeds two D canvases (320px sensor + high-res VR).
	// The VR-occlusion path converts the same SBS in eye space
	// (virt < object, no sensor remap) into its own high-res canvas.
	// GPU timestamps cover the volume chain only (timestamp_pool); VR
	// stages report CPU timings, their GPU cost folds into the mask span.
	struct volume_input
	{
		std::vector<instance_draw> draws; // depth objects, eye MVP space
		// VR occluders: 3d-passthrough positives in this group, eye MVP
		// space (same MVPs as the silhouette draws, reversed excluded).
		// Rasterized nearest-point at depth resolution and tested
		// eye-space virt < object; the seed MIN-cuts where VR wins.
		// Empty (or no paired virtual depth) = no VR occlusion in this
		// group.
		std::vector<instance_draw> vr_draws;
		vk::ImageView sensor_view = nullptr; // runtime D16 2-layer array
		float near_z = 0.1f; // depth frustum the raster must match
		float far_z = 10.f;
		// Virtual depth SBS (decoded stream 3): single image holding both
		// eyes side by side, left eye x[0..w/2), right eye x[w/2..w),
		// 10-bit 2mm/LSB luma clamped at 2.040m (code 1020, so
		// beyond-range VR reads as near and occludes); 1021+ decodes
		// as far (lossy-overshoot guard only). Sampled with the
		// decoder's own sampler (Y'CbCr); green channel (least chroma
		// crosstalk under 709) carries luma.
		vk::ImageView virt_view = nullptr;
		vk::Sampler virt_sampler = nullptr;
		bool virt_valid = false;
		// Per-eye virtual-depth resolution (SBS width/2 x height), from
		// the stream description. Sizes the high-res VR-occlusion
		// canvases; the sensor path stays at volume_size below.
		vk::Extent2D virt_eye_extent{0, 0};
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

	// Hidden proximity record (3d-passthrough-hidden): hard-edged
	// near-field reveal, submit-merged into the single union-B layer.
	// Dedicated path beside record()
	// (shared logic untouched): RP1 rasters the proximity spheres into a
	// depth field, RP2 rasters the silhouettes with a depth test against
	// it, so only in-radius fragments write mask. Quarter-res working +
	// submit (alert-grade, like the dots layer); tier-0 direct raster
	// (no blur/erode/volume/VR ever — bubbles never engage here).
	// spheres: one MVP pair per proximity volume (model =
	// translate(center) * scale(radius)), stream-built so this stays
	// timebase-free. Safety theorem baked in: LESS accumulates the
	// furthest sphere exit (max distance), so coverage can only
	// OVER-reveal (front-of-sphere imprecision, accepted); GREATER here
	// would under-reveal (unsafe miss) and is forbidden.
	// The sphere field is the dots-owned shared image (P4: one raster
	// per frame, not two): RP1 renders into field.view_array at
	// field.extent, RP2 tests against it, then the record transitions
	// the image to shader-readable for the dots color pass later in
	// the same command buffer. Extent MUST equal the quarter-res
	// working extent (the scene sizes mask, field and dots from one
	// quarter-res); mismatched views
	// would misalign the gate. Borrowed-framebuffer discipline: the
	// framebuffers built here reference the dots-owned views, so the
	// scene must reset_targets() before the dots renderer dies or
	// rebuilds.
	struct hidden_sphere_field
	{
		VkImage image = VK_NULL_HANDLE; // shared D16 2-layer field
		vk::Extent2D extent{0, 0}; // must equal the mask extent below
		vk::ImageView view_array{}; // both layers (mask builds its MV FB against this)
	};
	// One draw per silhouette: which uploaded mesh, per-eye transforms,
	// opacity, and the full-res pixel bounds per eye (scaled to the mask
	// extent for a per-draw scissor — the only tri-raster with one).
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
	void record_hidden(vk::raii::CommandBuffer & cmd,
	                   vk::Image image,
	                   vk::Extent2D extent,
	                   const std::vector<hidden_draw> & draws,
	                   const std::vector<std::array<glm::mat4, 2>> & spheres,
	                   hidden_cpu * cpu_stats,
	                   const std::array<vk::Extent2D, 2> & full_extents,
	                   const hidden_sphere_field & field);

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
		hidden_targets.clear();
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
	// Index: 0 = blur-H, 1 = upscale merges (blur-V source, volume MAX —
	// triple never touches it), 2 = blur-V upscale; 3 = volume
	// presmooth-H (must NOT be 0, which the triple blur-H overwrites);
	// 4,6,8 = downsample levels 1..3; 7 = VR MIN-merge upscale (split
	// from 1: the VR chain samples a different high-res image than the
	// volume chain in the same command buffer). 5 retired (was VR
	// presmooth, removed: smoothing reintroduced the rim).
	// 10 = erode-H, 12 = erode-V.
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
	// Whether the intermediates above currently exist (tier-0 groups
	// skip them — see ensure_targets; part of the rebuild gate).
	bool targets_has_intermediates = false;

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
	// VR-occlusion object raster: same shaders (volume.vert +
	// volume_raster.frag encode eye depth as sensor-compatible D) and
	// push layout as the volume raster above, but nearest-point state:
	// cull-nothing + MIN blend over a far (1.0) clear, so the closest
	// surface wins order-independently. Cull-nothing (not back) keeps
	// single-sided windows visible from either side; closed meshes still
	// resolve to the front wall. The compare gates on <0.9999, so the
	// far clear reads back as empty, never as geometry.
	vk::raii::Pipeline vr_raster_pipeline{nullptr};
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
	// VR-occlusion compare (fullscreen _mv, 2 samplers: object R16 +
	// virtual D canvas). Eye-space virt < object, no sensor remap, no
	// push constants (both inputs are already D). Renders into
	// vr_bin.blur_fb_mv under the shared R8 MV CLEAR pass (the volume
	// compare uses the separate 320px volume_bin), then presmooths +
	// MIN-merges below.
	vk::raii::DescriptorSetLayout vr_compare_set_layout{nullptr};
	vk::raii::PipelineLayout vr_compare_pipe_layout{nullptr};
	vk::raii::DescriptorPool vr_compare_pool{nullptr};
	vk::raii::DescriptorSet vr_compare_set{nullptr};
	vk::raii::Pipeline vr_compare_pipeline_mv{nullptr};
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
	// The VR path below renders the same conversion at depth resolution
	// into vr_virt; both share the pipeline/renderpass (size-independent),
	// only the framebuffer + target differ.
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
	vk::raii::DescriptorSet volume_virt_set{nullptr};
	// Upscale+MAX merge into the tier seed (LOAD, preserves the
	// silhouette + reversed-cut result, sets 1 where the sensor is
	// inside). Reuses the blur shaders with spread 0 (the linear
	// sampler is the upscale) under MAX blending; framebuffers are the
	// seed's own MV target (loadOp is not part of compatibility).
	vk::raii::RenderPass volume_max_r8_renderpass{nullptr};
	vk::raii::Pipeline volume_max_r8_pipeline{nullptr};
	vk::raii::RenderPass volume_max_rgb8_renderpass{nullptr};
	vk::raii::Pipeline volume_max_rgb8_pipeline{nullptr};
	// Upscale+MIN merge for VR occlusion (LOAD, preserves the silhouette
	// + reversed-cut result, clears where VR wins). Same blur shaders
	// with spread 0 and same renderpasses as the MAX merges above (loadOp
	// is not part of compatibility); only the blend op differs.
	vk::raii::Pipeline vr_min_r8_pipeline{nullptr};
	vk::raii::Pipeline vr_min_rgb8_pipeline{nullptr};
	blur_target volume_depth; // R16 furthest-D canvas (320, 2 layers)
	blur_target vr_depth; // R16 nearest-D canvas (depth-res, 2 layers):
	// object depth for the VR-occlusion compare above. Far (1.0) clear,
	// MIN blend, cull-nothing (see vr_raster_pipeline).
	blur_target volume_virt; // R16 virtual-depth D canvas (320, 2 layers):
	// SBS meters->D conversion output, cleared to far, sampled by the
	// volume compare pass alongside volume_depth. Own MV framebuffer
	// (volume_virt_fb_mv) under volume_virt_renderpass (CLEAR R16 MV).
	vk::raii::Framebuffer volume_virt_fb_mv{nullptr};
	blur_target volume_bin; // R8 coverage canvas (320, 2 layers):
	// compare output, presmoothed in place before the merge upscale
	blur_target volume_smooth; // R8 presmooth workspace (320, 2 layers):
	// H destination, read directly by the merge upscale (V folded in)
	bool volume_targets_ready = false;
	void ensure_volume_targets();
	// High-res VR-occlusion chain (3d-passthrough windows): same shapes
	// as the 320px volume chain above, sized at the per-eye depth
	// resolution (virt_eye_extent) and rebuilt if it changes. vr_depth
	// rasterizes the object nearest-D, vr_virt converts the SBS depth,
	// vr_bin holds the 4-level keep field merged straight (spread-0
	// upscale) into the seed — no presmooth: any VR smoothing would
	// bleed keep back up at the dilated rim, and feathering is the
	// seed chain's job. Created once per depth size (format-fixed).
	blur_target vr_virt; // R16 virtual-depth D canvas (depth-res, 2 layers)
	vk::raii::Framebuffer vr_virt_fb_mv{nullptr};
	blur_target vr_bin; // R8 coverage canvas (depth-res, 2 layers):
	// compare output, MIN-merged straight into the seed
	vk::Extent2D vr_extent{0, 0};
	bool vr_targets_ready = false;
	void ensure_vr_targets(vk::Extent2D eye);
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
		// Baked billboard dots (hidden overlay): per-dot center + face
		// normal, uploaded with the mesh above when set_mesh got a pitch.
		vk::raii::Buffer dot_buffer{nullptr};
		vk::raii::DeviceMemory dot_memory{nullptr};
		uint32_t dot_count = 0;
		vk::raii::Buffer staging_buffer{nullptr};
		vk::raii::DeviceMemory staging_memory{nullptr};
		vk::DeviceSize staging_size = 0;
		struct upload_sizes
		{
			vk::DeviceSize vertex_bytes = 0;
			vk::DeviceSize index_bytes = 0;
			vk::DeviceSize dot_bytes = 0;
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
	// depth-only + color-with-depth-LOAD), own pipelines (sphere fill +
	// silhouette-with-proximity-test, same mask.vert/mask.frag modules),
	// own unit-sphere verts. Created lazily on first record_hidden.
	vk::raii::RenderPass hidden_sphere_rp{nullptr};
	vk::raii::RenderPass hidden_color_rp{nullptr};
	vk::raii::Pipeline hidden_sphere_pipe{nullptr};
	vk::raii::Pipeline hidden_silhouette_pipe{nullptr};
	// Dedicated layout for the hidden multiview pipes (132B MVP-pair
	// push range; the shared raster_push layout stays 68B for the
	// per-eye feather paths).
	vk::raii::PipelineLayout hidden_mv_layout{nullptr};
	buffer_allocation hidden_sphere_verts_;
	uint32_t hidden_sphere_vert_count = 0;
	// (The sphere depth images moved to the dots renderer — P4 shared
	// field. output_depth_view is declared with output_depth below.)
	// Per-swapchain-image multiview framebuffers for the two hidden
	// passes (color + sphere-depth, layers=2 each); dropped by
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
	void ensure_hidden(vk::Image image, vk::Extent2D extent, const hidden_sphere_field & field);
	bool hidden_ready = false; // lazy RPs/pipes/sphere verts (first record_hidden)

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
