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
		double depth_ms = 0;  // sensor reveal MAX pass (3d-depth-passthrough, 0 when off)
		double total_ms = 0;  // whole record() body, incl. target setup
		int tier = -1;        // selected tier, -1 = bypass/clear-only/empty
		size_t draws = 0;   // drawIndexed calls: silhouettes (one per MV
		                      // broadcast)
		float box_frac = 1;   // scissor-box area fraction (0-1) at working
		                      // res; 1 = fullscreen fallback. Diagnostic for
		                      // cull effectiveness (log only, no plot).
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
	// one view-only MVP. Optional sensor reveal (depth_reveal above,
	// null = off) ORs the sensor test into the working alpha (binary on
	// the binary path, grey ramp under the AA when positives-only)
	// before the blur chain.
	// Sensor reveal for 3d-depth-passthrough (optional, null = off):
	// resampled sensor heightfield + merged copy (borrowed views,
	// same-frame use only — same discipline as submitted_view_array
	// above) plus the config bias. When set, record() runs the reveal
	// pass (single sensor<copy+bias test ORed into the working alpha)
	// right after the silhouette raster and before the
	// erode/blur chain, so the reveal feathers with the group. The
	// write-back twin (sensor depths into the shared layer) is a
	// union_merge pass the scene records after this returns.
	struct depth_reveal
	{
		vk::ImageView sensor_view = nullptr; // R16F array, sync-eye grid, hands in
		vk::ImageView copy_view = nullptr;   // R16F array, same grid
		float bias = 0;
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
	            const depth_reveal * reveal = nullptr);

	// Hidden proximity record (3d-passthrough-hidden): hard-edged
	// near-field reveal, submit-merged into the single merged layer.
	// Dedicated path beside record()
	// (shared logic untouched): RP1 rasters the proximity shapes
	// (spheres + body cylinders) into a
	// depth field, RP2 rasters the silhouettes with a depth test against
	// it, so only in-radius fragments write mask. Quarter-res working +
	// submit; tier-0 direct raster (no blur/erode ever).
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
	// output_depth). Saves ~17 MB
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
	// AA pipes (plane-coefficient mask_aa_mv.vert/frag, conservative
	// overestimate, ADD-saturate, backface-culled, no depth): RGBA for
	// tier-0 direct raster, R8 for tier-1 half-res raster.
	// Positives-only groups; reversed groups stay binary. Overestimate
	// emits the neighbour fragment at shared-edge pixels; 2x2
	// ordered-grid smooth coverage ADD-sums interior straddles to 1.0
	// (silhouette stays single-sided). Backfaces culled: a
	// behind-surface fragment covering a rim pixel would otherwise sum
	// the ramp to 1.0, so windows are front-side only (both-sides needs
	// the stencil two-pass follow-up). Same silhouette_mv_layout
	// (push-only) as the binary pipes. Null when the device lacks
	// VK_EXT_conservative_rasterization (record falls back to binary).
	vk::raii::Pipeline aa_pipeline_mv{nullptr};
	vk::raii::Pipeline aa_raster_r8_pipeline_mv{nullptr};
	// One-shot log state for the AA reversed fallback in record().
	// Render-thread only (record() records; the async CB only executes).
	bool aa_rev_logged_ = false;

	// All raster paths are multiview now (silhouette, hidden, cutout);
	// no per-eye raster renderpasses or pipelines remain. The marker
	// cutout re-begins the submit framebuffer_mv under
	// cutout_load_renderpass (LOAD twin of the V-upscale pass) with
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
	// Index: 0 = triple blur-H, 2 = blur-V upscale; 4,5 = downsample
	// levels 2,3; 7 = erode-H, 8 = erode-V. Sets 1, 3, 6 are unused
	// (kept so the pool size and set indices stay stable).
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

	// Depth reveal (3d-depth-passthrough): single test MAX-blended over
	// the working alpha (binary on the binary path, grey ramp under the
	// AA when positives-only) ahead of the erode/blur chain (the blend
	// unit does the OR — no work tap, which would be a feedback loop;
	// R8 MIN-blend precedent from the reversed raster proves the blend
	// path). R8 LOAD RP for the intermediates (tier-1 A, tiered seed —
	// the reveal blurs with the rest of the chain) plus a pipe under the
	// RGBA8 cutout LOAD twin for tier-0 submits (hard edge kept). Full
	// footprint every record (both bindings, Adreno partial-write
	// rule); ping-ponged sets (async mask CB may still read the other,
	// merge precedent); 4B bias push. Runs in every tier (seed
	// resolution on tiered: quarter for tier 2, eighth for tier 8).
	// Reversed cuts raster as usual first; the reveal MAXes over them
	// (physical visibility wins, hidden-wins-reversed precedent).
	vk::raii::RenderPass reveal_r8_rp{nullptr};
	vk::raii::DescriptorSetLayout reveal_layout{nullptr};
	vk::raii::DescriptorPool reveal_pool{nullptr};
	std::vector<vk::raii::DescriptorSet> reveal_sets;
	bool reveal_parity = false;
	vk::raii::PipelineLayout reveal_pipeline_layout{nullptr};
	vk::raii::Pipeline reveal_r8_pipeline{nullptr};   // MAX over R8 working target
	vk::raii::Pipeline reveal_rgba_pipeline{nullptr}; // MAX over RGBA8 submit (tier-0)
	// Cached LOAD framebuffer for the R8 working target under
	// reveal_r8_rp (creation-only, keyed like triple_fb below).
	vk::raii::Framebuffer reveal_fb{nullptr};
	vk::Image reveal_fb_src = VK_NULL_HANDLE;

	// Blur intermediates (full mask resolution, own images: full usage
	// control, unlike swapchain images). Recreated when the extent changes.
	struct blur_target
	{
		vk::raii::Image image{nullptr};
		vk::raii::DeviceMemory memory{nullptr};
		// Multiview sampling + render target: 2D-array view over both
		// layers and one layers=1 framebuffer (view mask routes both
		// eyes). The _mv fullscreen
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
		// Multiview depth companion: one layers=1 color+depth FB under the raster MV depth pass.
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

	// Marker cutout LOAD pass (RGBA8 multiview): LOAD twin of the
	// V-upscale pass above; the cutout punch re-begins the submit
	// framebuffer_mv under it (loadOp is not part of render-pass
	// compatibility, so the framebuffer is shared).
	vk::raii::RenderPass cutout_load_renderpass{nullptr};
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
		struct upload_sizes
		{
			vk::DeviceSize vertex_bytes = 0;
			vk::DeviceSize index_bytes = 0;
		};
		std::optional<upload_sizes> pending_upload;
		// AA de-indexed upload (3 verts per triangle + per-corner
		// bary): expanded at set_mesh time from the same soup, drawn
		// non-indexed. Dies with the entry on map-drop/erase (meshes
		// survive reset_targets(), like the indexed buffers above).
		vk::raii::Buffer aa_vertex_buffer{nullptr};
		vk::raii::DeviceMemory aa_vertex_memory{nullptr};
		uint32_t aa_vertex_count = 0;
		vk::raii::Buffer aa_staging_buffer{nullptr};
		vk::raii::DeviceMemory aa_staging_memory{nullptr};
		vk::DeviceSize aa_staging_size = 0;
		std::optional<vk::DeviceSize> aa_pending_upload;
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
	// pipes too — identical 140B range, one layout for all MV mask pipes).
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
	// passes (color + proximity-depth, layers=1 each, view mask routes
	// both eyes); dropped by
	// reset_targets() with the rest (swapchain recreate = waitIdle'd
	// upstream).
	struct hidden_frame_targets
	{
		vk::Extent2D extent{0, 0};
		vk::raii::ImageView view_array{nullptr}; // color, both layers
		vk::raii::Framebuffer framebuffer{nullptr}; // color MV, layers=1
		vk::raii::Framebuffer sphere_framebuffer{nullptr}; // sphere MV, layers=1
	};
	std::unordered_map<VkImage, hidden_frame_targets> hidden_targets;
	void ensure_hidden(vk::Image image, vk::Extent2D extent);
	bool hidden_ready = false; // lazy RPs/pipes/proximity verts (first record_hidden)

	// Framebuffer + view per swapchain image (images cycle; entries for
	// retired swapchains are dropped via reset_targets()).
	struct frame_targets
	{
		vk::Extent2D extent;
		// Multiview submit target: array view (both layers) + layers=1
		// framebuffer under blur_renderpass_mv. All output is multiview
		// (raster, V upscale, cutout punch — the
		// cutout re-begins this framebuffer under
		// cutout_load_renderpass).
		vk::raii::ImageView view_array{nullptr};
		vk::raii::Framebuffer framebuffer_mv{nullptr};
		// Depth-tested raster companion for tier-0 direct raster (color +
		// depth): one layers=1 color+depth FB under depth_renderpass_mv,
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
