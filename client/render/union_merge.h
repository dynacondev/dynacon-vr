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

// Mask merge: one submitted alpha layer for the flat passthrough
// feed. Per frame the compositor charges per layer (~10pp measured on
// Quest 3), while each feather group / the debug cutout group / the
// hidden group used to submit its own UNPREMULTIPLIED_ALPHA layer and
// union them with REPLACE-then-ADD dst-alpha blending. This pass
// MAX-combines every active contributor into a single full-res image
// (any-mask-reveals-wins), submitted once with REPLACE.
//
// MAX preserves the safety theorems: hidden-wins-reversed (hidden only
// adds coverage) and over-reveal (max >= every input). It differs from
// saturating-ADD only in overlap bands (0.6+0.6 saturated to 1.0, MAX
// gives 0.6) — accepted as a double-count bugfix.
//
// HARD CAP, DOCUMENTED: max_inputs = 8 contributor images. Past that
// the caller drops extras with a loud log (no fallback submit path —
// >8 contributors means a config the merge was never sized for, and
// dropping coverage under-reveals, so the error must stay visible).
// Typical configs carry 1-3.
//
// 4B FRAG-only count push (uniform early-out over the static taps:
// unused slots still re-bind an active input's view, but taps past
// the count never execute, so cost scales with real inputs; vert
// declares nothing, so no VERT|FRAG mismatch trap), no derivatives,
// no depth, no mat3/ternaries (Adreno rule). Own swapchain, render
// thread only.

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "decoder/decoder.h"
#include "render/passthrough_mesh.h"
#include "vk/allocation.h"
#include "xr/swapchain.h"

namespace xr
{
class instance;
class session;
} // namespace xr

class union_merge_renderer
{
public:
	// Hard input cap (see above): more contributors than this are
	// dropped by the caller with an error, never merged, never
	// submitted on a fallback path. Typical configs carry 1-3.
	// 8 is the static sampler-binding count in
	// union_merge_mv.frag.glsl (src0..src7, no descriptor indexing
	// per the Adreno rule): raising max_inputs requires adding
	// matching srcN shader bindings + layout slots in lockstep.
	static constexpr uint32_t max_inputs = 8;

	// One merge input: a just-recorded contributor image plus its
	// multiview array view (both layers, BORROWED from the owning
	// feather renderer — same-frame use only, never cached here).
	// extent is the contributor image size: when every input matches
	// the merge extent the bilinear upscale is an exact 1:1 copy, so
	// the merge takes the NEAREST path (1 tap instead of 4). Zero
	// extent (unknown) forces the safe LINEAR path.
	// All contributors end COLOR_ATTACHMENT_OPTIMAL and transition to
	// SHADER_READ_ONLY for the merge; every next record re-begins
	// UNDEFINED+CLEAR, so no transition back is needed.
	struct input
	{
		vk::Image image;
		vk::ImageView view_array;
		vk::Extent2D extent{0, 0};
	};

	// One common-layer draw: which uploaded mesh plus per-eye MVPs on
	// the server color-eye timebase (same matrices as the silhouette
	// raster). Positives only (3d-passthrough windows): reversed cuts
	// stay per-group alpha, hidden/debug write nothing.
	struct geom_draw
	{
		std::string mesh;
		std::array<glm::mat4, 2> mvp;
	};

	// Virtual depth for the global verdict: decoded stream-3 SBS frame
	// (single image, left eye x[0..w/2)), handle-pinned view +
	// sampler, set every frame from the paired blit handle. The sampler
	// is the handle's own format generation (never the decoder's live
	// one: decoders renegotiate, and a stale view sampled through a
	// destroyed conversion segfaults Adreno in updateDescriptorSets).
	// valid=false
	// skips the verdict pass + barrier and applies merged directly
	// (mask-only finish, pixel-identical to a cleared-to-1 verdict).
	// ycbcr carries the view's
	// conversion definition (Android AHB depth); the verdict layout
	// shares the handle's generation sampler as its immutable sampler
	// (VUID-VkWriteDescriptorSet-descriptorType-02738).
	// foveation carries the SHARED color runs (identical tables:
	// same source size, center and transport as color on every
	// depth-carrying warped frame — no separate depth params ride
	// the wire). render_wh is the warp grid size (requested render
	// eye size); packed_wh is the per-eye packed size (from the
	// decoded SBS extent). The CPU expands runs to a direct
	// packed-index map per full-res merge pixel (merge->render scale
	// baked in) into the SSBO every verdict frame.
	struct virt_input
	{
		vk::ImageView view = nullptr;
		vk::Sampler sampler = nullptr;
		bool valid = false;
		wivrn::decoder::ycbcr_definition ycbcr{};
		std::array<wivrn::to_headset::foveation_parameter, 2> foveation{};
		uint16_t render_w = 0, render_h = 0;
		uint16_t packed_w = 0, packed_h = 0;
	};

	explicit union_merge_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device);

	// Shared driver pipeline cache (one per device, same object the
	// feather mask renderer uses). Render-thread only.
	static std::shared_ptr<vk::raii::PipelineCache> pipeline_cache(vk::raii::Device & device);
	// Throwaway compile so the first real merge pays ~nothing.
	static void warm_pipelines(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device, vk::Format format);

	// (Re)create swapchain + targets + pipeline for format/size
	// (full mask working res). waitIdle inside on rebuild; no-op
	// when unchanged. Returns true when rebuilt.
	bool ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h);

	bool ready() const
	{
		return (bool)swapchain_;
	}

	// Acquire + wait (100ms); -1 on timeout (pairing-safe: the
	// untouched acquisition is released inside, skip the frame).
	int acquire();

	// Release the acquired image (after submit — the compositor takes
	// the last released image).
	void release();

	// Record the MAX merge inputs: validate (1..max_inputs, valid
	// borrowed views; empty, oversized or null-view returns false) and
	// write the merge descriptor set (contributor images cycle
	// independently; two sets ping-pong by call parity so an
	// inline-fallback record never rewrites the set a still-flight
	// async command buffer reads). No renderpass work here: the merge
	// draw happens in record_finish's fold. CPU-only (um_merge_ms
	// times it in the scene).
	bool record(vk::raii::CommandBuffer & cmd,
	            vk::Extent2D extent, std::span<const input> inputs);

	// Verdict subpass content, own renderpass (R8, CLEAR-to-1 +
	// content-box draw): fused foveated-decode + exact compare
	// (common layer vs virtual depth, NEAREST/NEAREST). Empty box
	// skips the update + draw, identical to all-absent. Ends
	// SHADER_READ_ONLY for the apply tap (own barrier below).
	// No-op (no stamp) unless virt is valid with view + sampler.
	void record_verdict(vk::raii::CommandBuffer & cmd,
	                    const virt_input & virt_in,
	                    const vk::raii::QueryPool * pool = nullptr, uint32_t qafter = 0);

	// Global VR-occlusion finish: folded merge→apply (merge MAX into
	// the owned tile-local attachment, then merged alpha x verdict
	// into both eye layers of the acquired submit image). Mask-only
	// (!virt.valid) takes the mask pipeline (merged straight through,
	// pixel-identical to x1). RGB zero; the submit-time REPLACE
	// preserves dst color.
	bool record_finish(vk::raii::CommandBuffer & cmd,
	                   size_t image_index,
	                   vk::Extent2D extent,
	                   const virt_input & virt_in,
	                   const vk::raii::QueryPool * pool = nullptr, uint32_t qafter_merge = 0);

	// Stage 0 of the common depth layer (positives only): CLEAR the
	// R16F intermediate (large-finite FAR) then MIN-blend every window
	// draw with the geometry pipeline (linear meters out, near-wins,
	// CULL_NONE like the silhouette raster). Runs before the feather
	// groups record in the same command buffer; the verdict samples
	// the layer after a barrier in record_finish (split-timing tap:
	// when pool != nullptr a TopOfPipe timestamp lands in qafter after
	// the layer barrier). The CLEAR runs even
	// with no draws (a LOAD of undefined contents would read as
	// geometry and wrongly gate the verdict — fail-open demands the
	// explicit FAR clear).
	void record_base(vk::raii::CommandBuffer & cmd, std::span<const geom_draw> draws,
	                 const vk::raii::QueryPool * pool = nullptr, uint32_t qafter = 0);

	// Sensor-compare copy (3d-depth-passthrough): clone the shared layer,
	// MIN-raster every placed hidden occluder (all placed, no proximity
	// gate — occlusion is physical), MIN-dilate the copy by dilate_texels
	// (known-occluder widening only; the volume back wall below stays
	// exact), then MIN-raster every depth-volume back wall (CULL_FRONT
	// pipe). Single test downstream: sensor < copy + bias. Recorded
	// once per frame after record_base and before the feather loop (same
	// command buffer; barriers inside order every stage). Ends
	// SHADER_READ for the feather compare tap. No-op (false) when the
	// copy target was never built.
	bool record_copy(vk::raii::CommandBuffer & cmd, std::span<const geom_draw> occluders,
	                 std::span<const geom_draw> volumes, int dilate_texels);

	// (Re)upload one window soup under its key. Synchronous (device
	// idle on replace); call when idle, e.g. on map change, not per
	// frame. Empty soup erases the entry.
	void set_layer_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup);
	void remove_layer_mesh(const std::string & key);
	// Model-space AABB for the scene-side frustum skip (nullopt =
	// unknown key: keep the draw). Computed once at upload.
	std::optional<std::pair<glm::vec3, glm::vec3>> layer_mesh_bounds(const std::string & key) const;
	// Copy-occluder (hidden, all placed) and volume (depth back wall)
	// mesh stores: same synchronous upload discipline as the layer store
	// above, separate namespaces (keys need not be distinct across stores).
	void set_occluder_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup);
	void remove_occluder_mesh(const std::string & key);
	std::optional<std::pair<glm::vec3, glm::vec3>> occluder_mesh_bounds(const std::string & key) const;
	void set_volume_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup);
	void remove_volume_mesh(const std::string & key);
	std::optional<std::pair<glm::vec3, glm::vec3>> volume_mesh_bounds(const std::string & key) const;

	xr::swapchain & swapchain()
	{
		return swapchain_;
	}

	// Borrowed debug-viz views of the owned common-layer and verdict
	// targets (2-layer arrays; layer + verdict live at half res now). Raw handles, same-frame
	// use only; null when never built. Images end SHADER_READ_ONLY
	// after the mask-phase barriers, so the viz pass needs only an
	// execution barrier on the async path (same-queue ordering).
	vk::Image layer_image() const
	{
		if (*layer_.image != vk::Image{})
			return *layer_.image;
		return {};
	}
	vk::ImageView layer_view_array() const
	{
		if (*layer_.view_array != vk::ImageView{})
			return *layer_.view_array;
		return {};
	}
	vk::Image verdict_image() const
	{
		if (*verdict_.image != vk::Image{})
			return *verdict_.image;
		return {};
	}
	vk::ImageView verdict_view_array() const
	{
		if (*verdict_.view_array != vk::ImageView{})
			return *verdict_.view_array;
		return {};
	}
	vk::ImageView copy_view_array() const
	{
		if (*copy_.view_array != vk::ImageView{})
			return *copy_.view_array;
		return {};
	}
	// Half-res working size (the layer raster + verdict target live
	// here; viz upsamples by sampling).
	vk::Extent2D debug_layer_extent() const
	{
		return half_extent_;
	}

private:
	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> make_buffer(vk::DeviceSize size,
	                                                                vk::BufferUsageFlags usage,
	                                                                vk::MemoryPropertyFlags properties);
	void flush_uploads(vk::raii::CommandBuffer & cmd);
	void init_pipelines(vk::Format format);
	struct mesh_buffers;
	// Shared synchronous upload into one of the mesh stores above.
	void set_mesh_in(std::map<std::string, mesh_buffers> & meshes, const std::string & key,
	                 const passthrough_mesh::triangle_soup & soup);

	vk::raii::Device * device = nullptr;
	vk::raii::PhysicalDevice * physical_device = nullptr;
	std::shared_ptr<vk::raii::PipelineCache> pipe_cache;

	vk::raii::PipelineLayout pipeline_layout{nullptr};
	// 8 static sampler2DArray bindings (see the shader): linear +
	// ClampToEdge, the bilinear tap is the half/quarter upscale.
	vk::raii::DescriptorSetLayout descriptor_layout{nullptr};
	vk::raii::DescriptorPool descriptor_pool{nullptr};
	// Ping-pong pair (see record()): rewritten per call, never while
	// a prior submission still reads the other.
	std::vector<vk::raii::DescriptorSet> descriptor_sets;
	bool set_parity = false;
	// Active merge input count for the uniform tap early-out push
	// (record() validates, record_finish() pushes before the merge
	// draw; same-frame use only, set on every successful record()).
	int merge_count_ = 0;
	vk::raii::Sampler sampler{nullptr};

	// Folded merge→apply renderpass (opt #8): one RP, fullscreen
	// passes as subpass 0 (merge) + subpass 1 (apply), tile-local
	// merged intermediate (triple precedent: GENERAL attachments
	// sampled across subpasses, subpass dependency replaces the
	// split-pass barrier). The verdict runs ahead of the fold in its
	// own RP (different size) and is sampled as a texture. Shaders,
	// layouts, sets and taps are unchanged — only the RP/FB/pipeline
	// shell folds. Subpass index is baked into each pipeline.
	vk::raii::RenderPass fold_rp{nullptr};
	// Per submit image (submit view varies): {merged, submit}
	// framebuffers, parallel to targets.
	std::vector<vk::raii::Framebuffer> fold_fbs;
	// Subpass pipelines: merge s0 + apply s1 (paired verdict tap) or
	// mask-apply s1 (merged straight through). Chosen per frame in
	// record_finish; no cross-call RP state (each function owns
	// complete renderpasses).
	vk::raii::Pipeline fold_merge_pipeline{nullptr};
	vk::raii::Pipeline fold_apply_pipeline{nullptr};
	vk::raii::Pipeline fold_maskapply_pipeline{nullptr};

	xr::swapchain swapchain_;
	vk::Format format_ = vk::Format::eUndefined;
	vk::Extent2D extent_{0, 0};

	struct target
	{
		vk::raii::ImageView view{nullptr}; // color array view (both eyes)
	};
	// [image], parallel to the swapchain image list (one multiview
	// view per image — both eyes broadcast in one draw; the folded
	// framebuffers live in fold_fbs below).
	std::vector<target> targets;

	// Common depth layer (R16F linear meters, uniform half res): one
	// owned 2-layer image + array view + layers=1 framebuffer (view mask
	// routes both eyes), rebuilt in ensure() alongside the swapchain
	// targets above. MIN-blended by record_base, sampled by the verdict.
	struct layer_target
	{
		vk::raii::Image image{nullptr};
		vk::raii::DeviceMemory memory{nullptr};
		vk::raii::ImageView view_array{nullptr};
		// Layer only: merged/verdict ride the per-image folded FBs.
		vk::raii::Framebuffer framebuffer{nullptr};
	};
	layer_target layer_;
	// Owned merged target (swapchain format, full merge res): record()
	// MAXes contributors here instead of the submit image, so the
	// verdict apply has a stable input.
	layer_target merged_;
	// Binary verdict (R8, uniform half res): fused foveated-decode +
	// compare output, upsampled 2x by the apply (LINEAR tent filter:
	// smooth ~2px reveal ramp, flat areas bit-identical).
	layer_target verdict_;
	// Full merge res (== extent_): merged target + submit image.
	vk::Extent2D layer_extent_{0, 0};
	// Uniform half working res (opt #U): the layer raster + verdict
	// loop + verdict target all run here (quarter pixels). The layer
	// and verdict are compared texel-to-texel in this domain; the
	// apply upsamples the verdict 2x (LINEAR tent ramp over the
	// exact NEAREST broadcast positions).
	// Always even (asserted in ensure: merge dims are 64-aligned).
	vk::Extent2D half_extent_{0, 0};

	// Geometry raster: R16F MIN-blend pipeline (CLEAR/STORE/OPTIMAL,
	// CULL_NONE like the silhouette raster) + VERT-only MVP-pair push
	// layout (fragment declares no push block: no offset trap).
	vk::raii::RenderPass geom_renderpass{nullptr};
	vk::raii::PipelineLayout geom_layout{nullptr};
	vk::raii::Pipeline geom_pipeline{nullptr};
	// Verdict (fused SBS decode + exact compare) in its own
	// single-attachment RP: half-res R8 (opt #U), CLEAR-to-1 + box draw
	// (opt #A), STORE + SHADER_READ final (viz samples it after the
	// frame). Fullscreen passes, no push, static sampler2DArray
	// bindings (Adreno rule, same as the merge shader).
	vk::raii::RenderPass verdict_rp{nullptr};
	// Verdict YCbCr immutable-sampler state (VUID-...-02738): the depth
	// SBS view carries a conversion, so binding 1 needs an immutable
	// sampler baked into the layout. The baked sampler is the paired
	// handle's own generation sampler (pinned by the handle: same
	// generation as the sampled view, always). A separately built
	// conversion proved unidentifiable to validation despite matching
	// parameters, so the layout shares the handle's object instead of
	// owning one. Rebuilt when the incoming definition OR the handle
	// sampler changes (renegotiation hands out a new generation;
	// waitIdle'd like ensure()). Plain views (has_conversion=false)
	// keep the mutable layout. Declaration order matters: layout
	// outlives sets (reverse destruction). Non-owning alias: refreshed
	// on every rebuild before any bind.
	vk::Sampler verdict_immutable_src{nullptr};
	wivrn::decoder::ycbcr_definition verdict_ycbcr_built{};
	// (Re)build the full verdict state for a conversion definition:
	// descriptor set layout (binding 1 immutable iff converted) +
	// ping-pong sets + pipeline layout + pipeline (a pipeline bakes
	// its set layout, so the two are never rebuilt apart).
	// Caller guarantees no in-flight readers: fresh object, or a
	// waitIdle'd rebuild.
	void create_verdict_state(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler);
	void ensure_verdict_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler);
	vk::raii::DescriptorSetLayout verdict_descriptor_layout{nullptr};
	vk::raii::DescriptorPool verdict_descriptor_pool{nullptr};
	std::vector<vk::raii::DescriptorSet> verdict_descriptor_sets;
	bool verdict_set_parity = false;
	vk::raii::PipelineLayout verdict_layout{nullptr};
	// Verdict pipeline (rebuilt with the layout above, bound to the
	// standalone verdict RP, subpass 0).
	vk::raii::Pipeline verdict_pipeline{nullptr};
	// Verdict content box (opt #A): conservative both-eyes pixel union
	// of the layer draws, computed in record_base from the same
	// meshes/MVPs it rasterizes (mirrors the frustum-skip projections
	// + feather's pass_scissor). The verdict subpass clears to
	// reveal-1 and draws only this box — outside it the layer is
	// CLEAR-far by construction, so reveal-1 is exact. Empty (no draws
	// or all skipped) skips the verdict draw entirely.
	vk::Rect2D verdict_box_{{0, 0}, {0, 0}};
	bool verdict_box_empty_ = true;
	// Foveation index maps (SSBO pair, ping-ponged with the verdict
	// sets): per-eye packed-texel index per half-res loop pixel,
	// CPU-expanded from the shared foveation runs with the
	// merge->render scale baked in. Layout: [x0(W)] [x1(W)] [y0(H)]
	// [y1(H)] uint32s, W/H = half merge extent. Host-visible,
	// rewritten
	// every verdict frame (depth_overlay UBO discipline: memcpy +
	// vmaFlushAllocation under the same waitIdle'd ownership as the
	// targets); ping-ponged so an inline record never rewrites what
	// an in-flight async verdict still reads.
	buffer_allocation verdict_tables[2];
	vk::DeviceSize verdict_tables_bytes = 0;
	// Expanded index-map scratch (capacity persists after the first
	// verdict frame): avoids a ~2*(W+H) uint32 heap alloc per frame.
	std::vector<uint32_t> verdict_map_scratch;
	// Sensor-compare copy (3d-depth-passthrough, R16F half res): clone
	// of the shared layer plus hidden occluders, MIN-dilated, plus
	// volume back walls. Pass FBs ride the LOAD twin of the geom RP
	// (raster) and the dilate RP (fullscreen H+V). Rebuilt in ensure()
	// alongside layer_/verdict_ (same waitIdle).
	layer_target copy_;
	layer_target copy_tmp_;
	vk::raii::Framebuffer copy_load_fb{nullptr};     // copy under geom_load_rp
	vk::raii::Framebuffer copy_dilate_fb{nullptr};   // copy under dilate_rp (V target)
	vk::raii::Framebuffer copy_tmp_dilate_fb{nullptr}; // tmp under dilate_rp (H target)
	// LOAD twin of geom_renderpass (same R16F attachment, LOAD/STORE):
	// loadOp is not part of render-pass compatibility, so geom_pipeline
	// binds here too (feather cutout precedent).
	vk::raii::RenderPass geom_load_rp{nullptr};
	// Back-wall pipe (CULL_FRONT, otherwise geom-identical): volume
	// interiors resolve to the exit wall. Baked against geom_renderpass,
	// bound under the compatible geom_load_rp.
	vk::raii::Pipeline backwall_pipeline{nullptr};
	// MIN-dilate fullscreen state (R16F, erode1d_mv.frag + blur.vert,
	// 24B push shared with the feather erode layout shape). Two sets
	// ping-ponged by call parity (async mask CB may still read the
	// other, merge precedent).
	vk::raii::RenderPass dilate_rp{nullptr};
	vk::raii::DescriptorSetLayout dilate_layout{nullptr};
	vk::raii::DescriptorPool dilate_pool{nullptr};
	std::vector<vk::raii::DescriptorSet> dilate_sets;
	bool dilate_parity = false;
	vk::raii::PipelineLayout dilate_pipeline_layout{nullptr};
	vk::raii::Pipeline dilate_pipeline{nullptr};
	// Dilate taps land on exact texel centers: reuses sampler_nearest.
	vk::raii::DescriptorSetLayout apply_descriptor_layout{nullptr};
	vk::raii::DescriptorPool apply_descriptor_pool{nullptr};
	std::vector<vk::raii::DescriptorSet> apply_descriptor_sets;
	bool apply_set_parity = false;
	vk::raii::PipelineLayout apply_layout{nullptr};
	// (Fold subpass apply pipelines live with the fold members above:
	// fold_apply_pipeline s1 paired, fold_maskapply_pipeline s1 mask.)
	// NEAREST + ClampToEdge (layer tap in the verdict, merged tap in
	// the apply: both inputs are already filtered). The half-res
	// verdict is LINEAR-tapped in the apply (tent ramp); merged stays
	// NEAREST (exact alpha).
	vk::raii::Sampler sampler_nearest{nullptr};

	// One uploaded window mesh per object id. Staging is per mesh and
	// shared-read (uploads are rare, map-change only); the pending
	// copy applies at the start of the next record_base().
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
		glm::vec3 aabb_min{0}, aabb_max{0};
	};
	std::map<std::string, mesh_buffers> layer_meshes;
	// Copy-occluder (hidden, all placed) and volume (depth back wall)
	// mesh stores: same upload discipline as layer_meshes above.
	std::map<std::string, mesh_buffers> occluder_meshes;
	std::map<std::string, mesh_buffers> volume_meshes;
};
