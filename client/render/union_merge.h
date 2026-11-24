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

	// Record subpass 0 of the folded pass: MAX merge of inputs into
	// the OWNED merged attachment (both eye layers, one fullscreen
	// multiview draw, full extent) — NOT the submit image. Begins the
	// folded renderpass (fold3 when paired, fold2 otherwise) on the
	// per-image framebuffer and LEAVES IT OPEN: record_finish()
	// continues with nextSubpass. inputs must hold 1..max_inputs
	// entries with valid borrowed views; empty, oversized or an
	// unknown image_index returns false WITHOUT beginning anything
	// (caller releases the acquisition to keep pairing). paired must
	// equal the virt.valid record_finish will see (caller derives both
	// from the same validity check — subpass counts differ, so a
	// mismatch would record the wrong tail). The descriptor set is
	// rewritten per call (contributor images cycle independently);
	// two sets ping-pong by call parity so an inline-fallback record
	// never rewrites the set a still-flight async command buffer reads.
	// Split-timing tap: when pool != nullptr a TopOfPipe timestamp is
	// written to qafter after the merge draw (GPU layer/merge/
	// verdict/apply split spans, read by the scene).
	bool record(vk::raii::CommandBuffer & cmd,
	            size_t image_index,
	            vk::Extent2D extent, std::span<const input> inputs,
	            bool paired,
	            const vk::raii::QueryPool * pool = nullptr, uint32_t qafter = 0);

	// Global VR-occlusion finish: continues the folded renderpass
	// record() began. Paired (virt.valid, same value passed as paired
	// above — asserted): nextSubpass, fused foveated-decode + exact
	// verdict (common layer vs virtual depth, NEAREST/NEAREST, R8 at
	// full merge res), nextSubpass, apply (merged alpha x verdict, both
	// NEAREST) into both eye layers of the acquired submit image, end.
	// Mask-only (!virt.valid): nextSubpass, merged straight through
	// (one tap, pixel-identical to x1), end. RGB zero; the submit-time
	// REPLACE preserves dst color.
	bool record_finish(vk::raii::CommandBuffer & cmd,
	                   size_t image_index,
	                   vk::Extent2D extent,
	                   const virt_input & virt_in,
	                   const vk::raii::QueryPool * pool = nullptr, uint32_t qafter_verdict = 0);

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

	// (Re)upload one window soup under its key. Synchronous (device
	// idle on replace); call when idle, e.g. on map change, not per
	// frame. Empty soup erases the entry.
	void set_layer_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup);
	void remove_layer_mesh(const std::string & key);
	// Model-space AABB for the scene-side frustum skip (nullopt =
	// unknown key: keep the draw). Computed once at upload.
	std::optional<std::pair<glm::vec3, glm::vec3>> layer_mesh_bounds(const std::string & key) const;

	xr::swapchain & swapchain()
	{
		return swapchain_;
	}

	// Borrowed debug-viz views of the owned common-layer and verdict
	// targets (2-layer arrays, full merge res). Raw handles, same-frame
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
	vk::Extent2D debug_layer_extent() const
	{
		return layer_extent_;
	}

private:
	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> make_buffer(vk::DeviceSize size,
	                                                                vk::BufferUsageFlags usage,
	                                                                vk::MemoryPropertyFlags properties);
	void flush_uploads(vk::raii::CommandBuffer & cmd);
	void init_pipelines(vk::Format format);

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
	vk::raii::Sampler sampler{nullptr};

	// Folded merge→verdict→apply (paired) and merge→apply (mask-only)
	// renderpasses (opt #8): one RP per path, fullscreen passes as
	// subpasses, tile-resident intermediates (triple precedent:
	// GENERAL attachments sampled across subpasses, subpass
	// dependencies replace the split-pass barrier sets). Shaders,
	// layouts, sets and taps are unchanged — only the RP/FB/pipeline
	// shell folds. Subpass index is baked into each pipeline (one
	// pipeline per subpass, same as every multiview pass).
	vk::raii::RenderPass fold3_rp{nullptr};
	vk::raii::RenderPass fold2_rp{nullptr};
	// Per submit image (submit view varies): {merged, verdict, submit}
	// and {merged, submit} framebuffers, parallel to targets.
	std::vector<vk::raii::Framebuffer> fold3_fbs;
	std::vector<vk::raii::Framebuffer> fold2_fbs;
	// Subpass pipelines: merge s0 + apply s2 on fold3 (verdict s1 is
	// rebuilt by create_verdict_state); merge s0 + mask-apply s1 on
	// fold2. The old split RPs/pipelines are gone.
	vk::raii::Pipeline fold_merge_pipeline{nullptr};
	vk::raii::Pipeline fold_apply_pipeline{nullptr};
	vk::raii::Pipeline fold2_merge_pipeline{nullptr};
	vk::raii::Pipeline fold2_maskapply_pipeline{nullptr};
	// Which folded RP record() began (fold3 iff paired): record_finish
	// asserts its do_verdict matches before continuing the subpasses.
	// Same-thread sequential use only, set on every record().
	bool fold_paired_ = false;

	xr::swapchain swapchain_;
	vk::Format format_ = vk::Format::eUndefined;
	vk::Extent2D extent_{0, 0};

	struct target
	{
		vk::raii::ImageView view{nullptr}; // color array view (both eyes)
	};
	// [image], parallel to the swapchain image list (one multiview
	// view per image — both eyes broadcast in one draw; the folded
	// framebuffers live in fold3_fbs/fold2_fbs below).
	std::vector<target> targets;

	// Common depth layer (R16F linear meters, full merge res): one
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
	// Binary verdict (R8, full merge res): fused foveated-decode +
	// compare output, multiplied directly by the apply (no upscale:
	// the compare already runs at submit res).
	layer_target verdict_;
	// Full merge res (== extent_): layer, merged and verdict all render
	// here. The raster layer used to be half res with an upsampled
	// verdict; foveated depth needs per-pixel precision, so the compare
	// moved up to full res and the upscale is gone.
	vk::Extent2D layer_extent_{0, 0};

	// Geometry raster: R16F MIN-blend pipeline (CLEAR/STORE/OPTIMAL,
	// CULL_NONE like the silhouette raster) + VERT-only MVP-pair push
	// layout (fragment declares no push block: no offset trap).
	vk::raii::RenderPass geom_renderpass{nullptr};
	vk::raii::PipelineLayout geom_layout{nullptr};
	vk::raii::Pipeline geom_pipeline{nullptr};
	// Verdict + apply sampler/descriptor state (static CIS layouts,
	// FRAG-only): the subpass-1 verdict pipeline is rebuilt by
	// create_verdict_state (fold3 s1), the rest in init_pipelines.
	// The fullscreen passes carry no push.
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
	// Fold3 subpass-1 pipeline (rebuilt with the layout above).
	vk::raii::Pipeline fold_verdict_pipeline{nullptr};
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
	// sets): per-eye packed-texel index per full-res merge pixel,
	// CPU-expanded from the shared foveation runs with the
	// merge->render scale baked in. Layout: [x0(W)] [x1(W)] [y0(H)]
	// [y1(H)] uint32s, W/H = merge extent. Host-visible, rewritten
	// every verdict frame (depth_overlay UBO discipline: memcpy +
	// vmaFlushAllocation under the same waitIdle'd ownership as the
	// targets); ping-ponged so an inline record never rewrites what
	// an in-flight async verdict still reads.
	buffer_allocation verdict_tables[2];
	vk::DeviceSize verdict_tables_bytes = 0;
	// Expanded index-map scratch (capacity persists after the first
	// verdict frame): avoids a ~2*(W+H) uint32 heap alloc per frame.
	std::vector<uint32_t> verdict_map_scratch;
	vk::raii::DescriptorSetLayout apply_descriptor_layout{nullptr};
	vk::raii::DescriptorPool apply_descriptor_pool{nullptr};
	std::vector<vk::raii::DescriptorSet> apply_descriptor_sets;
	bool apply_set_parity = false;
	vk::raii::PipelineLayout apply_layout{nullptr};
	// (Fold subpass apply pipelines live with the fold members above:
	// fold_apply_pipeline s2 on fold3, fold2_maskapply_pipeline s1 on
	// fold2. Mask-only: binding 1 written but never sampled.)
	// NEAREST + ClampToEdge (layer tap in the verdict, merged tap in
	// the apply: both inputs are already filtered). The verdict runs
	// at full merge res and multiplies directly in the apply (both
	// NEAREST) — no upscale, no LINEAR sampler exists.
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
};
