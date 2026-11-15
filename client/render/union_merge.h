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

#include "render/feather_mask.h"
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
	// All contributors end COLOR_ATTACHMENT_OPTIMAL and transition to
	// SHADER_READ_ONLY for the merge; every next record re-begins
	// UNDEFINED+CLEAR, so no transition back is needed.
	struct input
	{
		vk::Image image;
		vk::ImageView view_array;
	};

	// Virtual depth for the global verdict (decoded stream 3, SBS
	// half-res luma, Y'CbCr sampler): single image holding both eyes
	// side by side, left eye x[0..w/2), right eye x[w/2..w), 10-bit
	// 2mm/LSB luma clamped at 2.040m (code 1020, so beyond-range VR
	// reads as near and occludes); 1021+ decodes as far
	// (lossy-overshoot guard only). Sampled with the decoder's own
	// sampler; green channel (least chroma crosstalk under 709) carries
	// luma. valid=false skips decode + verdict (submit merged directly,
	// today's neutral parity). near/far ride the sensor frustum so all
	// D canvases agree.
	struct virt_input
	{
		vk::ImageView view = nullptr;
		vk::Sampler sampler = nullptr;
		bool valid = false;
		float near_z = 0.1f;
		float far_z = 10.f;
	};

	// Per-record host-side cost breakdown (steady_clock, milliseconds).
	// Filled when record_base/record/record_finish get a non-null
	// out-pointer. The scene sums these per frame into Statistics.
	struct cpu_stats
	{
		double base_ms = 0;    // record_base: layer ensure + CLEAR + window MIN
		double merge_ms = 0;   // record: contributor MAX into the merged target
		double verdict_ms = 0; // record_finish: virt decode + exact layer-vs-virt compare
		double apply_ms = 0;   // record_finish: merged x bilinear(verdict) to submit
	};

	explicit union_merge_renderer(vk::raii::Device & device);

	// Shared driver pipeline cache (one per device, same object the
	// feather mask renderer uses). Render-thread only.
	static std::shared_ptr<vk::raii::PipelineCache> pipeline_cache(vk::raii::Device & device);
	// Throwaway compile so the first real merge pays ~nothing.
	static void warm_pipelines(vk::raii::Device & device, vk::Format format);

	// (Re)create swapchain + targets + pipelines for format/size
	// (full mask working res; the common-layer targets derive W = half
	// of it). waitIdle inside on rebuild; no-op when unchanged.
	// Returns true when rebuilt.
	bool ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h);

	// (Re)upload one window soup under its key. Synchronous (device
	// idle on replace); call when idle, e.g. on map change, not per
	// frame. Per-frame hash check is caller-side (the scene keeps
	// layer_hashes). Empty soup erases the entry.
	void set_layer_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup);
	void remove_layer_mesh(const std::string & key);
	// Model-space AABB for the scene-side frustum skip (nullopt =
	// unknown key: keep the draw). Computed once at upload.
	std::optional<std::pair<glm::vec3, glm::vec3>> layer_mesh_bounds(const std::string & key) const;

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

	// Stage 0 of the common layer: CLEAR the R16F intermediate (large-
	// finite FAR) then MIN-blend every window draw with the geometry
	// pipeline (linear meters out, CULL_BACK). Runs before any feather
	// group records in the same command buffer; the feather depth stage
	// re-begins the image under LOAD after a barrier. The CLEAR runs
	// even with no draws (a LOAD without defined contents is garbage).
	// Null pool skips stamps (slots 0/1).
	void record_base(vk::raii::CommandBuffer & cmd,
	                 vk::Extent2D work_extent,
	                 const std::vector<feather_mask_renderer::instance_draw> & window_draws,
	                 vk::QueryPool ts_pool = nullptr,
	                 cpu_stats * cpu = nullptr);

	// Record the MAX merge of inputs into the owned merged target (both
	// eye layers, one fullscreen multiview draw, full extent) — NOT the
	// submit image: record_finish applies the global verdict after.
	// inputs must hold 1..max_inputs entries with valid borrowed
	// views; empty or oversized returns false without recording
	// (caller releases the acquisition to keep pairing). The
	// descriptor set is rewritten per call (contributor images cycle
	// independently); two sets ping-pong by call parity so an
	// inline-fallback record never rewrites the set a still-flight
	// async command buffer reads. Stamps slot 8; null pool skips.
	bool record(vk::raii::CommandBuffer & cmd,
	            vk::Extent2D extent,
	            std::span<const input> inputs,
	            vk::QueryPool ts_pool = nullptr,
	            cpu_stats * cpu = nullptr);

	// Global finish: virt SBS decode to a D canvas at W (moved verbatim
	// from the feather renderer, owning the volume_virt_* targets),
	// then the exact verdict (intermediate linear vs virt D, nearest/
	// nearest, R8 at W), then the apply (merged alpha x bilinear
	// verdict) into both eye layers of the acquired image. !virt.valid
	// clears the verdict to 1 and applies (submit merged directly:
	// today's neutral parity). Stamps slots 6/7/9; null pool skips.
	bool record_finish(vk::raii::CommandBuffer & cmd,
	                   size_t image_index,
	                   vk::Extent2D extent,
	                   const virt_input & virt_in,
	                   vk::QueryPool ts_pool = nullptr,
	                   cpu_stats * cpu = nullptr);

	// Borrowed common-layer image for the feather depth renderer (it
	// builds its own LOAD framebuffer on this image; see
	// volume_input. BORROWED: never cached scene-side, same-frame use
	// only; the feather renderer drops its framebuffer in
	// reset_targets() and on handle change, and every merge rebuild
	// waitIdles upstream).
	vk::Image layer_image();
	vk::ImageView layer_view();
	vk::Extent2D layer_extent();

	xr::swapchain & swapchain()
	{
		return swapchain_;
	}

private:
	void init_pipelines(vk::Format format);

	vk::raii::Device * device = nullptr;
	std::shared_ptr<vk::raii::PipelineCache> pipe_cache;

	vk::raii::PipelineLayout pipeline_layout{nullptr};
	// Single-attachment multiview renderpass (CLEAR/STORE/OPTIMAL:
	// submitted image, keeps UBWC live like the feather groups) + MAX pipeline
	// (fullscreen triangle, no vertex input, no blending — one write
	// wins, the submit-time REPLACE builds the dst-alpha union).
	vk::raii::RenderPass renderpass{nullptr};
	vk::raii::Pipeline pipeline{nullptr};
	// 8 static sampler2DArray bindings (see the shader): linear +
	// ClampToEdge, the bilinear tap is the half/quarter upscale.
	vk::raii::DescriptorSetLayout descriptor_layout{nullptr};
	vk::raii::DescriptorPool descriptor_pool{nullptr};
	// Ping-pong pair (see record()): rewritten per call, never while
	// a prior submission still reads the other.
	std::vector<vk::raii::DescriptorSet> descriptor_sets;
	bool set_parity = false;
	vk::raii::Sampler sampler{nullptr};

	xr::swapchain swapchain_;
	vk::Format format_ = vk::Format::eUndefined;
	vk::Extent2D extent_{0, 0};

	struct target
	{
		vk::raii::ImageView view{nullptr}; // color array view (both eyes)
		vk::raii::Framebuffer framebuffer{nullptr}; // merge MV pass
	};
	// [image], parallel to the swapchain image list (one multiview
	// target per image — both eyes broadcast in one draw).
	std::vector<target> targets;

	// Common depth layer (R16F linear meters at W = half of full, owned
	// here; the feather depth stage borrows it, see layer_image()).
	// W derives from the merge extent, so ensure() rebuilds everything
	// together (waitIdle inside, like the swapchain targets above).
	// Destroy order: targets/sets/views below die before the
	// renderpasses/pipelines above them (members destroy in reverse
	// declaration order).
	static constexpr vk::Format layer_format = vk::Format::eR16Sfloat;
	static constexpr float layer_far_clear = 65504.f;
	// Geometry push (byte-identical to the feather geom pipe: MVP pair
	// only, 128B; both renderers throw loudly below when the device
	// reports less).
	struct geom_depth_push
	{
		std::array<glm::mat4, 2> mvp;
	};
	static_assert(sizeof(geom_depth_push) == 128);
	vk::raii::RenderPass layer_renderpass{nullptr};
	vk::raii::PipelineLayout geom_layout{nullptr};
	vk::raii::Pipeline geom_pipeline{nullptr};
	// Virt SBS->D decode targets (moved verbatim from the feather
	// renderer): R16 D canvas at W, cleared to far (neutral when
	// unpaired). Own CLEAR renderpass (R8-compare-incompatible shape).
	static constexpr vk::Format virt_depth_format = vk::Format::eR16Unorm;
	struct volume_virt_push
	{
		float near_z = 0.1f;
		float far_z = 10.f;
	};
	static_assert(sizeof(volume_virt_push) == 8);
	vk::raii::RenderPass volume_virt_renderpass{nullptr};
	vk::raii::PipelineLayout volume_virt_layout{nullptr};
	vk::raii::Pipeline volume_virt_pipeline{nullptr};
	vk::raii::DescriptorSetLayout volume_virt_set_layout{nullptr};
	vk::raii::DescriptorPool volume_virt_pool{nullptr};
	vk::raii::DescriptorSet volume_virt_set{nullptr};
	// Exact global verdict (R8 at W, nearest/nearest) + its CLEAR
	// renderpass. Own 8B near/far push (linearizes virt; the layer is
	// already meters).
	struct depth_verdict_push
	{
		float near_z = 0.1f;
		float far_z = 10.f;
	};
	static_assert(sizeof(depth_verdict_push) == 8);
	vk::raii::RenderPass verdict_renderpass{nullptr};
	vk::raii::DescriptorSetLayout verdict_set_layout{nullptr};
	vk::raii::PipelineLayout verdict_pipe_layout{nullptr};
	vk::raii::DescriptorPool verdict_pool{nullptr};
	vk::raii::DescriptorSet verdict_set{nullptr};
	vk::raii::Pipeline verdict_pipeline{nullptr};
	// Verdict apply (merged alpha x bilinear verdict -> submit). Own
	// 2-binding layout (merged NEAREST + verdict LINEAR via the two
	// samplers below), no push. Renders under renderpass (same RGBA
	// CLEAR/STORE/OPTIMAL MV shape as the merge).
	vk::raii::DescriptorSetLayout apply_set_layout{nullptr};
	vk::raii::PipelineLayout apply_pipe_layout{nullptr};
	vk::raii::DescriptorPool apply_pool{nullptr};
	vk::raii::DescriptorSet apply_set{nullptr};
	vk::raii::Pipeline apply_pipeline{nullptr};
	vk::raii::Sampler sampler_nearest{nullptr};
	// Owned layer working set at W (extent_ halved): intermediate R16F
	// + verdict R8 + virt D canvas + merged full-res RGBA (the merge
	// target record() writes; record_finish applies it to submit).
	// VMA images (device-only allocation: this renderer never sees the
	// physical device): views/framebuffers bake them per rebuild.
	struct layer_target
	{
		image_allocation image;
		vk::raii::ImageView view_array{nullptr};
		vk::raii::Framebuffer framebuffer{nullptr};
	};
	layer_target layer;   // R16F intermediate (borrowed by feather)
	layer_target verdict; // R8 binary verdict at W
	layer_target virt;    // R16 D canvas at W (SBS decode output)
	layer_target merged;  // full-res RGBA merge output (apply input)
	vk::Extent2D layer_extent_{0, 0};
	// The merged target additionally keys on the swapchain format/extent
	// (only known after ensure()): record_base runs before ensure()
	// every frame, so merged creation defers until both are committed.
	// Rebuilt merged-only (never the borrowed layer mid-frame: feather
	// may already have recorded into it in this command buffer).
	bool merged_ready_ = false;
	// Layout tracking for the borrowed image protocol: the base begins
	// the image from UNDEFINED (fresh build, discard) or READ_ONLY
	// (every recorded frame ends there); feather geometry always
	// re-begins from READ_ONLY. Set on (re)build, consumed by
	// record_base.
	bool layer_needs_discard_ = true;
	void ensure_layer_targets(vk::Extent2D w);
	// One uploaded window soup per key (scene-owned hashes, per-frame
	// check caller-side). Mirrors the feather mesh discipline: waitIdle
	// only on replace; VMA staging (host-visible) + device-local
	// buffers, flushed per record. In-flight buffers are never replaced
	// (new keys miss and skip), so first sight pays no waitIdle hitch.
	struct layer_mesh
	{
		buffer_allocation vertex_buffer;
		buffer_allocation index_buffer;
		uint32_t index_count = 0;
		buffer_allocation staging_buffer;
		vk::DeviceSize staging_size = 0;
		bool pending_upload = false;
		glm::vec3 aabb_min{0}, aabb_max{0};
	};
	std::map<std::string, layer_mesh> meshes;
	void flush_upload(vk::raii::CommandBuffer & cmd, layer_mesh & mesh);
};
