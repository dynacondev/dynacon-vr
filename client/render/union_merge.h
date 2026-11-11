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
#include <memory>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

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
	// general_layout: true for feather/debug images (end
	// COLOR_ATTACHMENT_OPTIMAL: availability barrier only, sampled
	// as OPTIMAL so the same-frame cutout LOAD re-begin still
	// matches), false for the hidden image (ends
	// COLOR_ATTACHMENT_OPTIMAL: transitioned to SHADER_READ_ONLY
	// for the merge; its next record re-begins UNDEFINED+CLEAR, so
	// no transition back is needed).
	struct input
	{
		vk::Image image;
		vk::ImageView view_array;
		bool general_layout = true;
	};

	explicit union_merge_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device);

	// Shared driver pipeline cache (one per device, same object the
	// feather mask and dots renderers use). Render-thread only.
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

	// Record the MAX merge of inputs into both eye layers of the
	// acquired image (one fullscreen multiview draw, full extent).
	// inputs must hold 1..max_inputs entries with valid borrowed
	// views; empty or oversized returns false without recording
	// (caller releases the acquisition to keep pairing). The
	// descriptor set is rewritten per call (contributor images cycle
	// independently); two sets ping-pong by call parity so an
	// inline-fallback record never rewrites the set a still-flight
	// async command buffer reads.
	bool record(vk::raii::CommandBuffer & cmd, size_t image_index,
	            vk::Extent2D extent, std::span<const input> inputs);

	xr::swapchain & swapchain()
	{
		return swapchain_;
	}

private:
	void init_pipelines(vk::Format format);

	vk::raii::Device * device = nullptr;
	vk::raii::PhysicalDevice * physical_device = nullptr;
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
};
