/*
 * WiVRn VR streaming
 * Copyright (C) 2026  Patrick Nicolas <patricknicolas@laposte.net>
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
 * along with this program.  If not, see <https://www.gnu.org/>.
 */

#pragma once

#include "vk/allocation.h"

#include "xrt/xrt_defines.h"

#include <array>
#include <vulkan/vulkan_raii.hpp>

// UBO mirror of server/compositor/shaders/reproject_splat.comp:
// source clip -> world -> target clip, plus depth sampling params.
// mat4 arrays are column-major on both sides: xrt_matrix_4x4::v copies
// straight across. The inverse-view * inverse-projection product is
// folded on the CPU.
struct reproject_splat_ubo
{
	xrt_matrix_4x4 render_inv_vp;
	xrt_matrix_4x4 warp_vp;
	float depth_rect[4]; // normalized depth sub-image: x, y, w, h
	float depth_flip; // 1 = mirror rows (Vulkan-ordered source)
	float src_reverse_z; // 1 = reverse-Z source (convert 1-z)
	float pad[2];
};
static_assert(sizeof(reproject_splat_ubo) == 160);

// UBO mirror of server/compositor/shaders/reproject_pull.comp:
// target clip -> world -> source clip, plus color sampling params.
struct reproject_pull_ubo
{
	xrt_matrix_4x4 dst_inv;
	xrt_matrix_4x4 src_vp;
	float color_rect[4]; // normalized color sub-image: x, y, w, h
	float color_flip; // 1 = mirror rows (Vulkan-ordered source)
	float pad[3];
};
static_assert(sizeof(reproject_pull_ubo) == 160);

namespace wivrn
{
struct vk_bundle;

// Server-side 6DoF reprojection (v3): every commit carrying depth warps
// the layer's own color+depth from its generation pose to the latest
// predicted pose. No retained sources, no freshness tracking, no gates:
// depth always travels paired with its color, so the warp reads the
// layer images directly and runs at most once per commit.
//
// Two compute passes per eye: splat forwards each source pixel's depth
// to the target pose (atomic min wins = foreground), pull warps color
// backward per target pixel through the splatted depth
// (premultiplied: valid (rgb, 1), holes black). A smear fill then copies
// the nearest background-depth donor into each hole (verbatim texel:
// smear instead of blur); holes with no background donor inside the
// search radius, or no established background reference, stay black.
//
// Fixed owned scratch per (slot, eye): warped color + splat depth at
// full render size — not a pool, no acquire/release. Per-warp
// descriptor rewrites and UBO uploads land in the commit's slot, so
// overlapping submits never share CPU-written state (the safety story
// is slot ownership via image-busy lifetime, not the commit tail).
class depth_reprojector
{
public:
	depth_reprojector(vk_bundle & vk, vk::Extent2D target_size, bool reverse_z);

	// Warps one eye's layer images (generation pose/params) to
	// dst_pose/dst_fov, recording into cmd. Slot selects the commit's
	// per-slot scratch/UBO/sets (== compositor image index); view
	// selects the eye. Norm-rects map each eye's sub-image; the flip
	// flags select the per-plane row order (true = caller rows are
	// Vulkan-ordered, top row first). Output ends in
	// SHADER_READ_ONLY_OPTIMAL, ready to feed foveation (see
	// get_views). Returns false on degenerate input (caller falls
	// back: no warp recorded, slot's ready flag untouched).
	bool reproject(vk::raii::Device & device,
	               vk::raii::CommandBuffer & cmd,
	               int slot,
	               int view,
	               vk::ImageView src_color,
	               vk::ImageView src_depth,
	               const std::array<float, 4> & color_rect,
	               const std::array<float, 4> & depth_rect,
	               bool flip_color,
	               bool flip_depth,
	               const xrt_pose & src_pose,
	               const xrt_fov & src_fov,
	               float src_near_z,
	               float src_far_z,
	               const xrt_pose & dst_pose,
	               const xrt_fov & dst_fov);

	std::array<vk::ImageView, 2> get_views(int slot);

private:
	// Depth encoding at construction (config source-reverse-z):
	// reverse-Z sources are converted in the splat pass. Read once
	// here so the hot path never touches the config.
	const bool reverse_z;
	const vk::Extent2D extent;

	vk::raii::Sampler color_sampler = nullptr;
	vk::raii::Sampler depth_sampler = nullptr;

	// Per-(slot, eye) warped color targets (plain UNORM, sampled raw by
	// foveation) plus per-(slot, eye) splatted foreground-depth scratch
	// (R32_UINT, cleared each warp). Targets end SHADER_READ_ONLY,
	// scratch ends GENERAL, between warps. Slot == compositor image
	// index: the commit for images[i] only touches slot i, so a warp
	// recording into slot i can never race a still-executing submit that
	// used slot 1-i.
	static constexpr size_t num_slots = 2;
	std::array<std::array<bool, 2>, num_slots> warp_ready = {};
	std::array<std::array<image_allocation, 2>, num_slots> target;
	std::array<std::array<vk::raii::ImageView, 2>, num_slots> target_view{{{{nullptr, nullptr}}, {{nullptr, nullptr}}}};
	std::array<std::array<image_allocation, 2>, num_slots> splat_depth;
	std::array<std::array<vk::raii::ImageView, 2>, num_slots> splat_view{{{{nullptr, nullptr}}, {{nullptr, nullptr}}}};
	// Coarse background reference: one R32_UINT texel per 64x64 block
	// (perimeter-max splat depth, EMPTY-skipped). Same slot/eye
	// ownership as splat/target; stays GENERAL between warps.
	std::array<std::array<image_allocation, 2>, num_slots> bgmax;
	std::array<std::array<vk::raii::ImageView, 2>, num_slots> bgmax_view{{{{nullptr, nullptr}}, {{nullptr, nullptr}}}};

	// Splat: depth sampler + splat UBO + splat storage image.
	// Pull: color sampler + pull UBO + splat storage image (read) +
	// color storage target (write). Bgmax: splat storage image (read) +
	// coarse reference storage image (write). Fill: splat storage image
	// (read) + color storage target (read-write) + coarse reference
	// storage image (read). Splat/pull sets are rewritten per warp (src
	// views change per app frame); bgmax/fill sets reference only owned
	// images and are written once. Only the UBO contents change
	// otherwise. Per-(slot, eye): CPU rewrites are unordered by GPU
	// barriers, so sharing one set across overlapping submits would race.
	vk::raii::DescriptorSetLayout splat_ds_layout = nullptr;
	vk::raii::PipelineLayout splat_layout = nullptr;
	vk::raii::Pipeline splat_pipeline = nullptr;
	vk::raii::DescriptorSetLayout pull_ds_layout = nullptr;
	vk::raii::PipelineLayout pull_layout = nullptr;
	vk::raii::Pipeline pull_pipeline = nullptr;
	vk::raii::DescriptorSetLayout bgmax_ds_layout = nullptr;
	vk::raii::PipelineLayout bgmax_layout = nullptr;
	vk::raii::Pipeline bgmax_pipeline = nullptr;
	vk::raii::DescriptorSetLayout fill_ds_layout = nullptr;
	vk::raii::PipelineLayout fill_layout = nullptr;
	vk::raii::Pipeline fill_pipeline = nullptr;
	vk::raii::DescriptorPool pool = nullptr;
	std::array<std::array<vk::DescriptorSet, 2>, num_slots> splat_sets;
	std::array<std::array<vk::DescriptorSet, 2>, num_slots> pull_sets;
	std::array<std::array<vk::DescriptorSet, 2>, num_slots> bgmax_sets;
	std::array<std::array<vk::DescriptorSet, 2>, num_slots> fill_sets;

	std::array<std::array<buffer_allocation, 2>, num_slots> splat_ubo;
	std::array<std::array<buffer_allocation, 2>, num_slots> pull_ubo;
};
} // namespace wivrn
