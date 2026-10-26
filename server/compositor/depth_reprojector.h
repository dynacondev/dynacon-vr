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
// (premultiplied: valid (rgb, 1), holes black). No hole fill: holes
// stay black by design.
//
// Fixed owned scratch (per-eye warped color + splat depth, full render
// size): not a pool, no acquire/release. Per-warp descriptor rewrites
// and UBO uploads mirror foveation exactly — including its safety
// story (the commit tail waits on the submitted work, serializing
// commits against the GPU).
class depth_reprojector
{
public:
	depth_reprojector(vk_bundle & vk, vk::Extent2D target_size, bool reverse_z);

	// Warps one eye's layer images (generation pose/params) to
	// dst_pose/dst_fov, recording into cmd. Norm-rects map each eye's
	// sub-image; the flip flags select the per-plane row order (true =
	// caller rows are Vulkan-ordered, top row first). Output ends in
	// SHADER_READ_ONLY_OPTIMAL, ready to feed foveation (see
	// get_views). Returns false on degenerate input (caller falls
	// back: no warp recorded).
	bool reproject(vk::raii::Device & device,
	               vk::raii::CommandBuffer & cmd,
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

	std::array<vk::ImageView, 2> get_views();

private:
	// Depth encoding at construction (config source-reverse-z):
	// reverse-Z sources are converted in the splat pass. Read once
	// here so the hot path never touches the config.
	const bool reverse_z;
	const vk::Extent2D extent;

	vk::raii::Sampler color_sampler = nullptr;
	vk::raii::Sampler depth_sampler = nullptr;

	// Per-eye warped color targets (plain UNORM, sampled raw by
	// foveation) plus per-eye splatted foreground-depth scratch
	// (R32_UINT, cleared each warp). Targets end SHADER_READ_ONLY,
	// scratch ends GENERAL, between warps. Per-eye images: the two
	// eyes warp independently.
	std::array<bool, 2> warp_ready = {false, false};
	std::array<image_allocation, 2> target;
	std::array<vk::raii::ImageView, 2> target_view = {nullptr, nullptr};
	std::array<image_allocation, 2> splat_depth;
	std::array<vk::raii::ImageView, 2> splat_view = {nullptr, nullptr};

	// Splat: depth sampler + splat UBO + splat storage image.
	// Pull: color sampler + pull UBO + splat storage image (read) +
	// color storage target (write). Sets are rewritten per warp (src
	// views change per app frame); only the UBO contents change
	// otherwise.
	vk::raii::DescriptorSetLayout splat_ds_layout = nullptr;
	vk::raii::PipelineLayout splat_layout = nullptr;
	vk::raii::Pipeline splat_pipeline = nullptr;
	vk::raii::DescriptorSetLayout pull_ds_layout = nullptr;
	vk::raii::PipelineLayout pull_layout = nullptr;
	vk::raii::Pipeline pull_pipeline = nullptr;
	vk::raii::DescriptorPool pool = nullptr;
	std::array<vk::DescriptorSet, 2> splat_sets;
	std::array<vk::DescriptorSet, 2> pull_sets;

	std::array<buffer_allocation, 2> splat_ubo;
	std::array<buffer_allocation, 2> pull_ubo;
};
} // namespace wivrn
