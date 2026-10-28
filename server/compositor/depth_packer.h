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

#include <array>
#include <vulkan/vulkan_raii.hpp>

namespace wivrn
{
struct vk_bundle;

// Packs warped virtual depth for the depth video stream (stream_idx 3).
//
// Input: the warp's per-eye target-view splat images (R32Uint,
// DEPTH_SCALE-encoded NDC, sentinel = no splat) at full render size, plus
// the dst frustum near/far per eye. Output: the Y plane (2D view on layer
// 3) of the 640x320 SBS staging image — left eye x[0..320), right eye
// x[320..640), 8-bit 1cm/LSB, 255 = >=2.55m / sky / hole.
//
// Resample (MIN-pool, foreground wins) runs before linearisation, in the
// encoded domain; linearisation + quantize run at 320x320. See
// shaders/depth_pack.comp.
//
// Slot ownership mirrors depth_reprojector: one descriptor set per commit
// slot, rewritten per pack; the caller records only into its owned slot
// while the other may still execute. Push constants carry the small
// params (src extent + near/far); no UBOs.
class depth_packer
{
public:
	static constexpr uint32_t sbs_width = 640;
	static constexpr uint32_t sbs_height = 320;
	static constexpr uint32_t eye_width = 320;

	depth_packer(vk_bundle & vk);

	// Records the pack into cmd. Returns false on degenerate input
	// (nothing recorded). The caller owns layout/barrier transitions of
	// dst_y (GENERAL on entry); splat inputs must be visible for shader
	// read (the warp's own barriers cover this when recorded after fill).
	bool pack(vk::raii::Device & device,
	          vk::raii::CommandBuffer & cmd,
	          int slot,
	          vk::ImageView splat_l,
	          vk::ImageView splat_r,
	          vk::Extent2D src_extent,
	          float near_l,
	          float far_l,
	          float near_r,
	          float far_r,
	          vk::ImageView dst_y);

private:
	struct push
	{
		float src_w, src_h;
		float near_l, far_l;
		float near_r, far_r;
	};
	static_assert(sizeof(push) == 24);

	static constexpr size_t num_slots = 2;
	vk::raii::DescriptorSetLayout set_layout = nullptr;
	vk::raii::PipelineLayout pipe_layout = nullptr;
	vk::raii::Pipeline pipeline = nullptr;
	vk::raii::DescriptorPool pool = nullptr;
	std::array<vk::DescriptorSet, num_slots> sets;
};
} // namespace wivrn
