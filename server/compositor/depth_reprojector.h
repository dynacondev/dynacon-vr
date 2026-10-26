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

#include "xrt/xrt_defines.h"

#include <array>
#include <vulkan/vulkan_raii.hpp>

namespace wivrn
{

// Server-side 6DoF reprojection (v3): every commit carrying depth warps
// the layer's own color+depth from its generation pose to the latest
// predicted pose. No retained sources, no freshness tracking, no gates:
// depth always travels paired with its color, so the warp reads the
// layer images directly and runs at most once per commit.
class depth_reprojector
{
public:
	depth_reprojector() = default;

	// Warps one eye's layer images (generation pose/params) to
	// dst_pose/dst_fov, recording into cmd. Norm-rects map each eye's
	// sub-image; the flip flags select the per-plane row order (true =
	// caller rows are Vulkan-ordered, top row first). Returns false
	// when nothing was recorded (caller serves the frame straight).
	// STUB: always returns false until the warp slices land.
	bool reproject(vk::raii::CommandBuffer & cmd,
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
};

} // namespace wivrn
