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

#include "depth_reprojector.h"

namespace wivrn
{

bool depth_reprojector::reproject(vk::raii::CommandBuffer &,
                                  int,
                                  vk::ImageView,
                                  vk::ImageView,
                                  const std::array<float, 4> &,
                                  const std::array<float, 4> &,
                                  bool,
                                  bool,
                                  const xrt_pose &,
                                  const xrt_fov &,
                                  float,
                                  float,
                                  const xrt_pose &,
                                  const xrt_fov &)
{
	// STUB: the warp slices land here. Until then every commit serves straight.
	return false;
}

} // namespace wivrn
