/*
 * WiVRn VR streaming
 * Copyright (C) 2024  galister <galister@librevr.org>
 * Copyright (C) 2025  Patrick Nicolas <patricknicolas@laposte.net>
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

#include "vk/allocation.h"
#include "wivrn_packets.h"
#include "xrt/xrt_defines.h"

#include <mutex>
#include <vulkan/vulkan_raii.hpp>

struct render_resources;

namespace wivrn
{

struct clock_offset;
struct vk_bundle;

class foveation
{
	std::mutex mutex;

	const vk::Extent3D foveated_size; // per eye

	// Natural vertical gaze angle
	const float angle_offset;
	// Optionally defined from environment variables
	const float convergence_distance;

	float eye_x[2] = {}; // eye x position
	xrt_quat gaze = {};
	from_headset::override_foveation_center manual_foveation = {};
	std::array<to_headset::foveation_parameter, 2> params;

	vk::raii::Sampler sampler;

	vk::raii::DescriptorSetLayout ds_layout;
	vk::raii::PipelineLayout layout;
	std::array<vk::raii::Pipeline, 2> pipeline;
	vk::raii::DescriptorPool descriptor_pool;
	// Per-slot upload/read state (slot == compositor image index).
	// Member params stays shared: rewritten-before-use per call.
	static constexpr size_t num_slots = 2;
	std::array<buffer_allocation, num_slots> gpu_buffers;
	std::array<vk::DescriptorSet, num_slots> descriptor_sets;

	// parameters used for last computation, per slot (a shared cache
	// would false-skip uploads against another slot's buffer).
	struct P
	{
		xrt_quat gaze = {};
		bool flip_y = false;
		xrt_rect src[2] = {};
		xrt_fov fovs[2] = {};
		float eye_x[2] = {};

		from_headset::override_foveation_center manual_foveation = {};
	};
	std::array<P, num_slots> last;

	// must hold lock on mutex to call it
	void compute_params(int slot);

	void update_ubo(
	        vk::raii::CommandBuffer & cmd,
	        int slot,
	        bool flip_y,
	        std::array<xrt_rect, 2> src_rect,
	        std::array<xrt_fov, 2> src_fov);

public:
	foveation(wivrn::vk_bundle &,
	          vk::Extent3D foveated_size);

	void update_tracking(const from_headset::tracking &);
	void update_foveation_center_override(const from_headset::override_foveation_center &);

	std::array<to_headset::foveation_parameter, 2> foveate(
	        vk::raii::Device &,
	        vk::raii::CommandBuffer & cmd,
	        int slot,
	        vk::ImageView y,
	        vk::ImageView cbcr,
	        bool flip_y,
	        std::array<vk::ImageView, 2> src,
	        std::array<xrt_rect, 2> src_rect,
	        std::array<xrt_fov, 2> src_fov,
	        bool alpha);
};
} // namespace wivrn
