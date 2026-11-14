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

#ifdef XR_USE_PLATFORM_ANDROID
#include <android_native_app_glue.h>
#endif

#include "utils/handle.h"
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <optional>
#include <string>
#include <vector>

namespace xr
{
class instance;
class session;
class system;

// Environment depth maps via XR_META_environment_depth (Quest 3/3S only;
// everything else degrades to "unsupported", never a crash).
//
// The runtime owns a 2-layer (left/right) D16 depth swapchain and fills it
// asynchronously once started. Per frame, between xrBeginFrame/xrEndFrame
// (exactly once), acquire() locks the latest image plus the metadata needed
// to interpret it: nearZ/farZ bounds and the per-eye depth-camera
// fov/pose in the requested space at the requested display time. The image
// stays GPU-resident: sample it before xrEndFrame, never touch it after.
//
// Lifecycle mirrors marker_tracker: supported() gates everything, ensure()
// lazily creates provider + swapchain + image views, start()/stop() bound
// the runtime's per-frame cost to the debug toggle (off = zero cost).
// Render thread only.
class environment_depth
{
public:
	// Latest acquired depth frame: index into the enumerated depth
	// images, frustum bounds in meters (farZ < nearZ = infinite
	// projection), per-eye depth-camera fov/pose, sensor capture
	// timestamp (0 when the runtime reports ext v1 / no timestamp).
	struct frame
	{
		uint32_t swapchain_index = 0;
		float near_z = 0;
		float far_z = 0;
		XrFovf fov[2]{};
		XrPosef pose[2]{};
		XrTime capture_time = 0;
	};

	// Construct only when supported() (extension procs load in the
	// init list and throw when absent, marker_tracker idiom).
	environment_depth(instance &, session &, system &, vk::raii::Device & device);

	// Extension enabled AND runtime reports supportsEnvironmentDepth.
	static bool supported(instance &, system &);

	// Create provider + readable swapchain + depth image views.
	// Idempotent; no-op when unsupported. Logs resolution on success.
	void ensure();

	bool ready() const;

	void start();
	void stop();
	bool started() const
	{
		return provider_started;
	}

	// Hand removal (XR_META_environment_depth ext v1+ where
	// supportsHandRemoval): replaces hand pixels with estimated
	// background so tracked meshes give crisp boundaries. No-op
	// (false) when unsupported or not ready. Render thread only.
	static bool supports_hand_removal(instance &, system &);
	bool set_hand_removal(bool enabled);

	// Lock the latest depth image for this frame. Must be called between
	// begin_frame/end_frame, at most once per frame. Returns false with
	// per-call logging when no frame is available yet (recently started)
	// or on error; out is left untouched then.
	bool acquire(XrSpace space, XrTime display_time, frame & out);

	// Depth-samplable view of an enumerated depth image (2-layer 2D
	// array, D16). Valid while the provider lives.
	vk::ImageView depth_view(uint32_t swapchain_index) const;

	uint32_t depth_image_count() const;
	uint32_t depth_width() const
	{
		return depth_w;
	}
	uint32_t depth_height() const
	{
		return depth_h;
	}

private:
	instance * inst = nullptr;
	session * sess = nullptr;
	XrSystemId system_id = XR_NULL_SYSTEM_ID;
	vk::raii::Device * device = nullptr;
	uint32_t ext_version = 0;
	bool logged_unsupported = false;
	bool provider_started = false;
	bool hand_removal_enabled = false;
	// Set on start: the runtime needs a frame to spin up, so the first
	// acquire would transiently fail. The start frame skips acquire
	// silently instead of logging it.
	bool start_frame_pending = false;

#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	PFN_xrCreateEnvironmentDepthProviderMETA xrCreateEnvironmentDepthProviderMETA{};
	PFN_xrDestroyEnvironmentDepthProviderMETA xrDestroyEnvironmentDepthProviderMETA{};
	PFN_xrCreateEnvironmentDepthSwapchainMETA xrCreateEnvironmentDepthSwapchainMETA{};
	PFN_xrDestroyEnvironmentDepthSwapchainMETA xrDestroyEnvironmentDepthSwapchainMETA{};
	PFN_xrEnumerateEnvironmentDepthSwapchainImagesMETA xrEnumerateEnvironmentDepthSwapchainImagesMETA{};
	PFN_xrGetEnvironmentDepthSwapchainStateMETA xrGetEnvironmentDepthSwapchainStateMETA{};
	PFN_xrStartEnvironmentDepthProviderMETA xrStartEnvironmentDepthProviderMETA{};
	PFN_xrStopEnvironmentDepthProviderMETA xrStopEnvironmentDepthProviderMETA{};
	PFN_xrAcquireEnvironmentDepthImageMETA xrAcquireEnvironmentDepthImageMETA{};
	PFN_xrSetEnvironmentDepthHandRemovalMETA xrSetEnvironmentDepthHandRemovalMETA{};

	using provider_handle = utils::handle<XrEnvironmentDepthProviderMETA>;
	using swapchain_handle = utils::handle<XrEnvironmentDepthSwapchainMETA>;

	provider_handle provider;
	swapchain_handle swapchain;

	std::vector<XrSwapchainImageVulkanKHR> depth_images;
	std::vector<vk::raii::ImageView> depth_views;
	uint32_t depth_w = 0, depth_h = 0;
#endif
};
} // namespace xr
