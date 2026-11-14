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

#include "environment_depth.h"
#include "xr/instance.h"
#include "xr/session.h"
#include "xr/system.h"
#include "xr/to_string.h"
#include <openxr/openxr_platform.h>
#include <spdlog/spdlog.h>

xr::environment_depth::environment_depth(instance & inst_, session & sess_, system & sys, vk::raii::Device & device_) :
        inst(&inst_),
        sess(&sess_),
        system_id(sys),
        device(&device_)
        ,
        ext_version(inst_.extension_version(XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME))
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
        ,
        xrCreateEnvironmentDepthProviderMETA(inst_.get_proc<PFN_xrCreateEnvironmentDepthProviderMETA>("xrCreateEnvironmentDepthProviderMETA")),
        xrDestroyEnvironmentDepthProviderMETA(inst_.get_proc<PFN_xrDestroyEnvironmentDepthProviderMETA>("xrDestroyEnvironmentDepthProviderMETA")),
        xrCreateEnvironmentDepthSwapchainMETA(inst_.get_proc<PFN_xrCreateEnvironmentDepthSwapchainMETA>("xrCreateEnvironmentDepthSwapchainMETA")),
        xrDestroyEnvironmentDepthSwapchainMETA(inst_.get_proc<PFN_xrDestroyEnvironmentDepthSwapchainMETA>("xrDestroyEnvironmentDepthSwapchainMETA")),
        xrEnumerateEnvironmentDepthSwapchainImagesMETA(inst_.get_proc<PFN_xrEnumerateEnvironmentDepthSwapchainImagesMETA>("xrEnumerateEnvironmentDepthSwapchainImagesMETA")),
        xrGetEnvironmentDepthSwapchainStateMETA(inst_.get_proc<PFN_xrGetEnvironmentDepthSwapchainStateMETA>("xrGetEnvironmentDepthSwapchainStateMETA")),
        xrStartEnvironmentDepthProviderMETA(inst_.get_proc<PFN_xrStartEnvironmentDepthProviderMETA>("xrStartEnvironmentDepthProviderMETA")),
        xrStopEnvironmentDepthProviderMETA(inst_.get_proc<PFN_xrStopEnvironmentDepthProviderMETA>("xrStopEnvironmentDepthProviderMETA")),
        xrAcquireEnvironmentDepthImageMETA(inst_.get_proc<PFN_xrAcquireEnvironmentDepthImageMETA>("xrAcquireEnvironmentDepthImageMETA")),
        xrSetEnvironmentDepthHandRemovalMETA(inst_.get_proc<PFN_xrSetEnvironmentDepthHandRemovalMETA>("xrSetEnvironmentDepthHandRemovalMETA")),
        provider(provider_handle{xrDestroyEnvironmentDepthProviderMETA}),
        swapchain(swapchain_handle{xrDestroyEnvironmentDepthSwapchainMETA})
#endif
{
	// Extension procs load above (init list, marker_tracker idiom), so
	// callers must gate construction on supported(): with the extension
	// enabled and advertised, every proc must exist.
}

#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
static bool query_depth_properties(xr::instance & inst, xr::system & sys, XrSystemEnvironmentDepthPropertiesMETA & out)
{
	out = XrSystemEnvironmentDepthPropertiesMETA{.type = XR_TYPE_SYSTEM_ENVIRONMENT_DEPTH_PROPERTIES_META};
	XrSystemProperties props{.type = XR_TYPE_SYSTEM_PROPERTIES, .next = &out};
	if (XrResult res = xrGetSystemProperties(inst, sys, &props); res != XR_SUCCESS)
	{
		spdlog::warn("environment_depth: xrGetSystemProperties failed: {}", xr::to_string(res));
		return false;
	}
	return true;
}
#endif

bool xr::environment_depth::supported(instance & inst, system & sys)
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	if (not inst.has_extension(XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME))
		return false;
	XrSystemEnvironmentDepthPropertiesMETA props;
	if (not query_depth_properties(inst, sys, props))
		return false;
	return props.supportsEnvironmentDepth == XR_TRUE;
#else
	(void)inst;
	(void)sys;
	return false;
#endif
}

void xr::environment_depth::ensure()
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	if (ready() or not xrCreateEnvironmentDepthProviderMETA)
	{
		if (not xrCreateEnvironmentDepthProviderMETA and not logged_unsupported)
		{
			logged_unsupported = true;
			spdlog::info("environment_depth: unsupported on this runtime");
		}
		return;
	}

	XrEnvironmentDepthProviderCreateInfoMETA create_info{
	        .type = XR_TYPE_ENVIRONMENT_DEPTH_PROVIDER_CREATE_INFO_META,
	        .createFlags = 0,
	};
	XrEnvironmentDepthProviderMETA prov = XR_NULL_HANDLE;
	if (XrResult res = xrCreateEnvironmentDepthProviderMETA(*sess, &create_info, &prov); res != XR_SUCCESS)
	{
		spdlog::warn("environment_depth: xrCreateEnvironmentDepthProviderMETA failed: {}", xr::to_string(res));
		return;
	}
	provider = provider_handle(prov, xrDestroyEnvironmentDepthProviderMETA);

	XrEnvironmentDepthSwapchainCreateInfoMETA swap_info{
	        .type = XR_TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_CREATE_INFO_META,
	        .createFlags = 0,
	};
	XrEnvironmentDepthSwapchainMETA sw = XR_NULL_HANDLE;
	if (XrResult res = xrCreateEnvironmentDepthSwapchainMETA(provider, &swap_info, &sw); res != XR_SUCCESS)
	{
		spdlog::warn("environment_depth: xrCreateEnvironmentDepthSwapchainMETA failed: {}", xr::to_string(res));
		provider = provider_handle{xrDestroyEnvironmentDepthProviderMETA};
		return;
	}
	swapchain = swapchain_handle(sw, xrDestroyEnvironmentDepthSwapchainMETA);

	XrEnvironmentDepthSwapchainStateMETA state{.type = XR_TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_STATE_META};
	if (XrResult res = xrGetEnvironmentDepthSwapchainStateMETA(swapchain, &state); res == XR_SUCCESS)
	{
		depth_w = state.width;
		depth_h = state.height;
	}

	uint32_t count = 0;
	if (XrResult res = xrEnumerateEnvironmentDepthSwapchainImagesMETA(swapchain, 0, &count, nullptr); res != XR_SUCCESS or count == 0)
	{
		spdlog::warn("environment_depth: enumerate count failed: {}", xr::to_string(res));
		return;
	}
	depth_images.assign(count, XrSwapchainImageVulkanKHR{.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
	if (XrResult res = xrEnumerateEnvironmentDepthSwapchainImagesMETA(
	            swapchain, count, &count, reinterpret_cast<XrSwapchainImageBaseHeader *>(depth_images.data()));
	    res != XR_SUCCESS)
	{
		spdlog::warn("environment_depth: enumerate images failed: {}", xr::to_string(res));
		depth_images.clear();
		return;
	}

	// One sampled view per depth image: the images are 2-layer arrays
	// (left/right); the eye rides the shader layer index, so a single
	// 2D-array view covers both. D16_UNORM is the spec-mandated Vulkan
	// format for this swapchain type (no format query exists).
	depth_views.clear();
	for (size_t i = 0; i < depth_images.size(); ++i)
	{
		vk::ImageViewCreateInfo iv_info{
		        .image = depth_images[i].image,
		        .viewType = vk::ImageViewType::e2DArray,
		        .format = vk::Format::eD16Unorm,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eDepth,
		                .baseMipLevel = 0,
		                .levelCount = 1,
		                .baseArrayLayer = 0,
		                .layerCount = 2,
		        },
		};
		depth_views.emplace_back(*device, iv_info);
	}

	spdlog::info("environment_depth: ready, {}x{} ({} images)", depth_w, depth_h, depth_images.size());
#endif
}

bool xr::environment_depth::ready() const
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	return (bool)provider and (bool)swapchain and not depth_views.empty();
#else
	return false;
#endif
}

void xr::environment_depth::start()
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	if (provider_started or not ready())
		return;
	if (XrResult res = xrStartEnvironmentDepthProviderMETA(provider); res != XR_SUCCESS)
	{
		spdlog::warn("environment_depth: start failed: {}", xr::to_string(res));
		return;
	}
	provider_started = true;
	start_frame_pending = true;
	spdlog::info("environment_depth: provider started");
#endif
}

void xr::environment_depth::stop()
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	if (not provider_started)
		return;
	provider_started = false;
	if (XrResult res = xrStopEnvironmentDepthProviderMETA(provider); res != XR_SUCCESS)
		spdlog::warn("environment_depth: stop failed: {}", xr::to_string(res));
	else
		spdlog::info("environment_depth: provider stopped");
#endif
}

bool xr::environment_depth::acquire(XrSpace space, XrTime display_time, frame & out)
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	if (not provider_started)
		return false;
	if (start_frame_pending)
	{
		// Provider warming up (see start()): skip silently, the UI
		// already reads "waiting for depth data".
		start_frame_pending = false;
		return false;
	}
	XrEnvironmentDepthImageAcquireInfoMETA acquire_info{
	        .type = XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_ACQUIRE_INFO_META,
	        .space = space,
	        .displayTime = display_time,
	};
	// captureTime rides the output chain on ext v2+ (content-age
	// forensics); v1 runtimes ignore the unknown struct per the chaining
	// rules, so it stays unchained there.
	XrEnvironmentDepthImageTimestampMETA timestamp{
	        .type = XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_TIMESTAMP_META,
	        .captureTime = 0,
	};
	// The runtime validates output structs: each views element must
	// carry its type (same trap as XrView in locate_views).
	XrEnvironmentDepthImageMETA image{
	        .type = XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_META,
	        .next = ext_version >= 2 ? &timestamp : nullptr,
	        .views =
	                {
	                        XrEnvironmentDepthImageViewMETA{.type = XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_VIEW_META},
	                        XrEnvironmentDepthImageViewMETA{.type = XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_VIEW_META},
	                },
	};
	XrResult res = xrAcquireEnvironmentDepthImageMETA(provider, &acquire_info, &image);
	if (res == XR_ENVIRONMENT_DEPTH_NOT_AVAILABLE_META)
		return false; // recently started: no data yet, not an error
	// Throttled failure log: acquire runs every frame, an unthrottled
	// warn spams logcat at refresh rate. First failure + heartbeat;
	// the count resets on success so it reads per-outage.
	static XrTime last_fail_log = 0;
	static uint64_t fail_count = 0;
	if (res != XR_SUCCESS)
	{
		++fail_count;
		XrTime now = inst->now();
		if (fail_count == 1 or now - last_fail_log > 5'000'000'000)
		{
			last_fail_log = now;
			spdlog::warn("environment_depth: acquire failed ({}x): {}", fail_count, xr::to_string(res));
		}
		return false;
	}
	fail_count = 0;
	out.swapchain_index = image.swapchainIndex;
	out.near_z = image.nearZ;
	out.far_z = image.farZ;
	out.fov[0] = image.views[0].fov;
	out.fov[1] = image.views[1].fov;
	out.pose[0] = image.views[0].pose;
	out.pose[1] = image.views[1].pose;
	out.capture_time = timestamp.captureTime;
	return true;
#else
	(void)space;
	(void)display_time;
	(void)out;
	return false;
#endif
}

vk::ImageView xr::environment_depth::depth_view(uint32_t swapchain_index) const
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	if (swapchain_index < depth_views.size())
		return *depth_views[swapchain_index];
#endif
	(void)swapchain_index;
	return nullptr;
}

uint32_t xr::environment_depth::depth_image_count() const
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
	return (uint32_t)depth_images.size();
#else
	return 0;
#endif
}

bool xr::environment_depth::supports_hand_removal(instance & inst, system & sys)
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
    if (not inst.has_extension(XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME))
        return false;
    XrSystemEnvironmentDepthPropertiesMETA props;
    if (not query_depth_properties(inst, sys, props))
        return false;
    return props.supportsHandRemoval == XR_TRUE;
#else
    (void)inst;
    (void)sys;
    return false;
#endif
}

bool xr::environment_depth::set_hand_removal(bool enabled)
{
#ifdef XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME
    if (not provider_started or not xrSetEnvironmentDepthHandRemovalMETA)
        return false;
    if (enabled == hand_removal_enabled)
        return true;
    XrEnvironmentDepthHandRemovalSetInfoMETA info{
        .type = XR_TYPE_ENVIRONMENT_DEPTH_HAND_REMOVAL_SET_INFO_META,
        .enabled = enabled ? XR_TRUE : XR_FALSE,
    };
    if (XrResult res = xrSetEnvironmentDepthHandRemovalMETA(provider, &info); res != XR_SUCCESS)
    {
        spdlog::warn("environment_depth: hand removal {} failed: {}", enabled, xr::to_string(res));
        return false;
    }
    hand_removal_enabled = enabled;
    spdlog::info("environment_depth: hand removal {}", enabled ? "on" : "off");
    return true;
#else
    (void)enabled;
    return false;
#endif
}
