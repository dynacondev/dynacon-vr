/*
 * WiVRn VR streaming
 * Copyright (C) 2022  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2022  Patrick Nicolas <patricknicolas@laposte.net>
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
#include "constants.h"
#include "xr/body_tracker.h"
#include "xr/face_tracker.h"
#include "xr/fb_face_tracker2.h"
#include "xr/space.h"
#include "xr/system.h"
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <magic_enum.hpp>
#include <openxr/openxr.h>
#define GLM_FORCE_RADIANS

#include "stream.h"

#include "application.h"
#include "audio/audio.h"
#include "decoder/shard_accumulator.h"
#include "inplace_vector.hpp"
#include "spdlog/spdlog.h"
#include "utils/i18n.h"
#include "utils/named_thread.h"
#include "utils/ranges.h"
#include "wivrn_packets.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <ranges>
#include <thread>
#include <vulkan/vulkan_raii.hpp>

#include "wivrn_config.h"
#if WIVRN_FEATURE_RENDERDOC
#include "vk/renderdoc.h"
#endif

using namespace wivrn;
using namespace beman::inplace_vector;

// clang-format off
static const std::unordered_map<std::string, device_id> device_ids = {
	{"/user/hand/left/input/dpad_up/click",       device_id::DPAD_UP_CLICK},
	{"/user/hand/left/input/dpad_up/touch",       device_id::DPAD_UP_TOUCH},
	{"/user/hand/left/input/dpad_down/click",     device_id::LEFT_X_CLICK},
	{"/user/hand/left/input/dpad_down/touch",     device_id::LEFT_X_TOUCH},
	{"/user/hand/left/input/dpad_left/click",     device_id::DPAD_LEFT_CLICK},
	{"/user/hand/left/input/dpad_left/touch",     device_id::DPAD_LEFT_TOUCH},
	{"/user/hand/left/input/dpad_right/click",    device_id::LEFT_Y_CLICK},
	{"/user/hand/left/input/dpad_right/touch",    device_id::LEFT_Y_TOUCH},
	{"/user/hand/left/input/x/click",             device_id::LEFT_X_CLICK},
	{"/user/hand/left/input/x/touch",             device_id::LEFT_X_TOUCH},
	{"/user/hand/left/input/y/click",             device_id::LEFT_Y_CLICK},
	{"/user/hand/left/input/y/touch",             device_id::LEFT_Y_TOUCH},
	{"/user/hand/left/input/view/click",          device_id::VIEW_CLICK},
	{"/user/hand/left/input/view/touch",          device_id::VIEW_TOUCH},
	{"/user/hand/left/input/menu/click",          device_id::LEFT_MENU_CLICK},
	{"/user/hand/left/input/system/click",        device_id::LEFT_MENU_CLICK},
	{"/user/hand/left/input/system/touch",        device_id::LEFT_SYSTEM_TOUCH},
	{"/user/hand/left/input/bumper/click",        device_id::LEFT_BUMPER_CLICK},
	{"/user/hand/left/input/bumper/touch",        device_id::LEFT_BUMPER_TOUCH},
	{"/user/hand/left/input/squeeze/click",       device_id::LEFT_SQUEEZE_CLICK},
	{"/user/hand/left/input/squeeze/force",       device_id::LEFT_SQUEEZE_FORCE},
	{"/user/hand/left/input/squeeze/touch",       device_id::LEFT_SQUEEZE_TOUCH},
	{"/user/hand/left/input/squeeze/value",       device_id::LEFT_SQUEEZE_VALUE},
	{"/user/hand/left/input/trigger/value",       device_id::LEFT_TRIGGER_VALUE},
	{"/user/hand/left/input/trigger/click",       device_id::LEFT_TRIGGER_CLICK},
	{"/user/hand/left/input/trigger/touch",       device_id::LEFT_TRIGGER_TOUCH},
	{"/user/hand/left/input/trigger/proximity",   device_id::LEFT_TRIGGER_PROXIMITY},
	{"/user/hand/left/input/trigger/proximity_fb",device_id::LEFT_TRIGGER_PROXIMITY},
	{"/user/hand/left/input/trigger/proximity_meta",device_id::LEFT_TRIGGER_PROXIMITY},
	{"/user/hand/left/input/trigger/curl_fb",     device_id::LEFT_TRIGGER_CURL},
	{"/user/hand/left/input/trigger/curl_meta",   device_id::LEFT_TRIGGER_CURL},
	{"/user/hand/left/input/trigger_curl/value",  device_id::LEFT_TRIGGER_CURL},
	{"/user/hand/left/input/trigger/slide_fb",    device_id::LEFT_TRIGGER_SLIDE},
	{"/user/hand/left/input/trigger/slide_meta",  device_id::LEFT_TRIGGER_SLIDE},
	{"/user/hand/left/input/trigger_slide/value", device_id::LEFT_TRIGGER_SLIDE},
	{"/user/hand/left/input/trigger/force",       device_id::LEFT_TRIGGER_FORCE},
	{"/user/hand/left/input/thumbstick",          device_id::LEFT_THUMBSTICK_X},
	{"/user/hand/left/input/thumbstick/click",    device_id::LEFT_THUMBSTICK_CLICK},
	{"/user/hand/left/input/thumbstick/touch",    device_id::LEFT_THUMBSTICK_TOUCH},
	{"/user/hand/left/input/thumbrest/touch",     device_id::LEFT_THUMBREST_TOUCH},
	{"/user/hand/left/input/thumbrest/force",     device_id::LEFT_THUMBREST_FORCE},
	{"/user/hand/left/input/thumb_resting_surfaces/proximity",device_id::LEFT_THUMB_PROXIMITY},
	{"/user/hand/left/input/thumb_meta/proximity_meta",device_id::LEFT_THUMB_PROXIMITY},
	{"/user/hand/left/input/trackpad",            device_id::LEFT_TRACKPAD_X},
	{"/user/hand/left/input/trackpad/click",      device_id::LEFT_TRACKPAD_CLICK},
	{"/user/hand/left/input/trackpad/touch",      device_id::LEFT_TRACKPAD_TOUCH},
	{"/user/hand/left/input/trackpad/force",      device_id::LEFT_TRACKPAD_FORCE},
	{"/user/hand/left/input/stylus/force",        device_id::LEFT_STYLUS_FORCE},
	{"/user/hand/left/input/stylus_fb/force",     device_id::LEFT_STYLUS_FORCE},

	{"/user/hand/right/input/a/click",             device_id::A_CLICK},
	{"/user/hand/right/input/a/touch",             device_id::A_TOUCH},
	{"/user/hand/right/input/b/click",             device_id::B_CLICK},
	{"/user/hand/right/input/b/touch",             device_id::B_TOUCH},
	{"/user/hand/right/input/x/click",             device_id::RIGHT_X_CLICK},
	{"/user/hand/right/input/x/touch",             device_id::RIGHT_X_TOUCH},
	{"/user/hand/right/input/y/click",             device_id::RIGHT_Y_CLICK},
	{"/user/hand/right/input/y/touch",             device_id::RIGHT_Y_TOUCH},
	{"/user/hand/right/input/system/click",        device_id::RIGHT_SYSTEM_CLICK},
	{"/user/hand/right/input/system/touch",        device_id::RIGHT_SYSTEM_TOUCH},
	{"/user/hand/right/input/menu/click",          device_id::RIGHT_MENU_CLICK},
	{"/user/hand/right/input/menu/touch",          device_id::RIGHT_MENU_TOUCH},
	{"/user/hand/right/input/bumper/click",        device_id::RIGHT_BUMPER_CLICK},
	{"/user/hand/right/input/bumper/touch",        device_id::RIGHT_BUMPER_TOUCH},
	{"/user/hand/right/input/squeeze/click",       device_id::RIGHT_SQUEEZE_CLICK},
	{"/user/hand/right/input/squeeze/force",       device_id::RIGHT_SQUEEZE_FORCE},
	{"/user/hand/right/input/squeeze/touch",       device_id::RIGHT_SQUEEZE_TOUCH},
	{"/user/hand/right/input/squeeze/value",       device_id::RIGHT_SQUEEZE_VALUE},
	{"/user/hand/right/input/trigger/value",       device_id::RIGHT_TRIGGER_VALUE},
	{"/user/hand/right/input/trigger/click",       device_id::RIGHT_TRIGGER_CLICK},
	{"/user/hand/right/input/trigger/touch",       device_id::RIGHT_TRIGGER_TOUCH},
	{"/user/hand/right/input/trigger/proximity",   device_id::RIGHT_TRIGGER_PROXIMITY},
	{"/user/hand/right/input/trigger/proximity_fb",device_id::RIGHT_TRIGGER_PROXIMITY},
	{"/user/hand/right/input/trigger/proximity_meta",device_id::RIGHT_TRIGGER_PROXIMITY},
	{"/user/hand/right/input/trigger/curl_fb",     device_id::RIGHT_TRIGGER_CURL},
	{"/user/hand/right/input/trigger/curl_meta",   device_id::RIGHT_TRIGGER_CURL},
	{"/user/hand/right/input/trigger_curl/value",  device_id::RIGHT_TRIGGER_CURL},
	{"/user/hand/right/input/trigger/slide_fb",    device_id::RIGHT_TRIGGER_SLIDE},
	{"/user/hand/right/input/trigger/slide_meta",  device_id::RIGHT_TRIGGER_SLIDE},
	{"/user/hand/right/input/trigger_slide/value", device_id::RIGHT_TRIGGER_SLIDE},
	{"/user/hand/right/input/trigger/force",       device_id::RIGHT_TRIGGER_FORCE},
	{"/user/hand/right/input/thumbstick",          device_id::RIGHT_THUMBSTICK_X},
	{"/user/hand/right/input/thumbstick/click",    device_id::RIGHT_THUMBSTICK_CLICK},
	{"/user/hand/right/input/thumbstick/touch",    device_id::RIGHT_THUMBSTICK_TOUCH},
	{"/user/hand/right/input/thumbrest/touch",     device_id::RIGHT_THUMBREST_TOUCH},
	{"/user/hand/right/input/thumbrest/force",     device_id::RIGHT_THUMBREST_FORCE},
	{"/user/hand/right/input/thumb_resting_surfaces/proximity",device_id::RIGHT_THUMB_PROXIMITY},
	{"/user/hand/right/input/thumb_meta/proximity_meta",device_id::RIGHT_THUMB_PROXIMITY},
	{"/user/hand/right/input/trackpad",            device_id::RIGHT_TRACKPAD_X},
	{"/user/hand/right/input/trackpad/click",      device_id::RIGHT_TRACKPAD_CLICK},
	{"/user/hand/right/input/trackpad/touch",      device_id::RIGHT_TRACKPAD_TOUCH},
	{"/user/hand/right/input/trackpad/force",      device_id::RIGHT_TRACKPAD_FORCE},
	{"/user/hand/right/input/stylus/force",        device_id::RIGHT_STYLUS_FORCE},
	{"/user/hand/right/input/stylus_fb/force",     device_id::RIGHT_STYLUS_FORCE},

	// XR_EXT_hand_interaction
	{"/user/hand/left/input/pinch_ext/value",      device_id::LEFT_PINCH_VALUE},
	{"/user/hand/left/input/pinch_ext/ready_ext",  device_id::LEFT_PINCH_READY},
	{"/user/hand/left/input/aim_activate_ext/value",device_id::LEFT_AIM_ACTIVATE_VALUE},
	{"/user/hand/left/input/aim_activate_ext/ready_ext",device_id::LEFT_AIM_ACTIVATE_READY},
	{"/user/hand/left/input/grasp_ext/value",      device_id::LEFT_GRASP_VALUE},
	{"/user/hand/left/input/grasp_ext/ready_ext",  device_id::LEFT_GRASP_READY},

	{"/user/hand/right/input/pinch_ext/value",      device_id::RIGHT_PINCH_VALUE},
	{"/user/hand/right/input/pinch_ext/ready_ext",  device_id::RIGHT_PINCH_READY},
	{"/user/hand/right/input/aim_activate_ext/value",device_id::RIGHT_AIM_ACTIVATE_VALUE},
	{"/user/hand/right/input/aim_activate_ext/ready_ext",device_id::RIGHT_AIM_ACTIVATE_READY},
	{"/user/hand/right/input/grasp_ext/value",      device_id::RIGHT_GRASP_VALUE},
	{"/user/hand/right/input/grasp_ext/ready_ext",  device_id::RIGHT_GRASP_READY},

	{"/user/gamepad/input/menu/click",             device_id::GAMEPAD_MENU_CLICK},
	{"/user/gamepad/input/view/click",             device_id::GAMEPAD_VIEW_CLICK},
	{"/user/gamepad/input/a/click",                device_id::GAMEPAD_A_CLICK},
	{"/user/gamepad/input/b/click",                device_id::GAMEPAD_B_CLICK},
	{"/user/gamepad/input/x/click",                device_id::GAMEPAD_X_CLICK},
	{"/user/gamepad/input/y/click",                device_id::GAMEPAD_Y_CLICK},
	{"/user/gamepad/input/dpad_down/click",        device_id::GAMEPAD_DPAD_DOWN_CLICK},
	{"/user/gamepad/input/dpad_right/click",       device_id::GAMEPAD_DPAD_RIGHT_CLICK},
	{"/user/gamepad/input/dpad_up/click",          device_id::GAMEPAD_DPAD_UP_CLICK},
	{"/user/gamepad/input/dpad_left/click",        device_id::GAMEPAD_DPAD_LEFT_CLICK},
	{"/user/gamepad/input/shoulder_left/click",    device_id::GAMEPAD_SHOULDER_LEFT_CLICK},
	{"/user/gamepad/input/shoulder_right/click",   device_id::GAMEPAD_SHOULDER_RIGHT_CLICK},
	{"/user/gamepad/input/thumbstick_left/click",  device_id::GAMEPAD_THUMBSTICK_LEFT_CLICK},
	{"/user/gamepad/input/thumbstick_right/click", device_id::GAMEPAD_THUMBSTICK_RIGHT_CLICK},
	{"/user/gamepad/input/trigger_left/value",     device_id::GAMEPAD_TRIGGER_LEFT_VALUE},
	{"/user/gamepad/input/trigger_right/value",    device_id::GAMEPAD_TRIGGER_RIGHT_VALUE},
	{"/user/gamepad/input/thumbstick_left/x",      device_id::GAMEPAD_THUMBSTICK_LEFT_X},
	{"/user/gamepad/input/thumbstick_left/y",      device_id::GAMEPAD_THUMBSTICK_LEFT_Y},
	{"/user/gamepad/input/thumbstick_right/x",     device_id::GAMEPAD_THUMBSTICK_RIGHT_X},
	{"/user/gamepad/input/thumbstick_right/y",     device_id::GAMEPAD_THUMBSTICK_RIGHT_Y},
};
// clang-format on

static const std::array supported_color_formats = {
        vk::Format::eR8G8B8A8Srgb,
        vk::Format::eB8G8R8A8Srgb,
};

static const std::array supported_depth_formats{
        vk::Format::eD32Sfloat,
        vk::Format::eX8D24UnormPack32,
};

scenes::stream::stream(std::string server_name, scene & parent_scene) :
        scene_impl<stream>(supported_color_formats, supported_depth_formats, parent_scene),
        apps{*this, std::move(server_name)}
{
	auto views = system.view_configuration_views(viewconfig);
	width = views[0].recommendedImageRectWidth;
	height = views[0].recommendedImageRectHeight;

	request_spatial_permissions();
	for (auto & c: stream_rx_bytes)
		c.store(0, std::memory_order_relaxed);
}

static from_headset::visibility_mask_changed::masks get_visibility_mask(xr::instance & inst, xr::session & session, int view)
{
	assert(inst.has_extension(XR_KHR_VISIBILITY_MASK_EXTENSION_NAME));
	static auto xrGetVisibilityMaskKHR = inst.get_proc<PFN_xrGetVisibilityMaskKHR>("xrGetVisibilityMaskKHR");

	from_headset::visibility_mask_changed::masks res{};
	for (auto [type, mask]: utils::enumerate(res))
	{
		XrVisibilityMaskKHR xr_mask{
		        .type = XR_TYPE_VISIBILITY_MASK_KHR,
		};
		CHECK_XR(xrGetVisibilityMaskKHR(session, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, view, XrVisibilityMaskTypeKHR(type + 1), &xr_mask));
		mask.vertices.resize(xr_mask.vertexCountOutput);
		mask.indices.resize(xr_mask.indexCountOutput);
		xr_mask = {
		        .type = XR_TYPE_VISIBILITY_MASK_KHR,
		        .vertexCapacityInput = uint32_t(mask.vertices.size()),
		        .vertices = mask.vertices.data(),
		        .indexCapacityInput = uint32_t(mask.indices.size()),
		        .indices = mask.indices.data(),
		};
		CHECK_XR(xrGetVisibilityMaskKHR(session, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, view, XrVisibilityMaskTypeKHR(type + 1), &xr_mask));
		mask.vertices.resize(xr_mask.vertexCountOutput);
		mask.indices.resize(xr_mask.indexCountOutput);
	}
	return res;
}

std::shared_ptr<scenes::stream> scenes::stream::create(std::unique_ptr<wivrn_session> network_session, float guessed_fps, std::string server_name, scene & parent_scene)
{
	std::shared_ptr<stream> self{new stream{std::move(server_name), parent_scene}};
	self->network_session = std::move(network_session);

	self->network_session->send_control([&]() {
		from_headset::headset_info_packet info{
		        .language = application::get_messages_info().language,
		        .country = application::get_messages_info().country,
		        .variant = application::get_messages_info().variant,
		};

		const auto & config = application::get_config();

		{
			auto view = self->system.view_configuration_views(self->viewconfig)[0];
			view = application::get_hmd_traits().override_view(view);

			info.render_eye_width = view.recommendedImageRectWidth * config.resolution_scale;
			info.render_eye_height = view.recommendedImageRectHeight * config.resolution_scale;
		}

		{
			auto scale = config.get_stream_scale();
			info.stream_eye_width = info.render_eye_width * scale;
			info.stream_eye_height = info.render_eye_height * scale;
		}

		if (self->instance.has_extension(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME))
		{
			info.available_refresh_rates = self->session.get_refresh_rates();
			info.settings.preferred_refresh_rate = config.preferred_refresh_rate;
			info.settings.minimum_refresh_rate = config.minimum_refresh_rate.value_or(0);
		}
		info.settings.fps_divider = config.fps_divider;

		if (info.available_refresh_rates.empty())
		{
			spdlog::warn("Unable to detect refresh rates");
			info.available_refresh_rates = {guessed_fps};
			info.settings.preferred_refresh_rate = guessed_fps;
		}

		info.settings.bitrate_bps = config.bitrate_bps;
		info.settings.mirror_gamepad = config.forward_gamepad;
		info.settings.enabled_body_parts = config.body_part_mask;

		info.hand_tracking = config.check_feature(feature::hand_tracking);
		info.eye_gaze = config.check_feature(feature::eye_gaze);

		if (self->instance.has_extension(XR_EXT_USER_PRESENCE_EXTENSION_NAME))
		{
			info.user_presence = self->system.user_presence_properties().supportsUserPresence;
		}

		if (config.check_feature(feature::face_tracking))
		{
			switch (self->system.face_tracker_supported())
			{
				case xr::face_tracker_type::none:
					info.face_tracking = from_headset::face_type::none;
					break;
				case xr::face_tracker_type::android:
					info.face_tracking = from_headset::face_type::android;
					break;
				case xr::face_tracker_type::fb:
				case xr::face_tracker_type::pico:
					info.face_tracking = from_headset::face_type::fb2;
					break;
				case xr::face_tracker_type::htc:
					info.face_tracking = from_headset::face_type::htc;
					break;
			}
		}

		if (config.check_feature(feature::body_tracking))
		{
			switch (self->system.body_tracker_supported())
			{
				case xr::body_tracker_type::none:
					info.body_tracking = from_headset::body_type::none;
					break;
				case xr::body_tracker_type::fb:
					info.body_tracking = from_headset::body_type::fb;
					break;
				case xr::body_tracker_type::meta:
					info.body_tracking = from_headset::body_type::meta;
					break;
				case xr::body_tracker_type::pico:
					info.body_tracking = from_headset::body_type::bd;
					break;
				case xr::body_tracker_type::htc:
					info.body_tracking = from_headset::body_type::htc;
					info.num_generic_trackers = application::get_generic_trackers().size();
					break;
			}
		}

		info.palm_pose = application::space(xr::spaces::palm_left) or application::space(xr::spaces::palm_right);
		info.passthrough = self->system.passthrough_supported() != xr::passthrough_type::none;
		info.system_name = std::string(self->system.properties().systemName);

		audio::get_audio_description(info);
		if (not(config.check_feature(feature::microphone)))
			info.microphone = {};

		if (config.codec)
		{
			const bool supports_10bit = decoder::supports_10bit(*config.codec);
			info.supported_codecs = {{.codec = *config.codec, .supports_10bit = supports_10bit}};
			switch (*config.codec)
			{
				case h264:
				case raw:
					break;
				case h265:
				case av1:
					info.bit_depth = supports_10bit ? config.bit_depth : 8;
			}
		}
		else
		{
			info.supported_codecs = decoder::supported_codecs();
		}

		return info;
	}());

	self->network_session->send_control(from_headset::session_state_changed{
	        .state = application::get_session_state(),
	});
	self->network_session->send_control(from_headset::stream_tab_changed{
	        .tab = self->gui_status,
	});

	if (self->instance.has_extension(XR_KHR_VISIBILITY_MASK_EXTENSION_NAME))
	{
		for (uint8_t view = 0; view < view_count; ++view)
		{
			try
			{
				self->network_session->send_control(from_headset::visibility_mask_changed{
				        .data = get_visibility_mask(self->instance, self->session, view),
				        .view_index = view});
			}
			catch (std::exception & e)
			{
				spdlog::warn("Failed to get visibility mask: ", e.what());
			}
		}
	}

	{
		const auto & config = application::get_config();
		self->override_foveation_enable = config.override_foveation_enable;
		self->override_foveation_pitch = config.override_foveation_pitch;
		self->override_foveation_distance = config.override_foveation_distance;

		if (self->override_foveation_enable)
			self->network_session->send_control(from_headset::override_foveation_center{
			        .enabled = self->override_foveation_enable,
			        .pitch = self->override_foveation_pitch,
			        .distance = self->override_foveation_distance,
			});
	}

	self->network_thread = utils::named_thread("network_thread", &stream::process_packets, self.get());

	self->command_buffer = std::move(self->device.allocateCommandBuffers({
	        .commandPool = *self->commandpool,
	        .level = vk::CommandBufferLevel::ePrimary,
	        .commandBufferCount = 1,
	})[0]);

	self->fence = self->device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled});

	self->mask_command_buffer = std::move(self->device.allocateCommandBuffers({
	        .commandPool = *self->commandpool,
	        .level = vk::CommandBufferLevel::ePrimary,
	        .commandBufferCount = 1,
	})[0]);

	self->mask_fence = self->device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled});

	// Look up the XrActions for haptics
	for (auto [id, path, output]: {
	             std::tuple(device_id::LEFT_CONTROLLER_HAPTIC, "/user/hand/left", "/output/haptic"),
	             std::tuple(device_id::RIGHT_CONTROLLER_HAPTIC, "/user/hand/right", "/output/haptic"),

	             std::tuple(device_id::LEFT_TRIGGER_HAPTIC, "/user/hand/left", "/output/haptic_trigger"),
	             std::tuple(device_id::RIGHT_TRIGGER_HAPTIC, "/user/hand/right", "/output/haptic_trigger"),
	             std::tuple(device_id::LEFT_TRIGGER_HAPTIC, "/user/hand/left", "/output/haptic_trigger_fb"),
	             std::tuple(device_id::RIGHT_TRIGGER_HAPTIC, "/user/hand/right", "/output/haptic_trigger_fb"),

	             std::tuple(device_id::LEFT_THUMB_HAPTIC, "/user/hand/left", "/output/haptic_thumb"),
	             std::tuple(device_id::RIGHT_THUMB_HAPTIC, "/user/hand/right", "/output/haptic_thumb"),
	             std::tuple(device_id::LEFT_THUMB_HAPTIC, "/user/hand/left", "/output/haptic_thumb_fb"),
	             std::tuple(device_id::RIGHT_THUMB_HAPTIC, "/user/hand/right", "/output/haptic_thumb_fb"),

	             std::tuple(device_id::GAMEPAD_HAPTIC_LEFT, "/user/gamepad", "/output/haptic_left"),
	             std::tuple(device_id::GAMEPAD_HAPTIC_RIGHT, "/user/gamepad", "/output/haptic_right"),
	             std::tuple(device_id::GAMEPAD_HAPTIC_LEFT_TRIGGER, "/user/gamepad", "/output/haptic_left_trigger"),
	             std::tuple(device_id::GAMEPAD_HAPTIC_RIGHT_TRIGGER, "/user/gamepad", "/output/haptic_right_trigger")})
	{
		if (auto action = application::get_action(std::string(path) + output); action.first)
		{
			self->haptics_actions.emplace(id, haptics_action{
			                                          .action = action.first,
			                                          .path = self->instance.string_to_path(path),
			                                  });
		}
	}

	// Look up the XrActions for input
	for (const auto & [action, action_type, name]: application::inputs())
	{
		auto it = device_ids.find(name);

		if (it == device_ids.end())
			continue;

		self->input_actions.emplace_back(it->second, action, action_type);
	}

	spdlog::info("Using format {}", vk::to_string(self->swapchain_format));

	self->query_pool = vk::raii::QueryPool(
	        self->device,
	        vk::QueryPoolCreateInfo{
	                .queryType = vk::QueryType::eTimestamp,
	                .queryCount = size_gpu_timestamps,
	        });

	self->mask_query_pool = vk::raii::QueryPool(
	        self->device,
	        vk::QueryPoolCreateInfo{
	                .queryType = vk::QueryType::eTimestamp,
	                .queryCount = 3,
	        });

	// Volume GPU brackets (raster/virt/compare/merge boundaries, 5 queries).
	// Dedicated pool so the main/mask read discipline (never wait on an
	// unwritten query) is untouched; volume_ts_valid gates the poll.
	self->volume_query_pool = vk::raii::QueryPool(
	        self->device,
	        vk::QueryPoolCreateInfo{
	                .queryType = vk::QueryType::eTimestamp,
	                .queryCount = 5,
	        });

	self->wifi = application::get_wifi_lock().get_wifi_lock();

	return self;
}

void scenes::stream::on_focused()
{
	gui_status_last_change = instance.now();

	const auto & profile = application::get_hmd_traits().controller_profile;
	input.emplace(
	        *this,
	        "assets://controllers/" + profile + "/profile.json",
	        layer_controllers,
	        layer_rays,
	        get_action("left_trigger").first,
	        get_action("right_trigger").first);

	spdlog::info("Loaded input profile {}", input->id);

	for (auto i: {xr::spaces::aim_left, xr::spaces::aim_right, xr::spaces::grip_left, xr::spaces::grip_right})
	{
		auto [p, q] = input->offset[i] = application::get_hmd_traits().controller_offset(i);

		auto rot = glm::degrees(glm::eulerAngles(q));
		spdlog::info("Initializing offset of space {} to ({}, {}, {}) mm, ({}, {}, {})°",
		             magic_enum::enum_name(i),
		             1000 * p.x,
		             1000 * p.y,
		             1000 * p.z,
		             rot.x,
		             rot.y,
		             rot.z);
	}

	std::array imgui_inputs{
	        imgui_context::controller{
	                .aim = get_action_space("left_aim"),
	                .offset = input->offset[xr::spaces::aim_left],
	                .trigger = get_action("left_trigger").first,
	                .squeeze = get_action("left_squeeze").first,
	                .scroll = get_action("left_scroll").first,
	                .haptic_output = get_action("left_haptic").first,
	        },
	        imgui_context::controller{
	                .aim = get_action_space("right_aim"),
	                .offset = input->offset[xr::spaces::aim_right],
	                .trigger = get_action("right_trigger").first,
	                .squeeze = get_action("right_squeeze").first,
	                .scroll = get_action("right_scroll").first,
	                .haptic_output = get_action("right_haptic").first,
	        },
	};

	xr::swapchain swapchain_imgui(
	        instance,
	        session,
	        device,
	        swapchain_format,
	        3600,
	        1200);

	std::vector<imgui_context::viewport> vps{
	        {
	                // Main window
	                .space = xr::spaces::world,
	                // Position and orientation are set at each frame
	                .size = {1.2, 0.6666},
	                .vp_origin = {0, 0},
	                .vp_size = {1800, 1000},
	        },
	        {
	                .space = xr::spaces::world,
	                .size = {1.2, 0.1333},
	                .vp_origin = {0, 1000},
	                .vp_size = {1800, 200},
	                .tooltip_viewport = true,
	        },
	        {
	                // popup window for combos and modals, tracks the main panel each frame at the same pixel density
	                .space = xr::spaces::world,
	                .size = {1.2, 0.6666},
	                .vp_origin = {1800, 0},
	                .vp_size = {1800, 1000},
	        },
	};

	imgui_ctx.emplace(physical_device,
	                  device,
	                  queue_family_index,
	                  queue,
	                  imgui_inputs,
	                  std::move(swapchain_imgui),
	                  std::move(vps),
	                  image_cache);

	// match the lobby's seasonal wordmark logo
	{
		auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		auto tm = std::localtime(&t);
		switch (tm->tm_mon)
		{
			case 5:
				wivrn_logo = imgui_ctx->load_texture("assets://wivrn-pride.ktx2");
				break;
			case 11:
				wivrn_logo = imgui_ctx->load_texture("assets://wivrn-christmas.ktx2");
				break;
			default:
				wivrn_logo = imgui_ctx->load_texture("assets://wivrn.ktx2");
				break;
		}
	}

	kiosk_chord_left_action = get_action("kiosk_chord_left").first;
	kiosk_chord_right_action = get_action("kiosk_chord_right").first;

	kiosk_planes_.emplace(*this, layer_controllers);
	recenter_left = get_action("recenter_left").first;
	recenter_right = get_action("recenter_right").first;
	gui_distance_left = get_action("gui_distance_left").first;
	gui_distance_right = get_action("gui_distance_right").first;
	settings_adjust = get_action("settings_adjust").first;
	foveation_distance = get_action("foveation_distance").first;
	foveation_ok = get_action("foveation_ok").first;
	foveation_cancel = get_action("foveation_cancel").first;

	if (application::get_config().high_power_mode)
	{
		session.set_performance_level(XR_PERF_SETTINGS_DOMAIN_CPU_EXT, XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT);
		session.set_performance_level(XR_PERF_SETTINGS_DOMAIN_GPU_EXT, XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT);
	}
	else
	{
		session.set_performance_level(XR_PERF_SETTINGS_DOMAIN_CPU_EXT, XR_PERF_SETTINGS_LEVEL_SUSTAINED_LOW_EXT);
		session.set_performance_level(XR_PERF_SETTINGS_DOMAIN_GPU_EXT, XR_PERF_SETTINGS_LEVEL_SUSTAINED_LOW_EXT);
	}
}

void scenes::stream::on_unfocused()
{
	renderer->wait_idle(); // Must be before the scene data because the renderer uses its descriptor sets;
	kiosk_planes_.reset();
	world.clear();
	input.reset();
	left_hand.reset();
	right_hand.reset();
	apps.reset();

	imgui_ctx.reset();
}

scenes::stream::~stream()
{
	exit();

	if (tracking_thread && tracking_thread->joinable())
		tracking_thread->join();

	if (network_thread.joinable())
		network_thread.join();
}

void scenes::stream::push_blit_handle(shard_accumulator * decoder, std::shared_ptr<shard_accumulator::blit_handle> handle)
{
	assert(handle);
	if (!application::is_visible())
		return;

	{
		std::shared_lock lock(decoder_mutex);
		std::unique_lock frame_lock(frames_mutex);
		auto stream = handle->feedback.stream_index;
		if (stream < decoders.size())
		{
			if (decoder != decoders[stream].decoder.get())
				return;
			handle->feedback.received_from_decoder = instance.now();
			std::swap(handle, decoders[stream].latest_frames[handle->feedback.frame_index % decoders[stream].latest_frames.size()]);
		}

		if (state_ != state::streaming and not(decoders[0].empty() or decoders[1].empty()))
		{
			set_state(state::streaming);
			spdlog::info("Stream scene ready at t={}", instance.now());
		}
	}

	if (handle and not handle->feedback.blitted)
	{
		send_feedback(handle->feedback);
	}
}

bool scenes::stream::accumulator_images::empty() const
{
	for (const auto & frame: latest_frames)
	{
		if (frame)
			return false;
	}
	return true;
}

std::array<std::shared_ptr<shard_accumulator::blit_handle>, scenes::stream::decoder_count> scenes::stream::common_frame(XrTime display_time)
{
	if (decoders.empty())
		return {};
	std::unique_lock lock(frames_mutex);
	inplace_vector<shard_accumulator::blit_handle *, decoder_count> common_frames;
	const bool alpha = decoders[0].latest_frames[0] and decoders[0].latest_frames[0]->view_info.alpha;
	for (size_t i = 0; i < view_count + alpha; ++i)
	{
		if (i == 0)
		{
			for (const auto & h: decoders[i].latest_frames)
				if (h)
					common_frames.push_back(h.get());
		}
		else
		{
			// clang-format off
			erase_if(common_frames,
				[this, i](auto & left)
				{
					return std::ranges::none_of(
						decoders[i].latest_frames,
						[&left](auto & right)
						{
							return right and left->feedback.frame_index == right->feedback.frame_index;
						});
				});
			// clang-format on
		}
	}
	std::array<std::shared_ptr<shard_accumulator::blit_handle>, decoder_count> result;
	if (not common_frames.empty())
	{
		auto min = std::ranges::min_element(common_frames,
		                                    std::ranges::less{},
		                                    [display_time](auto frame) {
			                                    if (not frame)
				                                    return std::numeric_limits<XrTime>::max();
			                                    return std::abs(frame->view_info.display_time - display_time);
		                                    });

		assert(*min);
		auto frame_index = (*min)->feedback.frame_index;
		const size_t color_count = view_count + (alpha ? 1 : 0);
		for (auto [i, decoder]: utils::enumerate(decoders))
		{
			if (i < color_count)
				result[i] = decoder.frame(frame_index);
		}
	}
	else
	{
		spdlog::warn("Failed to find a common frame for all decoders, dumping available frames per decoder");
		for (const auto & decoder: decoders)
		{
			std::string frames;
			for (const auto & frame: decoder.latest_frames)
			{
				if (frame)
					frames += " " + std::to_string(frame->feedback.frame_index);
				else
					frames += " -";
			}
			spdlog::warn(frames);
		}

		const size_t color_count = view_count + (alpha ? 1 : 0);
		for (auto [i, decoder]: utils::enumerate(decoders))
		{
			if (i < color_count)
			{
				auto min = std::ranges::min_element(decoder.latest_frames,
				                                    std::ranges::less{},
				                                    [display_time](auto frame) {
					                                    if (not frame)
						                                    return std::numeric_limits<XrTime>::max();
					                                    return std::abs(frame->view_info.display_time - display_time);
				                                    });
				result[i] = *min;
			}
		}
	}
	// Virtual depth (stream 3) never joins the common-frame intersection:
	// strict frame_idx pairing with the selected color frame, else null
	// (the volume compare skips the virtual term for that frame). No
	// hold-last: stale virtual depth occludes wrong pixels.
	if (result[0])
		result[depth_stream_index] = decoders[depth_stream_index].frame(result[0]->feedback.frame_index);
	return result;
}

std::shared_ptr<shard_accumulator::blit_handle> scenes::stream::accumulator_images::frame(uint64_t id) const
{
	auto & frame = latest_frames[id % latest_frames.size()];
	if (frame and frame->feedback.frame_index != id)
		return nullptr;
	return frame;
}

void scenes::stream::update_gui_position(xr::spaces controller, float predicted_display_period)
{
	std::optional<std::pair<glm::vec3, glm::quat>> aim = application::locate_controller(
	        application::space(controller),
	        application::space(xr::spaces::world),
	        predicted_display_time);

	if (not aim)
		return;

	auto [offset_position, offset_orientation] = input->offset[controller];

	auto world_controller_position = aim->first + glm::mat3_cast(aim->second * offset_orientation) * offset_position;
	auto world_controller_orientation = aim->second * offset_orientation;
	auto world_controller_direction = -glm::column(glm::mat3_cast(world_controller_orientation), 2);

	if (not recentering_context)
	{
		// First frame of recentering: get the GUI position relative to the controller

		// Compute the intersection of the ray with the GUI
		auto gui_controller_direction = glm::conjugate(world_gui_orientation) * world_controller_direction;
		auto gui_controller_position = glm::conjugate(world_gui_orientation) * (world_controller_position - world_gui_position);

		float lambda = -gui_controller_position.z / gui_controller_direction.z;
		auto gui_intersection = gui_controller_position + lambda * gui_controller_direction;

		auto viewport_size = imgui_ctx->layers()[0].size;
		if (std::isnan(lambda) or lambda < 0 or
		    std::abs(gui_intersection.x) > viewport_size.x / 2 or
		    std::abs(gui_intersection.y) > viewport_size.y / 2)
		{
			// Reset the relative GUI position if the ray does not intersect
			recentering_context.emplace(controller, glm::vec3{0, 0, -1}, glm::quat{1, 0, 0, 0});
		}
		else
		{
			glm::vec3 controller_gui_position = glm::conjugate(world_controller_orientation) * (world_gui_position - world_controller_position);
			glm::quat controller_gui_orientation = glm::conjugate(world_controller_orientation) * world_gui_orientation;

			recentering_context.emplace(controller, controller_gui_position, controller_gui_orientation);
		}
	}
	else
	{
		// Subsequent frames of recentering: keep the GUI locked to the controller
		auto & [_, controller_gui_position, controller_gui_orientation] = *recentering_context;

		if (auto gui_distance = application::read_action_float(controller == xr::spaces::aim_left ? gui_distance_left : gui_distance_right))
		{
			controller_gui_position.z *= std::pow(constants::stream::gui_max_layer_speed, gui_distance->second * predicted_display_period);
			controller_gui_position.z = -std::clamp<float>(-controller_gui_position.z, constants::stream::gui_min_layer_distance, constants::stream::gui_max_layer_distance);
		}

		world_gui_position = world_controller_position + world_controller_orientation * controller_gui_position;
		world_gui_orientation = world_controller_orientation * controller_gui_orientation;
	}
}

bool scenes::stream::is_interactable(stream_tab tab)
{
	switch (tab)
	{
		case stream_tab::stats:
		case stream_tab::settings:
		case stream_tab::foveation_settings:
		case stream_tab::applications:
		case stream_tab::application_launcher:
			return true;

		case stream_tab::hidden:
		case stream_tab::overlay_only:
		case stream_tab::compact:
			return false;
	}
	spdlog::warn("Invalid tab value {}", int(tab));
	return false;
}

bool scenes::stream::is_gui_interactable() const
{
	return is_interactable(gui_status);
}

bool scenes::stream::has_active_simulation()
{
	auto apps = running_applications.lock();
	for (auto & app: apps->applications)
		if (app.active and not app.overlay)
			return true;
	return false;
}

void scenes::stream::update_stream_kiosk(XrTime predicted_display_time)
{
	const auto & config = application::get_config();
	// no overlay (and no need to poll) when the in-stream GUI is disabled:
	// the chord that would reveal it is dead too
	const bool allowed = config.kiosk_mode and config.enable_stream_gui;

	// keep the running-apps picture fresh while the overlay may show; the
	// applications tab polls itself when visible, this covers the overlay
	if (allowed and predicted_display_time - last_kiosk_app_req > 2'000'000'000LL)
	{
		last_kiosk_app_req = predicted_display_time;
		network_session->send_control(from_headset::get_running_applications{});
	}

	bool has_active = allowed and stream_apps_known and has_active_simulation();
	if (stream_had_active_app and not has_active)
		stream_kiosk_dismissed = false; // app stopped: waiting overlay returns
	stream_had_active_app = has_active;

	stream_kiosk_shown = allowed and not stream_kiosk_dismissed and
	                     stream_apps_known and not has_active and not is_gui_interactable();
}

void scenes::stream::render(const XrFrameState & frame_state)
{
	if (state_ == state::shutdown)
		application::pop_scene();

	display_time_phase = frame_state.predictedDisplayPeriod ? frame_state.predictedDisplayTime % frame_state.predictedDisplayPeriod : 0;
	display_time_period = frame_state.predictedDisplayPeriod;
	real_display_period = last_display_time ? frame_state.predictedDisplayTime - last_display_time : frame_state.predictedDisplayPeriod;
	last_display_time = frame_state.predictedDisplayTime;

	// Kiosk/chord input runs above the no-video early return below: the
	// waiting overlay must work (and the chord must stay alive) while the
	// server sends no video for an idle session.
	read_actions();

	update_stream_kiosk(frame_state.predictedDisplayTime);

	if (application::get_config().enable_stream_gui)
	{
		XrActionStateGetInfo get_info{
		        .type = XR_TYPE_ACTION_STATE_GET_INFO,
		        .action = kiosk_chord_left_action,
		};

		XrActionStateBoolean state_1{XR_TYPE_ACTION_STATE_BOOLEAN};
		CHECK_XR(xrGetActionStateBoolean(session, &get_info, &state_1));
		get_info.action = kiosk_chord_right_action;
		XrActionStateBoolean state_2{XR_TYPE_ACTION_STATE_BOOLEAN};
		CHECK_XR(xrGetActionStateBoolean(session, &get_info, &state_2));

		// Click semantics with a time gate: arm while both sticks are held,
		// fire once on release. A press edge never refires while held, which
		// stranded anyone still holding through the first toggle; the gate
		// still eats switch bounce around the release.
		bool chord_down = state_1.currentState and state_2.currentState;
		if (!state_1.isActive || !state_2.isActive)
		{
			static bool chord_inactive_warned = false;
			if (!chord_inactive_warned)
			{
				chord_inactive_warned = true;
				spdlog::warn("Kiosk chord actions inactive, thumbstick toggle unavailable");
			}
		}
		if (chord_down)
		{
			stream_chord_held = true;
		}
		else if (stream_chord_held and
		         frame_state.predictedDisplayTime - last_stream_kiosk_toggle > 500'000'000LL)
		{
			stream_chord_held = false;
			last_stream_kiosk_toggle = frame_state.predictedDisplayTime;
			// Arbitrary transitions can happen from network commands
			// Ensure we can't have a set of 2 non interactable states
			if (stream_kiosk_shown)
			{
				stream_kiosk_dismissed = true;
				next_gui_status = stream_tab::settings;
			}
			else if (is_gui_interactable())
			{
				next_gui_status = stream_tab::hidden;
				stream_kiosk_dismissed = false;
				// fresh app list so the reshow decision isn't stale
				last_kiosk_app_req = frame_state.predictedDisplayTime;
				network_session->send_control(from_headset::get_running_applications{});
			}
			else if (is_interactable(stored_gui_status))
				next_gui_status = stored_gui_status;
			else
				next_gui_status = stream_tab::applications;
		}
	}

	std::shared_lock lock(decoder_mutex);
	if (not frame_state.shouldRender or decoders[0].empty() or decoders[1].empty() or state_ == state::shutdown)
	{
		// TODO: stop/restart video stream
		session.begin_frame();
		if (frame_state.shouldRender and state_ != state::shutdown)
		{
			// Idle server, no video: still submit the GUI so the waiting
			// overlay (or an open settings page) stays usable, with
			// passthrough behind the overlay mirroring the lobby kiosk.
			// Per-frame enable matches the normal path below; it resumes
			// owning passthrough state once video flows.
			if (stream_kiosk_shown)
				session.enable_passthrough(system);
			else
				session.disable_passthrough();
			render_start(stream_kiosk_shown, frame_state.predictedDisplayTime);
			draw_gui(frame_state.predictedDisplayTime, frame_state.predictedDisplayPeriod);
			render_end();
		}
		else
			session.end_frame(frame_state.predictedDisplayTime, {});
		return;
	}

	if (state_ == state::stalled)
	{
		network_session->send_control(from_headset::get_application_list{
		        .language = application::get_messages_info().language,
		        .country = application::get_messages_info().country,
		        .variant = application::get_messages_info().variant,
		});

		next_gui_status = stream_tab::hidden;
		application::pop_scene();
	}

	// Deliberately blocking: with a single in-flight command buffer
	// the CPU cannot usefully run ahead of the GPU, so this wait is
	// the correct backpressure (not a bug). Instrumented: when GPU
	// work overruns, the stall surfaces here as CPU time, and
	// fence_wait_ms names it.
	{
		auto fence_t0 = std::chrono::steady_clock::now();
		if (device.waitForFences(*fence, VK_TRUE, UINT64_MAX) == vk::Result::eTimeout)
			throw std::runtime_error("Vulkan fence timeout");
		fence_wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fence_t0).count();
	}

	// Split-submit decision: reuse the mask slot when its fence reads
	// signaled (previous mask work retired, descriptors safe to
	// rewrite); otherwise this frame records inline exactly like
	// before. Poll only, never wait: the inline fallback is always
	// available.
	bool use_async_masks = device.waitForFences(*mask_fence, VK_FALSE, 0) == vk::Result::eSuccess;
	// Set when this frame's mask CB is submitted (release gating +
	// results-source tracking below).
	bool mask_async_submitted = false;

	// We don't need those after vkWaitForFences
	current_blit_handles.fill(nullptr);

	// Poll, never wait (no eWait): the fence above already guarantees
	// last frame's queries on the common path, so this completes
	// immediately; on a skip-frame (only q0 written) the range reads
	// not-ready and last frame's values carry over instead of stalling
	// the render thread.
	if (query_pool_filled)
	{
		// Only the five written queries (0..4): the pool is sized for
		// the whole struct but the spare queries are never written,
		// and waiting on an unwritten query blocks forever (black
		// screen: the render thread stalls here after exactly one
		// frame).
		// q4 (post-first-group) is written only on inline frames; on
		// async frames the mask spans come from the mask pool below.
		// Either way only written queries are read (never wait on an
		// unwritten query).
		uint32_t main_qcount = use_async_masks ? 4 : 5;
		auto [res, timestamps2] = query_pool.getResults<uint64_t>(
		        0,
		        main_qcount,
		        main_qcount * sizeof(uint64_t),
		        sizeof(uint64_t),
		        vk::QueryResultFlagBits::e64);

		if (res == vk::Result::eSuccess)
		{
			float period = application::get_physical_device_properties().limits.timestampPeriod / 1e9f;
			last_gpu_timestamps.gpu_time = (timestamps2[1] - timestamps2[0]) * period;
			last_gpu_timestamps.defoveate_gpu_time = (timestamps2[2] - timestamps2[0]) * period;
			last_gpu_timestamps.gizmo_gpu_time = (timestamps2[3] - timestamps2[2]) * period;
			if (not use_async_masks)
			{
				last_gpu_timestamps.mask_gpu_time = (timestamps2[1] - timestamps2[3]) * period;
				last_gpu_timestamps.mask_first_gpu_time = (timestamps2[4] - timestamps2[3]) * period;
				last_gpu_timestamps.mask_rest_gpu_time = (timestamps2[1] - timestamps2[4]) * period;
			}
		}
	}

	// Mask-pool results (async path): q0 start, q1 after first group,
	// q2 end. Read only when the last mask submission went async (else
	// the pool holds older data); not-ready carries like the main pool.
	// Volume-pool results: raster/virt/compare/merge boundaries from the
	// last volume group. Same poll discipline (no eWait, carry over),
	// plus the retiring fence for the recording path: mask fence on
	// async frames, main fence (waited above) on inline frames.
	if (volume_ts_valid)
	{
		bool volume_pool_ready = not volume_ts_async or
		        device.waitForFences(*mask_fence, VK_FALSE, 0) == vk::Result::eSuccess;
		if (volume_pool_ready)
		{
			auto [vres, vtimestamps] = volume_query_pool.getResults<uint64_t>(
			        0,
			        5,
			        5 * sizeof(uint64_t),
			        sizeof(uint64_t),
			        vk::QueryResultFlagBits::e64);

			if (vres == vk::Result::eSuccess)
			{
				float period = application::get_physical_device_properties().limits.timestampPeriod / 1e9f;
				last_volume_gpu.raster = (vtimestamps[1] - vtimestamps[0]) * period;
				last_volume_gpu.virt = (vtimestamps[2] - vtimestamps[1]) * period;
				last_volume_gpu.compare = (vtimestamps[3] - vtimestamps[2]) * period;
				last_volume_gpu.merge = (vtimestamps[4] - vtimestamps[3]) * period;
			}
		}
	}

	if (mask_results_in_mask_pool)
	{
		auto [mres, mtimestamps] = mask_query_pool.getResults<uint64_t>(
		        0,
		        3,
		        3 * sizeof(uint64_t),
		        sizeof(uint64_t),
		        vk::QueryResultFlagBits::e64);

		if (mres == vk::Result::eSuccess)
		{
			float period = application::get_physical_device_properties().limits.timestampPeriod / 1e9f;
			last_gpu_timestamps.mask_gpu_time = (mtimestamps[2] - mtimestamps[0]) * period;
			last_gpu_timestamps.mask_first_gpu_time = (mtimestamps[1] - mtimestamps[0]) * period;
			last_gpu_timestamps.mask_rest_gpu_time = (mtimestamps[2] - mtimestamps[1]) * period;
		}
	}

	session.begin_frame();

	std::array<int, view_count> image_indices;

#if WIVRN_FEATURE_RENDERDOC
	renderdoc_begin(*vk_instance);
#endif
	command_buffer.reset();
	command_buffer.begin(vk::CommandBufferBeginInfo{
	        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
	});

	// Keep a reference to the resources needed to blit the images until vkWaitForFences

	command_buffer.resetQueryPool(*query_pool, 0, size_gpu_timestamps);
	command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *query_pool, 0);

	// Search for frame with desired display time on all decoders
	// If no such frame exists, use the latest frame for each decoder
	current_blit_handles = common_frame(frame_state.predictedDisplayTime);

	// Surface an explicit toast on sustained video starvation instead of
	// rendering an empty layer (reads as unexplained grey).
	bool video_starved = true;
	for (size_t i = 0; i < view_count; ++i)
	{
		if (current_blit_handles[i])
		{
			video_starved = false;
			ever_received_video = true;
		}
	}
	update_video_stall_status(video_starved and ever_received_video);
	std::array<XrPosef, view_count> pose;
	std::array<XrFovf, view_count> fov;
	std::array<wivrn::to_headset::foveation_parameter, view_count> foveation;
	bool use_alpha = false;

	std::array<stream_defoveator::input, view_count> images;
	for (size_t i = 0; i < view_count + use_alpha; ++i)
	{
		auto & blit_handle = current_blit_handles[i];
		if (not blit_handle)
		{
			if (i == view_count)
				use_alpha = false;
			continue;
		}

		blit_handle->feedback.blitted = instance.now();
		if (blit_handle->feedback.blitted - blit_handle->feedback.received_from_decoder > 1'000'000'000)
			set_state(state::stalled);
		++blit_handle->feedback.times_displayed;
		blit_handle->feedback.displayed = frame_state.predictedDisplayTime;

		use_alpha = blit_handle->view_info.alpha;

#if WIVRN_USE_V4L2
		if (blit_handle->foreign_queue_family != vk::QueueFamilyIgnored)
		{
			// decoder completed its write before handing us this image
			// acquire ownership from the external media engine before sampling it
			vk::ImageMemoryBarrier barrier{
			        .srcAccessMask = vk::AccessFlagBits::eNone,
			        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
			        .oldLayout = blit_handle->current_layout,
			        .newLayout = vk::ImageLayout::eGeneral,
			        .srcQueueFamilyIndex = blit_handle->foreign_queue_family,
			        .dstQueueFamilyIndex = queue_family_index,
			        .image = blit_handle->image,
			        .subresourceRange = {
			                .aspectMask = vk::ImageAspectFlagBits::eColor,
			                .levelCount = 1,
			                .layerCount = 1,
			        },
			};

			command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
			                               vk::PipelineStageFlagBits::eFragmentShader,
			                               {},
			                               {},
			                               {},
			                               barrier);
			blit_handle->current_layout = vk::ImageLayout::eGeneral;
		}
		else
#endif
		        if (blit_handle->current_layout == vk::ImageLayout::eUndefined)
		{
			vk::ImageMemoryBarrier barrier{
			        .srcAccessMask = vk::AccessFlagBits::eNone,
			        .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
			        .oldLayout = vk::ImageLayout::eUndefined,
			        .newLayout = vk::ImageLayout::eGeneral,
			        .image = blit_handle->image,
			        .subresourceRange = {
			                .aspectMask = vk::ImageAspectFlagBits::eColor,
			                .levelCount = 1,
			                .layerCount = 1,
			        },
			};

			command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, barrier);
			blit_handle->current_layout = vk::ImageLayout::eGeneral;
		}

		if (i < view_count)
		{
			foveation[i] = blit_handle->view_info.foveation[i];
			pose[i] = blit_handle->view_info.pose[i];
			fov[i] = blit_handle->view_info.fov[i];
			// colour image
			images[i] = {
			        .rgb = blit_handle->image_view,
			        .sampler_rgb = decoders[i].decoder->sampler(),
			        .rect_rgb = {
			                .extent = blit_handle->extent,
			        },
			        .layout_rgb = blit_handle->current_layout,
			};
		}
		else
		{
			// alpha image, must set for each view
			for (int j = 0; j < view_count; ++j)
			{
				images[j].a = blit_handle->image_view;
				images[j].sampler_a = decoders[i].decoder->sampler();
				images[j].rect_a = vk::Rect2D{
				        // in full size pixels
				        .offset = {
				                .x = j * video_stream_description->width,
				        },
				        .extent = {
				                .width = blit_handle->extent.width * 2,
				                .height = blit_handle->extent.height * 2,
				        },
				};
				images[j].layout_a = blit_handle->current_layout;
			}
		}
	}

	// Virtual depth (stream 3): never defoveated; same ownership/layout
	// barrier discipline as color so the volume compare samples defined
	// data. Bookkeeping mirrors color (blitted/displayed feed the
	// feedback loop below, which is what drives depth IDR + bitrate).
	if (auto & depth_handle = current_blit_handles[depth_stream_index])
	{
		depth_handle->feedback.blitted = instance.now();
		++depth_handle->feedback.times_displayed;
		depth_handle->feedback.displayed = frame_state.predictedDisplayTime;

#if WIVRN_USE_V4L2
		if (depth_handle->foreign_queue_family != vk::QueueFamilyIgnored)
		{
			vk::ImageMemoryBarrier barrier{
			        .srcAccessMask = vk::AccessFlagBits::eNone,
			        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
			        .oldLayout = depth_handle->current_layout,
			        .newLayout = vk::ImageLayout::eGeneral,
			        .srcQueueFamilyIndex = depth_handle->foreign_queue_family,
			        .dstQueueFamilyIndex = queue_family_index,
			        .image = depth_handle->image,
			        .subresourceRange = {
			                .aspectMask = vk::ImageAspectFlagBits::eColor,
			                .levelCount = 1,
			                .layerCount = 1,
			        },
			};

			command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
			                               vk::PipelineStageFlagBits::eFragmentShader,
			                               {},
			                               {},
			                               {},
			                               barrier);
			depth_handle->current_layout = vk::ImageLayout::eGeneral;
		}
		else
#endif
		        if (depth_handle->current_layout == vk::ImageLayout::eUndefined)
		{
			vk::ImageMemoryBarrier barrier{
			        .srcAccessMask = vk::AccessFlagBits::eNone,
			        .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
			        .oldLayout = vk::ImageLayout::eUndefined,
			        .newLayout = vk::ImageLayout::eGeneral,
			        .image = depth_handle->image,
			        .subresourceRange = {
			                .aspectMask = vk::ImageAspectFlagBits::eColor,
			                .levelCount = 1,
			                .layerCount = 1,
			        },
			};

			command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, barrier);
			depth_handle->current_layout = vk::ImageLayout::eGeneral;
		}
	}

	// Mask record/submit views: located live at predicted display time,
	// not the video frame's pose. The video layer keeps its own pose (and
	// its own runtime warp); masks warped from a stale video pose trail
	// live geometry (projected instances) and live camera under head
	// motion (black trailing edge). Each layer warps independently, so a
	// fresher mask pose strictly reduces warp error. Falls back to the
	// video pose when locating fails.
	std::array<XrPosef, view_count> mask_pose = pose;
	std::array<XrFovf, view_count> mask_fov = fov;
	// Located once per frame and reused by draw_gui below (same time +
	// space): a second locate_views is the same IPC twice.
	std::vector<XrView> live_views_for_gui;
	{
		XrSpace live_space = application::space(xr::spaces::world);
		auto [view_flags, live_views] = session.locate_views(viewconfig, frame_state.predictedDisplayTime, live_space);
		if (live_views.size() == view_count and (view_flags & XR_VIEW_STATE_POSITION_VALID_BIT) and
		    (view_flags & XR_VIEW_STATE_ORIENTATION_VALID_BIT))
		{
			for (size_t i = 0; i < view_count; ++i)
			{
				mask_pose[i] = live_views[i].pose;
				mask_fov[i] = live_views[i].fov;
			}
		}
		live_views_for_gui = std::move(live_views);
	}

	// Eye velocity history for the depth time-bias test slider: finite
	// difference of consecutive predicted mask poses (world frame).
	// Unconditional per frame so history exists when depth needs it.
	{
		XrTime now = frame_state.predictedDisplayTime;
		if (volume_have_prev_mask and now > volume_prev_mask_time)
		{
			double dt = (double)(now - volume_prev_mask_time) * 1e-9;
			if (dt > 1e-3 and dt < 0.1)
			{
				for (uint32_t view = 0; view < view_count; ++view)
				{
					const auto & pp = volume_prev_mask_pose[view].position;
					const auto & cp = mask_pose[view].position;
					volume_eye_lin_vel[view] = glm::vec3((float)((cp.x - pp.x) / dt), (float)((cp.y - pp.y) / dt), (float)((cp.z - pp.z) / dt));
					glm::quat q0(volume_prev_mask_pose[view].orientation.w, volume_prev_mask_pose[view].orientation.x,
					             volume_prev_mask_pose[view].orientation.y, volume_prev_mask_pose[view].orientation.z);
					glm::quat q1(mask_pose[view].orientation.w, mask_pose[view].orientation.x,
					             mask_pose[view].orientation.y, mask_pose[view].orientation.z);
					glm::quat dq = q1 * glm::inverse(q0);
					if (dq.w < 0)
						dq = -dq;
					float ang = 2.f * std::acos(std::clamp(dq.w, -1.f, 1.f));
					if (ang > 1e-6f)
						volume_eye_ang_vel[view] = glm::vec3(dq.x, dq.y, dq.z) * (ang / (std::sin(ang * 0.5f) * (float)dt));
					else
						volume_eye_ang_vel[view] = glm::vec3(0);
				}
			}
		}
		volume_prev_mask_pose = mask_pose;
		volume_prev_mask_time = now;
		volume_have_prev_mask = true;
	}

	// Allow the headset to time warp if we are redisplaying a frame
	if ((not application::get_hmd_traits().discard_frame) or
	    std::ranges::any_of(current_blit_handles, [](const auto & h) { return h and h->feedback.times_displayed < 2; }) or
	    is_gui_interactable())
	{
		XrExtent2Di extents[view_count];
		{
			int32_t max_width = 0;
			int32_t max_height = 0;
			for (size_t i = 0; i < view_count; ++i)
			{
				extents[i] = stream_defoveator::defoveated_size(foveation[i]);
				max_width = std::max(max_width, extents[i].width);
				max_height = std::max(max_height, extents[i].height);
			}
			if (not swapchain)
				setup_reprojection_swapchain(max_width, max_height);
			else if (swapchain.width() < max_width or swapchain.height() < max_height)
			{
				// If the defoveated image is larger than the swapchain, try to reallocate one
				try
				{
					spdlog::info("Recreating swapchain, from {}x{} to {}x{}",
					             swapchain.width(),
					             swapchain.height(),
					             max_width,
					             max_height);
					setup_reprojection_swapchain(max_width, max_height);
				}
				catch (std::exception & e)
				{
					spdlog::warn("failed to increase swapchain size");
					for (size_t i = 0; i < view_count; ++i)
					{
						extents[i].width = std::min(extents[i].width, swapchain.width());
						extents[i].height = std::min(extents[i].height, swapchain.height());
					}
				}
			}
		}
		bool mask_frame = composition_layer_alpha_blend_supported and view_count == 2;
		std::vector<mask_group *> acquired_groups;
		std::vector<std::array<glm::vec3, 6>> cutouts;

		// Marker tracking runs every frame: solved frames feed the
		// record phase below (cutouts, gizmos) and the status UI.
		update_fiducial_passthrough(frame_state.predictedDisplayTime);

		// Depth-volume objects (3d-depth-passthrough) need the runtime
		// depth feed this frame: placed + mesh ready, any feather. The
		// sensor acquire runs at the top of the mask record phase below
		// (both async and inline paths), gated on the mask path: without
		// alpha blend there is no mask to merge into.
		bool need_volume_depth = false;
		for (auto & [oid, ost]: passthrough_objects)
		{
			if (ost.def.type == "3d-depth-passthrough" and ost.soup_ready and ost.fused.placed)
			{
				need_volume_depth = true;
				break;
			}
		}
		vk::ImageView volume_sensor = nullptr;
		float volume_near = 0.1f, volume_far = 10.f;
		bool have_volume_sensor = false;
		volume_want_frame = need_volume_depth;


		// Once-per-frame depth acquire, shared by the volume chain and the
		// debug viz. The spec allows a single acquire per frame
		// (XR_ERROR_LIMIT_REACHED beyond that), so both consumers read
		// this one frame: volume takes the sensor view, the viz takes it
		// whole. Runs ahead of both record paths (async + inline).
		bool have_depth_frame = false;
		xr::environment_depth::frame shared_depth{};
		bool want_depth = (need_volume_depth and mask_frame) or debug_depth_overlay;
		if (want_depth)
		{
			if (not env_depth_probed)
			{
				env_depth_probed = true;
				env_depth_supported = xr::environment_depth::supported(instance, system);
				if (env_depth_supported)
					env_depth = std::make_unique<xr::environment_depth>(instance, session, system, device);
				else
				{
					spdlog::info("environment depth unsupported on this headset");
					depth_status = "environment depth unsupported on this headset";
				}
			}
			if (env_depth)
			{
				env_depth->ensure();
				if (env_depth->ready())
				{
					env_depth->start();
					auto acq_t0 = std::chrono::steady_clock::now();
					bool got_depth = env_depth->acquire(application::space(xr::spaces::world), frame_state.predictedDisplayTime, shared_depth);
					double acq_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - acq_t0).count();
					if (need_volume_depth and mask_frame)
						volume_acquire_ms += acq_ms;
					if (got_depth)
						have_depth_frame = true;
					else if (debug_depth_overlay)
						depth_status = "waiting for depth data";
				}
				else if (debug_depth_overlay)
					depth_status = "depth provider unavailable";
				}
		}
		else if (env_depth and env_depth->started())
		{
			env_depth->stop();
			depth_status.clear();
		}

		// Fresh frame accumulators for the Statistics plots (filled by
		// the gizmo and mask record phases below, sampled by
		// accumulate_metrics). fence_wait_ms is measured at frame top
		// above, so it is not reset here; gui_ms/submit_ms are
		// measured after this point and reset for freshness.
		mask_raster_ms = mask_erode_ms = mask_blur_ms = mask_gizmo_ms = 0;
		volume_acquire_ms = volume_raster_ms = volume_compare_ms = volume_merge_ms = 0;
		mask_draws = 0;
		gui_ms = submit_ms = 0;
		cutouts_punched = false;

		// Rotation + B2 lateral homographies (one per eye, shared by all
		// groups): H = Suv Qd Rcw Rew^-1 Qe^-1 E1 mapping eye UV to sensor
		// UV, exact for rotation; plus per-eye T with 1/depth = a*D + b
		// for sideways parallax (forward/radial + depth-value terms
		// excluded — the experiment). Built from the renderer's own
		// matrices (P rows straight out of scene::projection_matrix, R
		// from the view matrices) so no NDC/fov-sign convention is
		// assumed: whatever the house projection does, the raster and the
		// remap agree by construction. Wd carries the validated sensor
		// flip (identity reproduces today's direct tap). Default is that
		// flip matrix, so a degenerate frustum degrades to unrotated
		// instead of breaking.
		// The Passthrough tab can disable the reprojection (direct tap);
		// the defaults below are then what the shader receives, unchanged.
		// Timed into the compare CPU (the warp's CPU cost lives there).
		std::array<glm::vec4, 6> volume_remap{};
		std::array<glm::vec4, 2> volume_remap_corr{};
		for (int e = 0; e < 2; ++e)
		{
			volume_remap[3 * e + 0] = glm::vec4(1, 0, 0, 0);
			volume_remap[3 * e + 1] = glm::vec4(0, -1, 0, 0);
			volume_remap[3 * e + 2] = glm::vec4(0, 1, 1, 0);
			}
		if (have_depth_frame and need_volume_depth and mask_frame and volume_remap_enabled)
		{
			auto remap_t0 = std::chrono::steady_clock::now();
			// E1 (UV->NDC): Vulkan NDC(-1,-1) is the top-left pixel
			// (rasterizer truth, same as blur.vert; the old branch's eye
			// mapping is v_uv*2-1 straight, no eye-side flip).
			const glm::mat3 E1(glm::vec3(2, 0, 0), glm::vec3(0, 2, 0), glm::vec3(-1, -1, 1));
			// Sensor UV<-NDC: the flip lives on the depth side only
			// (anchored to validated direct sampling: overlay top samples
			// depth row 1). Identity reproduces today's (u, 1-v) tap.
			const glm::mat3 Suv(glm::vec3(0.5f, 0, 0), glm::vec3(0, -0.5f, 0), glm::vec3(0.5f, 0.5f, 1));
			bool h_ok = true;
			// Time-bias test: assume the depth content is bias_ms older than
			// its pose claims (+bias = older = larger correction, the
			// trailing direction). Extrapolate the reported depth pose
			// backward along head motion; bias 0 leaves it untouched.
			// Clamps (50mm, ~11deg) bound prediction glitches, not heads.
			std::array<XrPosef, 2> depth_pose{shared_depth.pose[0], shared_depth.pose[1]};
			if (volume_time_bias_ms != 0.f)
			{
				float ds = volume_time_bias_ms * 1e-3f;
				for (uint32_t view = 0; view < view_count; ++view)
				{
					glm::vec3 shift = volume_eye_lin_vel[view] * ds;
					float shift_len = glm::length(shift);
					if (shift_len > 0.05f)
						shift *= 0.05f / shift_len;
					auto & dp = depth_pose[view].position;
					const auto & rp = shared_depth.pose[view].position;
					dp.x = rp.x - shift.x;
					dp.y = rp.y - shift.y;
					dp.z = rp.z - shift.z;
					float rate = glm::length(volume_eye_ang_vel[view]);
					if (rate > 1e-4f)
					{
						float ang = rate * ds;
						if (ang > 0.2f or ang < -0.2f)
							ang = std::copysign(0.2f, ang);
						glm::vec3 axis = volume_eye_ang_vel[view] / rate;
						glm::quat dq = glm::angleAxis(-ang, axis);
						const auto & ro = shared_depth.pose[view].orientation;
						glm::quat qr(ro.w, ro.x, ro.y, ro.z);
						glm::quat qu = dq * qr;
						depth_pose[view].orientation = {qu.x, qu.y, qu.z, qu.w};
					}
				}
			}
			for (uint32_t view = 0; view < view_count; ++view)
			{
				// Q maps a camera-frame direction to [clip.x, clip.y,
				// clip.w] using the projection's own rows (x, y, w of
				// columns 0..2) — whatever convention P implements.
				auto Q_of = [](const glm::mat4 & P) {
					return glm::mat3(glm::vec3(P[0][0], P[0][1], P[0][3]), glm::vec3(P[1][0], P[1][1], P[1][3]),
					                   glm::vec3(P[2][0], P[2][1], P[2][3]));
				};
				glm::mat3 Qe = Q_of(scene::projection_matrix(mask_fov[view]));
				glm::mat3 Qd = Q_of(scene::projection_matrix(shared_depth.fov[view]));
				if (std::abs(glm::determinant(Qe)) < 1e-9f or std::abs(glm::determinant(Qd)) < 1e-9f)
				{
					h_ok = false;
					break;
				}
				glm::mat3 Rew = glm::mat3(scene::view_matrix(mask_pose[view]));
				glm::mat3 Rcw = glm::mat3(scene::view_matrix(depth_pose[view]));
				glm::mat3 H = Suv * Qd * Rcw * glm::inverse(Rew) * glm::inverse(Qe) * E1;
				for (int c = 0; c < 3; ++c)
					volume_remap[3 * view + c] = glm::vec4(H[c], 0);
				// B2 lateral: T maps 1/depth (1/m) to sensor-UV shift,
				// forward (qt.z) deliberately dropped here. a, b give
				// 1/depth = a*D + b for the sensor D encoding.
				{
					const auto & pe = mask_pose[view].position;
					const auto & pcap = depth_pose[view].position;
					glm::vec3 dw(pe.x - pcap.x, pe.y - pcap.y, pe.z - pcap.z);
					glm::vec3 tc = Rcw * dw;
					glm::vec3 qt = Qd * tc;
					float n = shared_depth.near_z;
					float f = shared_depth.far_z;
					float a = 0, b = 0;
					// Same infinite-far convention as volume_raster.frag:
					// !(f > n) or huge f means D = 1 - n/d.
					if (n > 1e-6f and (!(f > n) or f > 1e9f))
					{
						a = -1.f / n;
						b = 1.f / n;
					}
					else if (n > 1e-6f and f > n)
					{
						a = -(f - n) / (f * n);
						b = 1.f / n;
					}
					else
					{
						h_ok = false;
						break;
					}
					volume_remap_corr[view] = glm::vec4(0.5f * qt.x, -0.5f * qt.y, a, b);
				}
			}
			if (not h_ok)
			{
				for (int e = 0; e < 2; ++e)
				{
					volume_remap[3 * e + 0] = glm::vec4(1, 0, 0, 0);
					volume_remap[3 * e + 1] = glm::vec4(0, -1, 0, 0);
					volume_remap[3 * e + 2] = glm::vec4(0, 1, 1, 0);
					volume_remap_corr[e] = glm::vec4(0, 0, 0, 0);
				}
			}
			if (not volume_remap_logged)
			{
				volume_remap_logged = true;
				spdlog::info("Volume remap eye0 cols: ({:.3f},{:.3f},{:.3f}) ({:.3f},{:.3f},{:.3f}) ({:.3f},{:.3f},{:.3f}) txy ({:.4f},{:.4f}) ab ({:.3f},{:.3f})",
				             (double)volume_remap[0].x, (double)volume_remap[0].y, (double)volume_remap[0].z,
				             (double)volume_remap[1].x, (double)volume_remap[1].y, (double)volume_remap[1].z,
				             (double)volume_remap[2].x, (double)volume_remap[2].y, (double)volume_remap[2].z,
				             (double)volume_remap_corr[0].x, (double)volume_remap_corr[0].y,
				             (double)volume_remap_corr[0].z, (double)volume_remap_corr[0].w);
				}
			volume_compare_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - remap_t0).count();
		}

		// View-only transforms for the mask record below. Live
		// predicted views (mask_pose/mask_fov), not the video frame's
		// pose: see the mask_pose block above.
		std::array<glm::mat4, view_count> world_mvp;
		for (uint32_t view = 0; view < view_count; ++view)
			world_mvp[view] = scene::projection_matrix(mask_fov[view]) * scene::view_matrix(mask_pose[view]);

		// Mask record phase, factored to run at two points: async on
		// its own command buffer ahead of the video wait (masks need
		// only tracking + live views, not video), or inline exactly
		// where it always ran. Only the recording CB, query pool and
		// split-stamp index vary. The main CB never touches mask
		// outputs, so no cross-submit semaphore is needed: the
		// compositor joins via swapchain release (gated on the mask
		// fence below).
		auto record_masks_phase = [&](vk::raii::CommandBuffer & rec_cmd, vk::raii::QueryPool & rec_pool, uint32_t qsplit) {
			// Volume sensor from the shared frame (acquired once above).
			// Content age + index rate feed the Volume heartbeat: age is
			// the deferred pose-correction signal, the index rate is the
			// effective sensor Hz (the runtime republishes at a fraction
			// of display rate, so most frames re-lock the same image).
			if (have_depth_frame and need_volume_depth and mask_frame)
			{
				volume_sensor = env_depth->depth_view(shared_depth.swapchain_index);
				volume_near = shared_depth.near_z;
				volume_far = shared_depth.far_z;
				have_volume_sensor = volume_sensor != vk::ImageView{};
				if (have_volume_sensor)
				{
					++volume_win_acquired;
					if (shared_depth.capture_time != 0)
					{
						double age_ms = (frame_state.predictedDisplayTime - shared_depth.capture_time) * 1e-6;
						volume_last_age_ms = (float)age_ms;
						volume_win_age_sum += age_ms;
						volume_win_age_max = std::max(volume_win_age_max, age_ms);
					}
					if (shared_depth.swapchain_index != volume_last_index)
					{
						volume_last_index = shared_depth.swapchain_index;
						++volume_win_new_content;
					}
				}
			}
			// q4 splits the mask span after the first recorded group (set
			// below; fallback write after the loop keeps the poll range
			// valid on group-less frames). Groups record sequentially, so
			// first = q4-q3 localizes one group's chain, rest = q1-q4.
			bool mask_split_written = false;
			// Fresh validity for this frame's volume stamps (consumed at
			// next frame top; groups set it below when volume records).
			volume_ts_valid = false;
			if (mask_frame)
			{
				int mw = std::max(64, (extents[0].width + 32) / 64 * 64);
				int mh = std::max(64, (extents[0].height + 32) / 64 * 64);
				// Member swapchains submit below full (tiered): tier-0 stays
				// full (exact hard edge), tier-1 submits half, tiered submits
				// quarter (V writes the submitted size, the compositor
				// upscales). Intermediates stay keyed off full mw/mh.
				int hw = std::max(64, mw / 2);
				int hh = std::max(64, mh / 2);
				int qw = std::max(64, mw / 4);
				int qh = std::max(64, mh / 4);
				// Marker window cutouts: raw pose of every live vote,
				// expanded by the debug window. Fully manual path (no blur):
				// the punch below is hard-edged by construction. Fills
				// the hoisted cutouts vector (read at submit below).
				if (debug_overlays and debug_window_mm >= 0)
				{
					for (const auto & v: fiducial_votes)
					{
						if (v.extents.width <= 0 or v.extents.height <= 0)
							continue;
						glm::quat mq(v.observed.orientation.w, v.observed.orientation.x, v.observed.orientation.y, v.observed.orientation.z);
						glm::vec3 c(v.observed.position.x, v.observed.position.y, v.observed.position.z);
						glm::vec3 r = mq * glm::vec3(1, 0, 0);
						glm::vec3 u = mq * glm::vec3(0, 1, 0);
						float hw = v.extents.width * 0.5f + debug_window_mm / 1000.f;
						float hh = v.extents.height * 0.5f + debug_window_mm / 1000.f;
						glm::vec3 v0 = c - r * hw - u * hh;
						glm::vec3 v1 = c + r * hw - u * hh;
						glm::vec3 v2 = c + r * hw + u * hh;
						glm::vec3 v3 = c - r * hw + u * hh;
						cutouts.push_back({v0, v1, v2, v0, v2, v3});
					}
				}
				// Fresh cutouts punch camera with no geometry to map it
				// onto: the flat feed must stay submitted (see submit).
				cutouts_punched = not cutouts.empty();
				// Claim groups + sync member soups (upload on hash change).
				// Groups are keyed (feather, union): A (true) drives the
				// projected layer, B (false) the flat layer. Record order
				// is A ascending, then B ascending (submit matches).
				std::set<float> live_a, live_b;
				// Pre-warmed (map-known, not necessarily placed) combos.
				// Kept by the stale-erase below alongside live_groups, but
				// kept OUT of live_a/live_b: those drive record order and
				// the debug-cutout fallback, which must reflect placed
				// objects only.
				std::set<std::pair<float, bool>> prewarmed_keys;
				// Pre-warm pass (first-visible-frame hitch fix): claim
				// renderer + swapchain + intermediates for every (feather,
				// union) combo in the map, before anything is placed or
				// uploaded. Placement frames then only rasterize. Combos
				// with nothing placed record no draws and stay inactive.
				// Mirrors the storage blocks in the record loop (which
				// become no-ops once pre-warmed).
				for (auto & [oid, ost]: passthrough_objects)
				{
					// Depth volumes share their feather's flat group (they
					// merge into its seed; surface projection never applies).
					bool pdepth = ost.def.type == "3d-depth-passthrough";
					if (ost.def.type != "3d-passthrough" and not pdepth)
						continue;
					float pf = ost.def.feather_px;
					bool pproj = surface_projected_effective(oid, ost);
					prewarmed_keys.emplace(pf, pproj);
					auto pkey = std::make_pair(pf, pproj);
					auto & pg = mask_groups[pkey];
					pg.feather_px = pf;
					pg.projected = pproj;
					if (not pg.renderer)
					{
						pg.renderer = std::make_unique<feather_mask_renderer>(device, physical_device, swapchain.format());
					}
					int ptier = feather_mask_renderer::tier_for_feather(pf);
					int pgw = (ptier == 0) ? mw : (ptier == 1) ? hw : qw;
					int pgh = (ptier == 0) ? mh : (ptier == 1) ? hh : qh;
					if (not pg.swapchain or pg.swapchain.width() != pgw or pg.swapchain.height() != pgh)
					{
						device.waitIdle();
						pg.swapchain = xr::swapchain(instance, session, device, swapchain.format(), pgw, pgh, 1, view_count);
						pg.renderer->reset_targets();
						spdlog::info("Fiducial mask swapchain: {}x{} (feather {}px union {}, {} images)", pgw, pgh, pf, pproj ? "A" : "B", pg.swapchain.image_count());
					}
					pg.renderer->ensure_targets({(uint32_t)mw, (uint32_t)mh});
				}
				for (auto & [oid, ost]: passthrough_objects)
				{
					bool is_depth_vol = ost.def.type == "3d-depth-passthrough";
					if ((ost.def.type != "3d-passthrough" and not is_depth_vol) or not ost.soup_ready or not ost.fused.placed)
						continue;
					float f = ost.def.feather_px;
					bool proj = surface_projected_effective(oid, ost);
					(proj ? live_a : live_b).insert(f);
					auto key = std::make_pair(f, proj);
					auto & g = mask_groups[key];
					g.feather_px = f;
					g.projected = proj;
					if (not g.renderer)
					{
						g.renderer = std::make_unique<feather_mask_renderer>(device, physical_device, swapchain.format());
					}
					auto mh = g.mesh_hashes.find(oid);
					if (mh == g.mesh_hashes.end() or mh->second != ost.soup_hash)
					{
						g.renderer->set_mesh(oid, ost.soup);
						g.mesh_hashes[oid] = ost.soup_hash;
					}
				}
				// Reversed objects (negatives) cut every group of both
				// unions: appended after all positives (the renderer
				// resolves positives first, then depth-tested MIN cuts).
				std::vector<std::pair<float, bool>> live_groups;
				for (float f: live_a)
					live_groups.emplace_back(f, true);
				for (float f: live_b)
					live_groups.emplace_back(f, false);
				for (auto & [oid, ost]: passthrough_objects)
				{
					if (ost.def.type != "3d-passthrough-reversed" or not ost.soup_ready or not ost.fused.placed)
						continue;
					for (auto & key: live_groups)
					{
						auto git = mask_groups.find(key);
						if (git == mask_groups.end() or not git->second.renderer)
							continue;
						auto & g = git->second;
						auto mh = g.mesh_hashes.find(oid);
						if (mh == g.mesh_hashes.end() or mh->second != ost.soup_hash)
						{
							g.renderer->set_mesh(oid, ost.soup);
							g.mesh_hashes[oid] = ost.soup_hash;
						}
					}
				}
				// Debug-only cutout group (no objects): lets the marker
				// window punch with a fiducials-only config. Claimed
				// whenever cutouts exist but no B group does (real B groups
				// punch cutouts themselves, so no extra group then).
				// Promoted to the boundary-veil carrier while the veil is
				// active: claimed whenever veiled so a B canvas always
				// exists to lift to the uniform floor (boundary-only /
				// projected-only / all-culled configs). The depth debug
				// fullscreen and the passthrough effects demo claim it
				// the same way. Empty draws (clear-only record); the
				// submit-time bias lifts it.
				bool need_debug = (live_b.empty() and not cutouts.empty()) or boundary_veil or depth_fullscreen_passthrough() or session.effects_fullscreen_passthrough();
				if (need_debug)
				{
					auto & g = debug_group;
					if (not g.renderer)
						g.renderer = std::make_unique<feather_mask_renderer>(device, physical_device, swapchain.format());
				}
				else
					debug_group.active = false;
				// Permanent record-order log: group costs are positional, so
				// the per-group loop order below is load-bearing diagnostics.
				// Logged on change only (config/map edits, anchor flapping).
				{
					static std::vector<std::pair<float, bool>> last_live_feathers;
					if (live_groups != last_live_feathers)
					{
						last_live_feathers = live_groups;
						std::string order;
						for (auto [f, proj]: live_groups)
							order += std::to_string(f) + (proj ? "A " : "B ");
						spdlog::info("Fiducial mask groups order: {}", order);
					}
				}
				// Drop stale groups + stale member meshes. Use-then-advance:
				// touching git->second after ++git reads past end().
				for (auto git = mask_groups.begin(); git != mask_groups.end();)
				{
					// Pre-warmed combos are kept like live ones (they become
					// live on placement without re-creating storage).
					if (std::ranges::find(live_groups, git->first) == live_groups.end() and
					    prewarmed_keys.find(git->first) == prewarmed_keys.end())
					{
						// Rare path: swapchain images may be in flight.
						device.waitIdle();
						git = mask_groups.erase(git);
						continue;
					}
					auto & g = git->second;
					for (auto hit = g.mesh_hashes.begin(); hit != g.mesh_hashes.end();)
					{
						auto oit = passthrough_objects.find(hit->first);
						// Reversed objects live in every group (no feather of
						// their own): evict on removal/unready only, never on
						// feather/union mismatch. Positives must match both
						// feather and union: a flag flip evicts from the old
						// union's group (remove_mesh) so no frame ever
						// rasterizes the object in both unions (ghost).
						bool member = oit != passthrough_objects.end() and oit->second.soup_ready and
						        oit->second.fused.placed and
						        (oit->second.def.type == "3d-passthrough-reversed" or
						         (oit->second.def.feather_px == g.feather_px and
						          surface_projected_effective(oit->first, oit->second) == g.projected) or
						         (oit->second.def.type == "3d-depth-passthrough" and
						          oit->second.def.feather_px == g.feather_px and not g.projected));
						if (not member)
						{
							g.renderer->remove_mesh(hit->first);
							hit = g.mesh_hashes.erase(hit);
						}
						else
							++hit;
					}
					++git;
				}
				// Record per group: draws from placed objects (union-scoped),
				// full opacity. Cutouts punch union B only (flat): no geometry
				// exists to map them onto, so A must never see them.
				static const std::vector<std::array<glm::vec3, 6>> no_cutouts;
				// Box-coverage diagnostics share one 5s window: the flag is
				// latched before the loop so every group reports in the same
				// frame (a per-group throttle starves all but the first).
				static XrTime last_box_log = 0;
				bool log_boxes = false;
				if (instance.now() - last_box_log > 5'000'000'000LL)
				{
					last_box_log = instance.now();
					log_boxes = true;
				}
				for (auto [f, proj]: live_groups)
				{
					auto & g = mask_groups[{f, proj}];
					// Eager setup: member swapchain + intermediates ready
					// before first draws, so first-visible frames pay
					// record+submit only. Tier-0 groups keep a full-res
					// swapchain (exact hard edge); tier-1 submits half,
					// tiered submits quarter (V writes the submitted size,
					// the compositor upscales). g.extent tracks the submitted
					// size; record() still works intermediates at full.
					int tier = feather_mask_renderer::tier_for_feather(f);
					int gw = (tier == 0) ? mw : (tier == 1) ? hw : qw;
					int gh = (tier == 0) ? mh : (tier == 1) ? hh : qh;
					if (not g.swapchain or g.swapchain.width() != gw or g.swapchain.height() != gh)
					{
						// Rare path (first frame, feather/config change):
						// nothing outstanding (previous images released last
						// frame), mirroring setup_reprojection_swapchain.
						device.waitIdle();
						g.swapchain = xr::swapchain(instance, session, device, swapchain.format(), gw, gh, 1, view_count);
						g.renderer->reset_targets();
						spdlog::info("Fiducial mask swapchain: {}x{} (feather {}px union {}, {} images)", gw, gh, f, proj ? "A" : "B", g.swapchain.image_count());
					}
					g.renderer->ensure_targets({(uint32_t)mw, (uint32_t)mh});
					std::vector<feather_mask_renderer::instance_draw> draws;
					// Frustum skip: drop draws fully outside both eyes (the
					// empty-draws path below then skips acquire/record for
					// fully-culled groups, which cost nothing). Conservative
					// and exact: per eye, skip only if every corner is either
					// behind the camera or outside a common side; straddles
					// are always kept. The NDC box grows by the feather bleed
					// + 1px so blur contributions survive culling.
					float margin_x = (f * 0.5f + 1.f) / ((float)mw * 0.5f);
					float margin_y = (f * 0.5f + 1.f) / ((float)mh * 0.5f);
					auto draw_visible = [&](const std::string & mesh, const std::array<glm::mat4, 2> & draw_mvp) {
						auto bounds = g.renderer->mesh_bounds(mesh);
						if (not bounds)
							return true;
						for (uint32_t view = 0; view < 2; ++view)
						{
							bool any_front = false, any_behind = false, any_inside = false;
							for (int corner = 0; corner < 8; ++corner)
							{
								glm::vec4 q((corner & 1) ? bounds->second.x : bounds->first.x,
								            (corner & 2) ? bounds->second.y : bounds->first.y,
								            (corner & 4) ? bounds->second.z : bounds->first.z, 1);
								glm::vec4 c = draw_mvp[view] * q;
								if (c.w < 0)
								{
									any_behind = true;
									continue;
								}
								any_front = true;
								if (glm::abs(c.x) <= c.w * (1.f + margin_x) and
								    glm::abs(c.y) <= c.w * (1.f + margin_y) and c.z >= 0 and c.z <= c.w)
									any_inside = true;
							}
							bool skip_eye = not any_front or (not any_behind and not any_inside);
							if (not skip_eye)
								return true;
						}
						return false;
					};
					// One draw per placed object: model matrix from the fused
					// placement, per-eye MVPs, full opacity. Polarity rides
					// the draw (false = fill, true = cut); the renderer pushes
					// full cut strength for reversed.
					auto append_draw = [&](const std::string & oid, const passthrough_object_state & ost, bool reversed,
					                       std::vector<feather_mask_renderer::instance_draw> & out) {
						const auto & fu = ost.fused;
						if (not fu.placed)
							return;
						glm::quat q(fu.world_pose.orientation.w, fu.world_pose.orientation.x,
						            fu.world_pose.orientation.y, fu.world_pose.orientation.z);
						glm::vec3 t(fu.world_pose.position.x, fu.world_pose.position.y, fu.world_pose.position.z);
						glm::mat4 model = glm::translate(glm::mat4(1), t) * glm::mat4_cast(q) *
						                  glm::scale(glm::mat4(1), glm::vec3(fu.world_scale.x, fu.world_scale.y, fu.world_scale.z));
						std::array<glm::mat4, 2> mvp;
						for (uint32_t view = 0; view < view_count; ++view)
							mvp[view] = world_mvp[view] * model;
						constexpr float op = 1;
						if (not draw_visible(oid, mvp))
							return;
						out.push_back({oid, mvp, op, reversed});
					};
					for (auto & [oid, ost]: passthrough_objects)
					{
						if (ost.def.type != "3d-passthrough" or not ost.soup_ready or not ost.fused.placed or ost.def.feather_px != f or
						    surface_projected_effective(oid, ost) != proj)
							continue;
						append_draw(oid, ost, false, draws);
					}
					// Depth volumes for this feather's flat group: same
					// eye-space MVPs, no reversed polarity (the renderer
					// MAXes them over the cuts instead of cutting).
					std::vector<feather_mask_renderer::instance_draw> volume_draws;
					bool any_volume = false;
					if (have_volume_sensor and not proj)
					{
						for (auto & [oid, ost]: passthrough_objects)
						{
							if (ost.def.type != "3d-depth-passthrough" or not ost.soup_ready or not ost.fused.placed or
							    ost.def.feather_px != f)
								continue;
							append_draw(oid, ost, false, volume_draws);
						}
						any_volume = not volume_draws.empty();
					}
					// Reversed objects (negatives) cut every group of both
					// unions: appended after all positives (the renderer
					// resolves positives first, then depth-tested MIN cuts).
					// Same mvp/frustum path, full cut strength.
					for (auto & [oid, ost]: passthrough_objects)
					{
						if (ost.def.type != "3d-passthrough-reversed" or not ost.soup_ready or not ost.fused.placed)
							continue;
						append_draw(oid, ost, true, draws);
					}
					// Reversed-only draws rasterize to a provable no-op, so a
					// group with no positive draws is skipped exactly like an
					// empty one.
					bool any_positive = std::ranges::any_of(draws, [](const feather_mask_renderer::instance_draw & d) { return !d.reverse; });
					if ((draws.empty() or not any_positive) and not any_volume)
					{
						g.active = false;
						// Mesh-pipeline edge forensics: draws present ->
						// absent means the object stopped rasterizing
						// (frustum-culled out of view vs state loss).
						// Throttled: frame-edge flicker would spam.
						if (g.had_draws and instance.now() - g.last_draws_log > 1'000'000'000LL)
						{
							g.last_draws_log = instance.now();
							spdlog::info("Fiducial mask draws lost (feather {}px union {})", f, proj ? "A" : "B");
						}
						g.had_draws = false;
						continue;
					}
					g.had_draws = true;
					int mask_index = g.swapchain.acquire();
					// (wait clocked below: mask image waits are silent
					// below timeout and sit inside the cpu measurement)
					// Paired from acquisition: the group joins the release
					// list immediately, so wait/record throws release its
					// image instead of leaking it (one leaked image per frame
					// drains the pool in pool-depth frames, then acquires fail
					// permanently).
					acquired_groups.push_back(&g);
					if (not g.swapchain.wait(100'000'000))
					{
						// Never park forever on an unavailable image: release
						// the untouched acquisition to keep pairing and skip
						// the group this frame.
						g.swapchain.release();
						acquired_groups.pop_back();
						g.active = false;
						if (not g.wait_warned)
						{
							g.wait_warned = true;
							spdlog::warn("Fiducial mask image wait timed out, skipping group (feather {}px union {})", f, proj ? "A" : "B");
						}
						continue;
					}
					g.wait_warned = false;
					g.acquired = true;
					g.active = true;
					g.extent = {(int32_t)gw, (int32_t)gh};
					feather_mask_renderer::mask_stage_cpu mstats{};
					bool vol_virt_paired = false;
					try
					{
						feather_mask_renderer::volume_input vol_in;
						if (any_volume)
						{
							vol_in.draws = volume_draws;
							vol_in.remap_cols = volume_remap;
							vol_in.remap_corr = volume_remap_corr;
							vol_in.sensor_view = volume_sensor;
							vol_in.near_z = volume_near;
							vol_in.far_z = volume_far;
							// Virtual depth: strict pairing already resolved
							// in current_blit_handles[3] (null when the
							// color frame had no depth warp).
							if (auto & dh = current_blit_handles[depth_stream_index])
							{
								vol_in.virt_view = dh->image_view;
								vol_in.virt_sampler = decoders[depth_stream_index].decoder->sampler();
								vol_in.virt_valid = true;
								vol_virt_paired = true;
							}
							if (*volume_query_pool)
								vol_in.timestamp_pool = *volume_query_pool;
						}
						g.renderer->record(rec_cmd, g.swapchain.image(mask_index), {(uint32_t)mw, (uint32_t)mh}, draws,
						                   true, f, proj ? no_cutouts : cutouts, world_mvp, &mstats, {(uint32_t)gw, (uint32_t)gh},
						                   any_volume ? &vol_in : nullptr);
					}
					catch (std::exception & e)
					{
						spdlog::warn("Fiducial mask record failed: {} (feather {}px union {})", e.what(), f, proj ? "A" : "B");
						g.swapchain.release();
						acquired_groups.pop_back();
						g.acquired = false;
						g.active = false;
					}
					// Cutout punch (debug-only) folds into raster.
					mask_raster_ms += mstats.raster_ms + mstats.punch_ms;
					mask_erode_ms += mstats.erode_ms;
					mask_blur_ms += mstats.blur_ms;
					volume_raster_ms += mstats.volume_raster_ms;
					volume_virt_ms += mstats.volume_virt_ms;
					volume_compare_ms += mstats.volume_compare_ms;
					volume_merge_ms += mstats.volume_merge_ms;
					if (any_volume and vol_virt_paired)
						++volume_win_virt_paired;
					mask_draws += (int)mstats.draws;
					if (any_volume)
					{
						// Last-writer-wins: matches the GPU pool, and a
						// single feather value is the baseline case anyway.
						volume_ts_valid = true;
						volume_last_tier = tier;
						volume_last_feather = f;
						volume_last_draws = (int)volume_draws.size();
						volume_last_box_pct = (float)(mstats.volume_box_frac * 100.0);
						volume_last_seed_w = (uint32_t)gw;
						volume_last_seed_h = (uint32_t)gh;
						// Depth-vs-eye pose divergence at record time: the
						// time-difference the future delay compensation
						// must absorb (eye 0; both eyes share the warp).
						{
							const auto & dp = shared_depth.pose[0];
							const auto & mp = mask_pose[0];
							float ex = dp.position.x - mp.position.x;
							float ey = dp.position.y - mp.position.y;
							float ez = dp.position.z - mp.position.z;
							volume_last_dpos_mm = std::sqrt(ex * ex + ey * ey + ez * ez) * 1000.f;
							float odot = std::abs(dp.orientation.w * mp.orientation.w +
							              dp.orientation.x * mp.orientation.x +
							              dp.orientation.y * mp.orientation.y +
							              dp.orientation.z * mp.orientation.z);
							volume_last_dang_deg = 2.f * std::acos(std::clamp(odot, 0.f, 1.f)) * 57.29578f;
							volume_win_dpos_sum += volume_last_dpos_mm;
							volume_win_dang_sum += volume_last_dang_deg;
						}
						if (env_depth)
						{
							volume_sensor_w = env_depth->depth_width();
							volume_sensor_h = env_depth->depth_height();
						}
						volume_sensor_near = volume_near;
						volume_sensor_far = volume_far;
					}
					// Box-coverage diagnostics for the scissor cull (see
					// flag above; tier-0 has no scissored passes to report).
					if (log_boxes and mstats.tier > 0)
					{
						spdlog::info("MaskBox: feather {}px union {} box {:.0f}% tier {} draws {}",
						             f, proj ? "A" : "B", mstats.box_frac * 100.f, mstats.tier, (int)mstats.draws);
					}
					if (not mask_split_written)
					{
						mask_split_written = true;
						rec_cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *rec_pool, qsplit);
					}
				}
				if (not mask_split_written)
				{
					// No group recorded (all culled/skipped): zero-length
					// first span, so the poll range stays valid and mask
					// numbers carry over instead of going stale.
					mask_split_written = true;
					rec_cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *rec_pool, qsplit);
				}
				// Debug group record (cutouts only, full-res, clear + hard
				// punch): runs while claimed above. During the veil (or
				// the depth/effects debug fullscreen) it also runs with
				// empty cutouts (clear-only): the promoted carrier lifted
				// to the uniform floor at submit.
				if ((live_b.empty() and not cutouts.empty()) or boundary_veil or depth_fullscreen_passthrough() or session.effects_fullscreen_passthrough())
				{
					auto & g = debug_group;
					if (g.extent.width != mw or g.extent.height != mh)
					{
						// Match the mask-group recreation discipline above:
						// wait for in-flight GPU work before destroying
						// the swapchain out from under it.
						device.waitIdle();
						g.swapchain = xr::swapchain(instance, session, device, swapchain.format(), mw, mh, 1, 2);
						g.extent = {(int32_t)mw, (int32_t)mh};
						g.renderer->reset_targets();
					}
					int mask_index = g.swapchain.acquire();
					if (not g.swapchain.wait(100'000'000))
					{
						g.swapchain.release();
						g.active = false;
						cutouts_punched = false;
						if (not g.wait_warned)
						{
							g.wait_warned = true;
							spdlog::warn("Debug mask image wait timed out, skipping frame");
						}
					}
					else
					{
						g.wait_warned = false;
						g.acquired = true;
						g.active = true;
						acquired_groups.push_back(&g);
						feather_mask_renderer::mask_stage_cpu mstats{};
						try
						{
							// Full-res submit (feather 0 = tier 0): the hard
							// punch needs no blur intermediates.
							vk::Extent2D mex{(uint32_t)mw, (uint32_t)mh};
							g.renderer->record(rec_cmd, g.swapchain.image(mask_index), mex, {}, true, 0,
							                   cutouts, world_mvp, &mstats, mex);
						}
						catch (std::exception & e)
						{
							spdlog::warn("Debug mask record failed: {}", e.what());
							g.swapchain.release();
							acquired_groups.erase(std::remove(acquired_groups.begin(), acquired_groups.end(), &g), acquired_groups.end());
							g.acquired = false;
							g.active = false;
							cutouts_punched = false;
						}
						mask_raster_ms += mstats.raster_ms + mstats.punch_ms;
						mask_erode_ms += mstats.erode_ms;
						mask_blur_ms += mstats.blur_ms;
						volume_raster_ms += mstats.volume_raster_ms;
						volume_compare_ms += mstats.volume_compare_ms;
						volume_merge_ms += mstats.volume_merge_ms;
						mask_draws += (int)mstats.draws;
					}
				}
				else
					debug_group.active = false;
			}
		};

		// Split-submit the mask phase ahead of the video wait when the
		// slot is free (fence polled at frame top): GPU mask work then
		// overlaps the CPU video stall instead of idling through it,
		// and the post-video tail is only defoveate + submit. Uniform
		// submit (even empty: stamps only) keeps the fence cycling and
		// the mask queries fresh.
		if (mask_frame && use_async_masks)
		{
			mask_command_buffer.reset();
			mask_command_buffer.begin(vk::CommandBufferBeginInfo{
			        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
			});
			mask_command_buffer.resetQueryPool(*mask_query_pool, 0, 3);
			mask_command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *mask_query_pool, 0);
			record_masks_phase(mask_command_buffer, mask_query_pool, 1);
			mask_command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *mask_query_pool, 2);
			mask_command_buffer.end();
			vk::SubmitInfo mask_submit_info;
			mask_submit_info.setCommandBuffers(*mask_command_buffer);
			device.resetFences(*mask_fence);
			queue.lock()->submit(mask_submit_info, *mask_fence);
			mask_async_submitted = true;
			mask_results_in_mask_pool = true;
			volume_ts_async = true;
		}
		assert(swapchain);
		// defoveate the image, apply scale/bias
		int image_index = swapchain.acquire();
		if (not swapchain.wait(100'000'000))
		{
			// Never park forever on an unavailable video image: the
			// command buffer so far holds only the query reset + q0
			// stamp, so end it and submit bare to keep the fence
			// cycling, hand the acquired image back, and submit an
			// empty frame. Next frame's query poll carries the GPU
			// numbers over the unwritten q1..q3.
			if (not video_wait_warned)
			{
				video_wait_warned = true;
				spdlog::warn("Video swapchain image wait timed out, skipping frame");
			}
			command_buffer.end();
			vk::SubmitInfo skip_submit;
			skip_submit.setCommandBuffers(*command_buffer);
			device.resetFences(*fence);
			queue.lock()->submit(skip_submit, *fence);
#if WIVRN_FEATURE_RENDERDOC
			renderdoc_end(*vk_instance);
#endif
			swapchain.release();
			// Masks already went out async ahead of the video wait:
			// release immediately (submission-before-release is the
			// runtime contract; completion is never waited on the
			// frame path) so the degraded frame still ends promptly.
			if (mask_async_submitted)
			{
				for (auto * g: acquired_groups)
				{
					g->acquired = false;
					g->swapchain.release();
				}
				acquired_groups.clear();
			}
			session.end_frame(frame_state.predictedDisplayTime, {});
			return;
		}
		video_wait_warned = false;

		switch (gui_status)
		{
			case stream_tab::hidden:
			case stream_tab::foveation_settings:
			case stream_tab::compact:
			case stream_tab::overlay_only:
				dimming = dimming - frame_state.predictedDisplayPeriod / (1e9 * constants::stream::fade_duration);
				break;
			case stream_tab::stats:
			case stream_tab::settings:
			case stream_tab::applications:
			case stream_tab::application_launcher:
				dimming = dimming + frame_state.predictedDisplayPeriod / (1e9 * constants::stream::fade_duration);
				break;
		}

		dimming = std::clamp<float>(dimming, 0, 1);
		float x = dimming * dimming * (3 - 2 * dimming); // Easing function

		const float scale = std::lerp(1, constants::stream::dimming_scale, x);
		const float bias = std::lerp(0, constants::stream::dimming_bias, x);

		defoveator->defoveate(command_buffer,
		                      foveation,
		                      images,
		                      {scale, scale, scale, 1.},
		                      {bias, bias, bias, 0.},
		                      image_index);
		// GPU span point: closes the defoveate span (q2-q0), opens the
		// gizmo gap below. Unconditional (defoveate runs every frame).
		command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *query_pool, 2);

#if WIVRN_USE_V4L2
		// the blit handles keep CAPTURE buffers alive until this cmd buffer's fence has completed
		inplace_vector<vk::ImageMemoryBarrier, decoder_count> foreign_release_barriers;
		for (const auto & handle: current_blit_handles)
		{
			if (handle && handle->foreign_queue_family != vk::QueueFamilyIgnored)
			{
				foreign_release_barriers.push_back({
				        .srcAccessMask = vk::AccessFlagBits::eShaderRead,
				        .dstAccessMask = vk::AccessFlagBits::eNone,
				        .oldLayout = handle->current_layout,
				        .newLayout = vk::ImageLayout::eGeneral,
				        .srcQueueFamilyIndex = queue_family_index,
				        .dstQueueFamilyIndex = handle->foreign_queue_family,
				        .image = handle->image,
				        .subresourceRange = {
				                .aspectMask = vk::ImageAspectFlagBits::eColor,
				                .levelCount = 1,
				                .layerCount = 1,
				        },
				});
				handle->current_layout = vk::ImageLayout::eGeneral;
			}
		}
		if (!foreign_release_barriers.empty())
		{
			command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
			                               vk::PipelineStageFlagBits::eBottomOfPipe,
			                               {},
			                               {},
			                               {},
			                               foreign_release_barriers);
		}
#endif

		// QR debug gizmos: tag boxes (green = configured, red = other),
		// axes tripods, orange fused-corrected boxes, board origin +
		// smoothed tripods. One transparent projection layer, staged
		// here and submitted after the camera layers below.
		int dbg_image = -1;
		if (debug_overlays)
		{
			auto gizmo_t0 = std::chrono::steady_clock::now();
			if (not gizmo_lines)
				gizmo_lines = std::make_unique<debug_lines_renderer>(device);
			// Reset the append cursor: all records in a frame share one
			// upload, and without the reset later memcpys would clobber
			// earlier draws' data (same buffer, draw at submit). Missing
			// this freezes the layer once the staging cap trips.
			gizmo_lines->begin_frame();
			using vtx = debug_lines_renderer::vertex;
			std::vector<vtx> tris;
			tris.reserve(512);
			XrTime dbg_now = instance.now();
			// Single map latch for the whole gizmo build below
			// (was one lock per section per frame).
			auto fid_map = fiducial_map.lock();
			auto tri = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, float r, float g, float bl, float al) {
				tris.push_back({{a.x, a.y, a.z}, {r, g, bl, al}});
				tris.push_back({{b.x, b.y, b.z}, {r, g, bl, al}});
				tris.push_back({{c.x, c.y, c.z}, {r, g, bl, al}});
			};
			// Filled quad (double-sided: cull is off, so no facing math).
			auto box_fill = [&](glm::vec3 c, glm::vec3 r, glm::vec3 u, float hw, float hh, float cr, float cg, float cb, float ca) {
				glm::vec3 v0 = c - r * hw - u * hh, v1 = c + r * hw - u * hh;
				glm::vec3 v2 = c + r * hw + u * hh, v3 = c - r * hw + u * hh;
				tri(v0, v1, v2, cr, cg, cb, ca);
				tri(v0, v2, v3, cr, cg, cb, ca);
			};
			// Shaft from o along dir, width across wdir (flat quad, 2 tris).
			auto shaft = [&](glm::vec3 o, glm::vec3 dir, glm::vec3 wdir, float len, float wdt, float cr, float cg, float cb, float ca) {
				glm::vec3 e = o + dir * len, s = wdir * (wdt * 0.5f);
				tri(o - s, o + s, e + s, cr, cg, cb, ca);
				tri(o - s, e + s, e - s, cr, cg, cb, ca);
			};
			auto tripod = [&](glm::vec3 o, glm::vec3 x, glm::vec3 y, glm::vec3 z, float len, float wdt, float a) {
				shaft(o, x, y, len, wdt, 1.f, 0.f, 0.f, a);
				shaft(o, y, x, len, wdt, 0.f, 1.f, 0.f, a);
				shaft(o, z, x, len, wdt, 0.f, 0.f, 1.f, a);
			};
			auto tag_frame = [&](const XrPosef & p, glm::vec3 & o, glm::vec3 & x, glm::vec3 & y, glm::vec3 & z) {
				glm::quat q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
				o = glm::vec3(p.position.x, p.position.y, p.position.z);
				x = q * glm::vec3(1, 0, 0);
				y = q * glm::vec3(0, 1, 0);
				z = q * glm::vec3(0, 0, 1);
			};
			// Configured payloads, for green vs red.
			auto is_configured = [&](const std::string & payload) {
				for (const auto & f: fid_map->fiducials)
				{
					for (const auto & m: f.markers)
					{
						if (m.marker_data == payload)
							return true;
					}
				}
				return false;
			};
			// Green boxes come from votes (one sighting per marker via
			// its own group's tracker, duplicates collapsed): the same
			// physical code is reported by every group tracker, so raw
			// sightings would draw it N times at N slightly different
			// poses. Red boxes (unmatched codes, no correct group) use
			// deduped raw sightings, newest wins.
			if (debug_matched)
			{
				for (const auto & v: fiducial_votes)
				{
					if (v.extents.width <= 0 or v.extents.height <= 0)
						continue;
					float a = std::clamp(debug_opacity, 0.f, 1.f) * stale_hold(dbg_now, v.time);
					if (a <= 0.01f)
						continue;
					glm::vec3 c, x, y, z;
					tag_frame(v.observed, c, x, y, z);
					box_fill(c, x, y, v.extents.width * 0.5f, v.extents.height * 0.5f, 0.f, 1.f, 0.f, a);
					++dbg_boxes;
					if (debug_axes and v.extents.width > 0)
						tripod(c, x, y, z, v.extents.width, v.extents.width * 0.08f, a);
				}
			}
			if (debug_unmatched)
			{
				// Newest sighting per unmatched payload across all
				// trackers (group trackers plus the no-config discovery
				// vehicle): one physical code, one red box.
				struct red_code
				{
					XrPosef pose{{0, 0, 0, 1}, {0, 0, 0}};
					XrExtent2Df extents{0, 0};
					XrTime time = 0;
				};
				std::map<std::string, red_code> red;
				for (auto & [tkey, tr]: fiducial_trackers)
				{
					(void)tkey;
					for (const auto & s: tr.sightings())
					{
						if (is_configured(s.payload))
							continue;
						auto it = red.find(s.payload);
						if (it == red.end() or s.time > it->second.time)
							red[s.payload] = {s.pose, s.extents, s.time};
					}
				}
				for (const auto & [payload, rc]: red)
				{
					(void)payload;
					if (rc.extents.width <= 0 or rc.extents.height <= 0)
						continue;
					float a = std::clamp(debug_opacity, 0.f, 1.f) * stale_hold(dbg_now, rc.time);
					if (a <= 0.01f)
						continue;
					glm::vec3 c, x, y, z;
					tag_frame(rc.pose, c, x, y, z);
					box_fill(c, x, y, rc.extents.width * 0.5f, rc.extents.height * 0.5f, 1.f, 0.f, 0.f, a);
					++dbg_boxes;
					if (debug_axes and rc.extents.width > 0)
						tripod(c, x, y, z, rc.extents.width, rc.extents.width * 0.08f, a);
				}
			}
			if (debug_corrected)
			{
				// Orange fused-corrected boxes: each configured marker
				// re-projected from its fiducial's smoothed pose with the
				// authored offset (the inverse of the vote path). Green
				// raw vs orange corrected shows disagreement at a glance.
				for (const auto & f: fid_map->fiducials)
				{
					auto bpit = board_poses.find(f.id);
					if (bpit == board_poses.end() or not bpit->second.live)
						continue;
					for (const auto & m: f.markers)
					{
						if (m.marker_size_m <= 0)
							continue;
						float a = std::clamp(debug_opacity, 0.f, 1.f) *
						          stale_hold(dbg_now, bpit->second.time);
						if (a <= 0.01f)
							continue;
						glm::quat bq(bpit->second.pose.orientation.w, bpit->second.pose.orientation.x, bpit->second.pose.orientation.y, bpit->second.pose.orientation.z);
						glm::vec3 bp_(bpit->second.pose.position.x, bpit->second.pose.position.y, bpit->second.pose.position.z);
						glm::quat oq(m.orientation[3], m.orientation[0], m.orientation[1], m.orientation[2]);
						glm::vec3 op(m.position[0], m.position[1], m.position[2]);
						// Authored offsets are tag-in-board: tag = board *
						// offset directly (the vote path inverts once).
						glm::quat tq = bq * oq;
						glm::vec3 c = bp_ + bq * op;
						glm::vec3 x = tq * glm::vec3(1, 0, 0);
						glm::vec3 y = tq * glm::vec3(0, 1, 0);
						glm::vec3 z = tq * glm::vec3(0, 0, 1);
						box_fill(c, x, y, m.marker_size_m * 0.5f, m.marker_size_m * 0.5f, 1.f, 0.55f, 0.f, a);
						++dbg_boxes;
					}
				}
			}
			if (debug_origin or debug_smoothed)
			{
				for (const auto & [fid, bp]: board_poses)
				{
					(void)fid;
					float a = std::clamp(debug_opacity, 0.f, 1.f) * stale_hold(dbg_now, bp.time);
					if (a <= 0.01f)
						continue;
					// Board origin: use the largest configured tag as the
					// scale reference (offsets are meaningless without one).
					float ref_size = 0;
					for (const auto & f: fid_map->fiducials)
					{
						if (f.id != fid)
							continue;
						for (const auto & m: f.markers)
							ref_size = std::max(ref_size, m.marker_size_m);
					}
					if (ref_size <= 0)
						continue;
					if (debug_origin)
					{
						// Fused origin tripod + white square (offset check).
						glm::vec3 o, x, y, z;
						tag_frame(bp.pose, o, x, y, z);
						float len = ref_size * 1.5f;
						tripod(o, x, y, z, len, len * 0.05f, a);
						float d = len * 0.25f;
						box_fill(o, x, y, d, d, 1.f, 1.f, 1.f, a);
						++dbg_boxes;
					}
					if (debug_smoothed)
					{
						// Smoothed twin (smaller, cyan square).
						glm::vec3 o, x, y, z;
						tag_frame(bp.pose, o, x, y, z);
						float len = ref_size * 1.0f;
						tripod(o, x, y, z, len, len * 0.06f, a);
						float d = len * 0.25f;
						box_fill(o, x, y, d, d, 0.f, 1.f, 1.f, a);
						++dbg_boxes;
					}
				}
			}
			dbg_tris = tris.size() / 3;
						// Own swapchain: acquire once when gizmos will record. Skip
			// the frame on timeout (paired release inside, like masks).
			if (not tris.empty())
			{
				gizmo_lines->ensure(instance, session, swapchain.format(), swapchain.width(), swapchain.height());
				dbg_image = gizmo_lines->acquire();
			}
			if (dbg_image >= 0 and not tris.empty())
			{
				std::array<vk::Extent2D, 2> eye_ext;
				std::array<glm::mat4, 2> eye_mvp;
				for (uint32_t view = 0; view < view_count; ++view)
				{
					eye_ext[view] = {(uint32_t)extents[view].width, (uint32_t)extents[view].height};
					eye_mvp[view] = world_mvp[view];
				}
				gizmo_lines->record(command_buffer, (size_t)dbg_image, eye_ext, eye_mvp, tris.data(), tris.size());
			}
			mask_gizmo_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gizmo_t0).count();
		}

		// Environment-depth debug overlay: records the fullscreen
		// false-color viz from the shared frame (acquired once above) into
		// its own transparent swapchain. Provider cost is bound to the
		// toggle + volume need: the shared site above stops it below.
		int depth_image = -1;
		if (debug_depth_overlay and have_depth_frame)
		{
			if (not depth_overlay)
				depth_overlay = std::make_unique<depth_overlay_renderer>(device);
			depth_overlay->ensure(instance, session, swapchain.format(), swapchain.width(), swapchain.height());
			depth_image = depth_overlay->acquire();
			if (depth_image >= 0)
			{
				depth_overlay_renderer::frame_params params;
				params.near_z = shared_depth.near_z;
				params.far_z = shared_depth.far_z;
				params.opacity = debug_depth_opacity;
				std::array<vk::Extent2D, 2> depth_ext;
				for (uint32_t view = 0; view < view_count; ++view)
					depth_ext[view] = {(uint32_t)extents[view].width, (uint32_t)extents[view].height};
				depth_overlay->record(command_buffer, (size_t)depth_image, depth_ext,
				                      env_depth->depth_view(shared_depth.swapchain_index), params);
				depth_status = fmt::format("live {}x{} near {:.2f}m far {:.2f}m",
				                           env_depth->depth_width(), env_depth->depth_height(),
				                           (double)shared_depth.near_z, (double)shared_depth.far_z);
			}
			else
				depth_status = "overlay swapchain busy, skipped frame";
		}
		// else: status already set by the shared acquire site above
		// (unsupported / unavailable / waiting).

		// Debug mask record: the local debug group (union B) carries the
		// marker-window cutouts. No silhouette raster yet (no objects):
		// clear + hard cutout punch at full alpha.
		// GPU span point: closes the gizmo gap (q3-q2). On inline
		// frames it also opens the mask span (closed by query 1 after
		// the phase); on async frames the mask spans live in the mask
		// pool and this just measures submit prep. Unconditional so no
		// validity tracking is needed.
		command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *query_pool, 3);
		if (!mask_async_submitted)
		{
			// Inline fallback (mask fence busy): identical work on the
			// main command buffer. Throttled note — persistent fallback
			// means the GPU never drains between frames.
			if (mask_frame && !use_async_masks)
			{
				static XrTime last_async_fallback_log = 0;
				XrTime now = instance.now();
				if (now - last_async_fallback_log > 5'000'000'000LL)
				{
					last_async_fallback_log = now;
					spdlog::info("Mask submit: inline fallback (mask fence busy)");
				}
			}
			record_masks_phase(command_buffer, query_pool, 4);
			mask_results_in_mask_pool = false;
			volume_ts_async = false;
		}

		command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *query_pool, 1);

		command_buffer.end();
		vk::SubmitInfo submit_info;
		submit_info.setCommandBuffers(*command_buffer);

		inplace_vector<vk::Semaphore, decoder_count> semaphores;
		inplace_vector<uint64_t, decoder_count> semaphore_vals;
		inplace_vector<vk::PipelineStageFlags, decoder_count> wait_stages;
		for (auto b: current_blit_handles)
		{
			if (b and b->semaphore)
			{
				assert(b->semaphore_val);
				semaphores.push_back(b->semaphore);
				semaphore_vals.push_back(*b->semaphore_val);
				wait_stages.push_back(vk::PipelineStageFlagBits::eFragmentShader);
			}
		}
		submit_info.setWaitDstStageMask(wait_stages);
		submit_info.setWaitSemaphores(semaphores);
		vk::TimelineSemaphoreSubmitInfo sem_info{
		        .waitSemaphoreValueCount = uint32_t(semaphore_vals.size()),
		        .pWaitSemaphoreValues = semaphore_vals.data(),
		};
		submit_info.pNext = &sem_info;

		device.resetFences(*fence);
		queue.lock()->submit(submit_info, *fence);
#if WIVRN_FEATURE_RENDERDOC
		renderdoc_end(*vk_instance);
#endif
		swapchain.release();
		// Release immediately after submit: submission-before-release
		// is the runtime contract (device-side sync is the runtime's
		// job — the pre-split code released here with mask work still
		// in flight). Never wait for mask completion on the frame
		// path: parking the thread here delays end_frame past vsync
		// on spike frames (single black frames). Slot safety comes
		// from the frame-top fence poll, not from this wait.
		// Paired with the group acquires above: the compositor takes
		// the last released image, so release right after submit and
		// reference the swapchains in the layers below.
		for (auto * g: acquired_groups)
		{
			g->acquired = false;
			g->swapchain.release();
		}
		acquired_groups.clear();

		bool have_fiducial_map = false;
		{
			auto map = fiducial_map.lock();
			have_fiducial_map = not map->fiducials.empty() or not map->objects.empty();
		}
		// The debug window cutout needs a live passthrough to reveal,
		// map or not (it punches every live vote). The depth debug
		// overlay and the passthrough effects demo need fullscreen
		// passthrough underneath them.
		bool want_debug_cutout = debug_overlays and debug_window_mm >= 0;
		bool effects_full = session.effects_fullscreen_passthrough();
		if (use_alpha or session.has_projected_passthrough_mesh() or have_fiducial_map or want_debug_cutout or depth_fullscreen_passthrough() or effects_full)
			session.enable_passthrough(system);
		else
			session.disable_passthrough();

		// Mask path owns the display when the alpha-blend extension is
		// present (stereo only, like the mask targets): video, then one
		// union pass each (A projected, B flat), then debug/GUI. Without
		// it, previous behavior is untouched (flat auto-submit), except the
		// boundary veil hard switch forces passthrough when outside.
		bool veil_hard_nomask = ((boundary_veil and boundary_s < 0.5f) or depth_fullscreen_passthrough() or effects_full) and not mask_frame;
		render_start(mask_frame ? false : (use_alpha or veil_hard_nomask), frame_state.predictedDisplayTime);

		// Add the layer with the streamed content
		std::array<XrCompositionLayerProjectionView, view_count> layer_view;
		for (uint32_t view = 0; view < view_count; view++)
		{
			layer_view[view] =
			        {
			                .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
			                .pose = pose[view],
			                .fov = fov[view],

			                .subImage = {
			                        .swapchain = swapchain,
			                        .imageRect = {
			                                .offset = {0, 0},
			                                .extent = extents[view],
			                        },
			                        .imageArrayIndex = view,
			                },
			        };
		}
		add_projection_layer(
		        // Mask-blend mode drives the window from the feather mask;
		        // server alpha (if any) is ignored so the two never double-cut.
		        (use_alpha and not mask_frame) ? XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT : 0,
		        application::space(xr::spaces::world),
		        layer_view);

		if (mask_frame)
		{
			// Split-union compositing: union A (projected content) drives
			// the projected layer, union B (the rest) drives the flat
			// layer, composited projected-below-flat so flat wins overlaps.
			// Video (opaque) was already added above, then each union's
			// masks, then the camera layers, then debug/GUI. The single
			// dst-alpha channel carries one union at a time: the first mask
			// of each union REPLACEs destination alpha (color preserved),
			// the rest ADD (saturating union). B's REPLACE resets the alpha
			// the projected layer consumed, so flat reads only B.
			//
			// EITHER/OR camera layers: co-submitting flat reconstruction and
			// projected mapping breaks video compositing on current runtimes
			// (video goes black; instantly reversible), so exactly one camera
			// layer submits per frame.
			//
			// This branch has no window groups yet (no objects, no blur):
			// union A is empty, union B holds only the debug cutout group.
			// Debug cutouts punch union B only (no geometry to map onto),
			// so the flat feed must stay submitted.
			//
			// With objects: union A holds the surface-projected windows
			// (drives the projected layer), union B the rest (drives the
			// flat layer). Flat wins whenever anything needs it:
			// unprojected windows, debug cutouts (camera with no geometry
			// to map it onto). Otherwise the projected layer alone carries
			// every projected window.
			bool need_flat = cutouts_punched;
			for (auto & [oid, ost]: passthrough_objects)
			{
				if (not ost.soup_ready or not ost.fused.placed)
					continue;
				// Depth volumes reveal the flat feed wherever the sensor
				// resolves inside the volume: flat must stay submitted.
				if (ost.def.type == "3d-depth-passthrough")
				{
					need_flat = true;
					break;
				}
				if (ost.def.type != "3d-passthrough")
					continue;
				if (not surface_projected_effective(oid, ost))
				{
					need_flat = true;
					break;
				}
			}
			if (cutouts_punched)
				need_flat = true;
			bool have_projected = session.has_projected_instances();
			size_t a_active = 0, b_active = 0;
			for (auto & [key, g]: mask_groups)
			{
				if (not g.active)
					continue;
				if (key.second)
					++a_active;
				else
					++b_active;
			}
			if (debug_group.active)
				++b_active;
			// Boundary veil (fresh values: update_fiducial_passthrough ran
			// just above): lift the flat union's floor by (1 - s) through
			// colorScaleBias on the B masks (first submitted B mask gets
			// scale+lift, the rest scale-only: ADD would multi-add the lift).
			// Skipped at full-inside so steady state stays bit-identical.
			// Without colorScaleBias (or without the mask path) the veil
			// degrades to a hard switch at s = 0.5 (opaque fullscreen flat
			// when outside, no masks needed).
			float veil_s = std::clamp(boundary_s, 0.f, 1.f);
			// Depth debug fullscreen + effects demo fullscreen: full
			// passthrough under the viz via the same B floor lift
			// (scale 0 + bias 1), no geometry.
			bool depth_full = depth_fullscreen_passthrough();
			bool effects_full_inner = session.effects_fullscreen_passthrough();
			bool full_passthrough_demo = depth_full or effects_full_inner;
			if (full_passthrough_demo)
				veil_s = 0;
			bool veil_lift = (boundary_veil or full_passthrough_demo) and veil_s < 0.999f and composition_layer_color_scale_bias_supported and composition_layer_alpha_blend_supported;
			bool veil_hard_out = ((boundary_veil or full_passthrough_demo) and veil_s < 0.5f) and not composition_layer_color_scale_bias_supported;
			// No-mask-path runtimes have no alpha control to ramp through
			// either: hard switch there too (forced in render_start below
			// for the non-mask branch; here for the mask branch without
			// colorScaleBias).
			if ((boundary_veil or full_passthrough_demo) and not composition_layer_alpha_blend_supported and boundary_s < 0.5f)
				veil_hard_out = true;
			bool submit_projected = have_projected and a_active > 0;
			bool submit_flat = (need_flat or ((boundary_veil or full_passthrough_demo) and composition_layer_color_scale_bias_supported and composition_layer_alpha_blend_supported) or veil_hard_out) and
			        (b_active > 0 or veil_hard_out);
			// Forensic submit decisions (change-triggered): which camera layer
			// submits and why. Permanent: coexistence bugs read as wrong output
			// with no error, so the decision itself must be in the log.
			{
				static bool last_submit_flat = true;
				static bool last_submit_projected = false;
				if (submit_flat != last_submit_flat or submit_projected != last_submit_projected)
				{
					last_submit_flat = submit_flat;
					last_submit_projected = submit_projected;
					spdlog::info("Camera submit: {} (need_flat={} have_projected={} A_groups={} B_groups={} cutouts={} depth_full={} effects_full={})",
					             (submit_flat and submit_projected) ? "projected+flat"
					             : submit_flat                      ? "flat"
					             : submit_projected                 ? "projected"
					                                                : "video-only",
					             need_flat,
					             have_projected,
					             a_active,
					             b_active,
					             cutouts_punched,
					             depth_full,
					             effects_full_inner);
				}
			}
			// One mask submit pass per union (A then B): each union's first
			// active group REPLACEs, the rest ADD.
			for (bool proj_union: {true, false})
			{
				// Opaque hard-outside flat needs no masks at all.
				if (veil_hard_out and not proj_union)
					continue;
				bool first_group = true;
				// Union A: window groups with projected=true (none yet).
				for (auto & [key, g]: mask_groups)
				{
					if (key.second != proj_union)
						continue;
					if (not g.active)
						continue;
					std::array<XrCompositionLayerProjectionView, view_count> mask_views;
					for (uint32_t view = 0; view < view_count; ++view)
					{
						mask_views[view] = {
						        .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
						        .pose = mask_pose[view],
						        .fov = mask_fov[view],
						        .subImage = {
						                .swapchain = g.swapchain,
						                .imageRect = {
						                        .offset = {0, 0},
						                        .extent = {g.extent.width, g.extent.height},
						                },
						                .imageArrayIndex = view,
						        },
						};
					}
					// imageIndex is implicit: the compositor uses the last
					// released image, which record() acquired above.
					add_projection_layer(XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT,
					                     application::space(xr::spaces::world),
					                     mask_views);
					if (first_group)
						set_alpha_blend(XR_BLEND_FACTOR_ZERO_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ZERO_FB);
					else
						set_alpha_blend(XR_BLEND_FACTOR_ZERO_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ONE_FB);
					if (veil_lift and not proj_union)
					{
						// Coverage·s + (1−s): window interiors stay ~1, empty
						// areas lift to the uniform veil floor, feather bands
						// interpolate. RGB untouched (ignored downstream anyway).
						if (first_group)
							set_color_scale_bias({1, 1, 1, veil_s}, {0, 0, 0, 1 - veil_s});
						else
							set_color_scale_bias({1, 1, 1, veil_s}, {0, 0, 0, 0});
					}
					first_group = false;
				}
				// Union B also carries the local debug cutout group.
				if (not proj_union and debug_group.active)
				{
					auto & g = debug_group;
					std::array<XrCompositionLayerProjectionView, view_count> mask_views;
					for (uint32_t view = 0; view < view_count; ++view)
					{
						mask_views[view] = {
						        .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
						        .pose = mask_pose[view],
						        .fov = mask_fov[view],
						        .subImage = {
						                .swapchain = g.swapchain,
						                .imageRect = {
						                        .offset = {0, 0},
						                        .extent = {g.extent.width, g.extent.height},
						                },
						                .imageArrayIndex = view,
						        },
						};
					}
					add_projection_layer(XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT,
					                     application::space(xr::spaces::world),
					                     mask_views);
					if (first_group)
						set_alpha_blend(XR_BLEND_FACTOR_ZERO_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ZERO_FB);
					else
						set_alpha_blend(XR_BLEND_FACTOR_ZERO_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ONE_FB);
					if (veil_lift and not proj_union)
					{
						if (first_group)
							set_color_scale_bias({1, 1, 1, veil_s}, {0, 0, 0, 1 - veil_s});
						else
							set_color_scale_bias({1, 1, 1, veil_s}, {0, 0, 0, 0});
					}
					first_group = false;
				}
				// Each union's camera layer consumes its own alpha: A feeds
				// projected, then B REPLACEs the alpha and feeds flat, so
				// flat wins overlaps.
				if (proj_union)
				{
					if (submit_projected)
					{
						add_projected_passthrough_layer();
						set_alpha_blend(XR_BLEND_FACTOR_DST_ALPHA_FB, XR_BLEND_FACTOR_ONE_MINUS_DST_ALPHA_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ZERO_FB);
					}
				}
				else
				{
					if (submit_flat)
					{
						add_passthrough_layer();
						// Hard-outside veil is opaque fullscreen: no blend.
						if (not veil_hard_out)
							set_alpha_blend(XR_BLEND_FACTOR_DST_ALPHA_FB, XR_BLEND_FACTOR_ONE_MINUS_DST_ALPHA_FB, XR_BLEND_FACTOR_ONE_FB, XR_BLEND_FACTOR_ZERO_FB);
					}
				}
			}
		}

		// Debug gizmo layer: ONE transparent projection layer for all debug
		// geometry (standard alpha blend: no FB extensions, works wherever
		// the old quads did). Always exactly one layer: maxLayerCount can
		// never overflow. Above passthrough (it annotates real-world tags),
		// below the GUI.
		if (dbg_image >= 0)
		{
			std::array<XrCompositionLayerProjectionView, view_count> dbg_views;
			for (uint32_t view = 0; view < view_count; ++view)
			{
				dbg_views[view] = {
				        .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
				        .pose = mask_pose[view],
				        .fov = mask_fov[view],
				        .subImage = {
				                .swapchain = gizmo_lines->swapchain(),
				                .imageRect = {
				                        .offset = {0, 0},
				                        .extent = extents[view],
				                },
				                .imageArrayIndex = view,
				        },
				};
			}
			add_projection_layer(XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT,
			                     application::space(xr::spaces::world),
			                     dbg_views);
			// Paired with the acquire in the record phase above: the
			// compositor takes the last released image; release after submit.
			gizmo_lines->release();
		}

		// Depth overlay: ONE transparent projection layer, above the
		// gizmos and below the GUI quads (submitted here, before
		// draw_gui). Same discipline as the debug lines: standard
		// alpha blend, no FB extensions.
		if (depth_image >= 0 and depth_overlay)
		{
			std::array<XrCompositionLayerProjectionView, view_count> depth_views;
			for (uint32_t view = 0; view < view_count; ++view)
			{
				depth_views[view] = {
				        .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
				        .pose = mask_pose[view],
				        .fov = mask_fov[view],
				        .subImage = {
				                .swapchain = depth_overlay->swapchain(),
				                .imageRect = {
				                        .offset = {0, 0},
				                        .extent = extents[view],
				                },
				                .imageArrayIndex = view,
				        },
				};
			}
			add_projection_layer(XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT,
			                     application::space(xr::spaces::world),
			                     depth_views);
			// Paired with the acquire above: release after submit.
			depth_overlay->release();
		}

		if (const configuration::openxr_post_processing_settings openxr_post_processing = application::get_config().openxr_post_processing;
		    (openxr_post_processing.sharpening | openxr_post_processing.super_sampling) > 0)
			set_layer_settings(openxr_post_processing.sharpening | openxr_post_processing.super_sampling);

		{
			auto gui_t0 = std::chrono::steady_clock::now();
			draw_gui(frame_state.predictedDisplayTime, frame_state.predictedDisplayPeriod, live_views_for_gui);
			gui_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gui_t0).count();
		}

		// render_end (submit + end_frame IPC) is the remaining
		// uninstrumented span inside the cpu= measurement: time it so
		// compositor-side stalls are named instead of unattributed.
		auto submit_t0 = std::chrono::steady_clock::now();
		try
		{
			render_end();
		}
		catch (std::system_error & e)
		{
			if (not (e.code().category() == xr::error_category() and e.code().value() == XR_ERROR_POSE_INVALID))
				throw;
			spdlog::info("Invalid pose submitted");
		}
		submit_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submit_t0).count();

		// Sampled after render_end so gui_ms/submit_ms above are this
		// frame's, not last frame's.
		accumulate_metrics(frame_state.predictedDisplayTime, current_blit_handles, last_gpu_timestamps);
	}

	// Network operations may be blocking, do them once everything was submitted
	{
		// Keep a copy of the feedback packets as they can be modified if they're encrypted
		inplace_vector<from_headset::feedback, decoder_count> feedbacks;
		inplace_vector<serialization_packet, decoder_count> packets;

		feedbacks.reserve(current_blit_handles.size());
		packets.reserve(current_blit_handles.size());

		for (const auto & handle: current_blit_handles)
		{
			if (handle)
			{
				auto & packet = packets.emplace_back();
				wivrn_session::control_socket_t::serialize(packet, feedbacks.emplace_back(handle->feedback));
			}
		}
		if (not packets.empty())
		{
			try
			{
				network_session->send_control(std::span(packets));
			}
			catch (std::exception & e)
			{
				spdlog::warn("Exception while sending feedback packet: {}", e.what());
			}
		}
	}

	query_pool_filled = true;
}

void scenes::stream::update_video_stall_status(bool starved)
{
	static constexpr XrDuration stall_threshold_ns = 1'000'000'000;
	static const std::string stall_text = _("No video from the server - check the XR app on the PC");
	XrTime now = instance.now();

	if (not starved)
	{
		video_starved_since = 0;
		if (video_stall_toasted)
		{
			video_stall_toasted = false;
			// Clear only our own toast; a server message may have replaced it
			auto toast = gui_toast.lock();
			if (toast->has_value() and (*toast)->content == stall_text)
				toast->reset();
		}
		return;
	}

	if (not video_starved_since)
		video_starved_since = now;
	if (not video_stall_toasted and now - video_starved_since >= stall_threshold_ns)
	{
		video_stall_toasted = true;
		spdlog::warn("Video stalled: no frames from server for over 1s");
		auto toast = gui_toast.lock();
		toast->emplace(stall_text, true);
		gui_status_last_change = instance.now();
	}
}

void scenes::stream::exit()
{
	state_ = state::shutdown;
}

void scenes::stream::setup(const to_headset::video_stream_description & description)
{
	spdlog::info("setup, refresh rate {}", description.refresh_rate);
	session.set_refresh_rate(description.refresh_rate);

	std::unique_lock lock(decoder_mutex);
	if (video_stream_description == description)
		return;
	spdlog::info("Creating decoders, size {}x{}", description.width, description.height);
	video_stream_description = description;

	for (const auto & [stream_index, item]: utils::enumerate(decoders))
	{
		item = accumulator_images{
		        .decoder = std::make_unique<shard_accumulator>(device, physical_device, instance, queue_family_index, description, weak_from_this(), stream_index),
		};
	}

	if (defoveator)
		defoveator->reset_pipelines();
}

void scenes::stream::setup_reprojection_swapchain(uint32_t swapchain_width, uint32_t swapchain_height)
{
	assert(swapchain_width);
	assert(swapchain_height);
	device.waitIdle();
	spdlog::info("swapchain setup, refresh rate {}", video_stream_description->refresh_rate);
	session.set_refresh_rate(video_stream_description->refresh_rate);

	auto views = system.view_configuration_views(viewconfig);

	swapchain = xr::swapchain(instance, session, device, swapchain_format, swapchain_width, swapchain_height, 1, views.size());
	spdlog::info("Created stream swapchain: {}x{}", swapchain.width(), swapchain.height());
	for (auto view: views)
	{
		if (swapchain.width() > view.maxImageRectWidth or swapchain.height() > view.maxImageRectHeight)
			spdlog::warn("Swapchain size larger than maximum {}x{}", view.maxImageRectWidth, view.maxImageRectHeight);
	}

	spdlog::info("Initializing reprojector");
	vk::Extent2D extent = {(uint32_t)swapchain.width(), (uint32_t)swapchain.height()};

	defoveator.emplace(
	        device,
	        physical_device,
	        swapchain.images(),
	        extent,
	        swapchain.format());

	// Retain the shared mask-pipeline cache for the session, then
	// force-build it now (first video swapchain, before any map exists).
	// Without this the first feather_mask_renderer pays ~85ms of driver
	// compiles on the first visible frame (later groups hit the cache
	// anyway).
	mask_pipe_cache = feather_mask_renderer::pipeline_cache(device);
	feather_mask_renderer::warm_pipelines(device, physical_device, swapchain.format());
}

scene::meta & scenes::stream::get_meta_scene()
{
	static meta m{
	        .name = "Stream",
	        .actions = {
	                {"left_aim", XR_ACTION_TYPE_POSE_INPUT},
	                {"left_trigger", XR_ACTION_TYPE_FLOAT_INPUT},
	                {"left_squeeze", XR_ACTION_TYPE_FLOAT_INPUT},
	                {"left_scroll", XR_ACTION_TYPE_VECTOR2F_INPUT},
	                {"left_haptic", XR_ACTION_TYPE_VIBRATION_OUTPUT},
	                {"right_aim", XR_ACTION_TYPE_POSE_INPUT},
	                {"right_trigger", XR_ACTION_TYPE_FLOAT_INPUT},
	                {"right_squeeze", XR_ACTION_TYPE_FLOAT_INPUT},
	                {"right_scroll", XR_ACTION_TYPE_VECTOR2F_INPUT},
	                {"right_haptic", XR_ACTION_TYPE_VIBRATION_OUTPUT},

	                {"kiosk_chord_left", XR_ACTION_TYPE_BOOLEAN_INPUT},
	                {"kiosk_chord_right", XR_ACTION_TYPE_BOOLEAN_INPUT},

	                {"recenter_left", XR_ACTION_TYPE_BOOLEAN_INPUT},
	                {"recenter_right", XR_ACTION_TYPE_BOOLEAN_INPUT},
	                {"gui_distance_left", XR_ACTION_TYPE_FLOAT_INPUT},
	                {"gui_distance_right", XR_ACTION_TYPE_FLOAT_INPUT},

	                {"settings_adjust", XR_ACTION_TYPE_FLOAT_INPUT},
	                {"foveation_distance", XR_ACTION_TYPE_FLOAT_INPUT},
	                {"foveation_ok", XR_ACTION_TYPE_BOOLEAN_INPUT},
	                {"foveation_cancel", XR_ACTION_TYPE_BOOLEAN_INPUT},
	        },
	        .bindings = {
	                suggested_binding{
	                        {
	                                "/interaction_profiles/oculus/touch_controller",
	                                "/interaction_profiles/facebook/touch_controller_pro",
	                                "/interaction_profiles/meta/touch_pro_controller",
	                                "/interaction_profiles/meta/touch_controller_plus",
	                                "/interaction_profiles/meta/touch_plus_controller",
	                                "/interaction_profiles/bytedance/pico_neo3_controller",
	                                "/interaction_profiles/bytedance/pico4_controller",
	                                "/interaction_profiles/bytedance/pico4s_controller",
	                                "/interaction_profiles/yvr/touch_controller_yvr",
	                                "/interaction_profiles/htc/vive_focus3_controller",
	                                "/interaction_profiles/valve/frame_controller_valve",
	                        },
	                        {
	                                {"left_aim", "/user/hand/left/input/aim/pose"},
	                                {"left_trigger", "/user/hand/left/input/trigger/value"},
	                                {"left_squeeze", "/user/hand/left/input/squeeze/value"},
	                                {"left_scroll", "/user/hand/left/input/thumbstick"},
	                                {"left_haptic", "/user/hand/left/output/haptic"},
	                                {"right_aim", "/user/hand/right/input/aim/pose"},
	                                {"right_trigger", "/user/hand/right/input/trigger/value"},
	                                {"right_squeeze", "/user/hand/right/input/squeeze/value"},
	                                {"right_scroll", "/user/hand/right/input/thumbstick"},
	                                {"right_haptic", "/user/hand/right/output/haptic"},

	                                {"recenter_left", "/user/hand/left/input/squeeze/value"},
	                                {"recenter_right", "/user/hand/right/input/squeeze/value"},
	                                {"gui_distance_left", "/user/hand/left/input/thumbstick/y"},
	                                {"gui_distance_right", "/user/hand/right/input/thumbstick/y"},
	                                {"settings_adjust", "/user/hand/right/input/thumbstick/y"},
	                                {"foveation_distance", "/user/hand/left/input/thumbstick/y"},
	                                {"foveation_ok", "/user/hand/right/input/a/click"},
	                                {"foveation_cancel", "/user/hand/right/input/b/click"},

	                                {"kiosk_chord_left", "/user/hand/left/input/thumbstick/click"},
	                                {"kiosk_chord_right", "/user/hand/right/input/thumbstick/click"},
	                                {"kiosk_chord_right", "/user/hand/right/input/thumbstick/click"},
	                        },
	                },
	                suggested_binding{
	                        {
	                                "/interaction_profiles/khr/simple_controller",
	                        },
	                        {},
	                },
	        },
	};

	return m;
}

std::optional<std::string> scenes::stream::pop_stream_error()
{
	auto queue = stream_error_queue.lock();
	if (queue->empty())
		return std::nullopt;

	auto err = std::make_optional(queue->front());
	queue->pop();

	return err;
}

void scenes::stream::on_xr_event(const xr::event & event)
{
	switch (event.header.type)
	{
		case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
			if (event.space_changed_pending.referenceSpaceType == XrReferenceSpaceType::XR_REFERENCE_SPACE_TYPE_LOCAL)
				recenter_requested = true;
			break;
		case XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB:
			network_session->send_control(from_headset::refresh_rate_changed{
			        .from = event.refresh_rate_changed.fromDisplayRefreshRate,
			        .to = event.refresh_rate_changed.toDisplayRefreshRate,
			});
			break;
		case XR_TYPE_EVENT_DATA_VISIBILITY_MASK_CHANGED_KHR:
			network_session->send_control(from_headset::visibility_mask_changed{
			        .data = get_visibility_mask(instance, session, event.visibility_mask_changed.viewIndex),
			        .view_index = uint8_t(event.visibility_mask_changed.viewIndex),
			});
			break;
		case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
			// Override session state if the GUI is interactable
			if (event.state_changed.state == XR_SESSION_STATE_FOCUSED and is_gui_interactable())
				network_session->send_control(from_headset::session_state_changed{
				        .state = XR_SESSION_STATE_VISIBLE,
				});
			else
				network_session->send_control(from_headset::session_state_changed{
				        .state = event.state_changed.state,
				});
			break;
		case XR_TYPE_EVENT_DATA_USER_PRESENCE_CHANGED_EXT: {
			bool present = (bool)event.user_presence_changed.isUserPresent;
			handle_user_presence(present);
			network_session->send_control(from_headset::user_presence_changed{
			        .present = present,
			        .change_time = instance.now(),
			});
			break;
		}
		case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
			on_interaction_profile_changed(event.interaction_profile_changed);
			break;
		default:
			break;
	}
}

bool scenes::stream::forward_hid_input(from_headset::hid::input_t packet, bool device_enabled)
{
	// hid_forwarding is whether the server permits it; device_enabled is the headset toggle.
	if (not hid_forwarding or not device_enabled)
		return false;
	network_session->send_control(from_headset::hid::input{packet});
	return true;
}

bool scenes::stream::on_input_key_down(uint8_t key_code)
{
	return forward_hid_input(from_headset::hid::key_down{key_code}, application::get_config().forward_keyboard);
}
bool scenes::stream::on_input_key_up(uint8_t key_code)
{
	return forward_hid_input(from_headset::hid::key_up{key_code}, application::get_config().forward_keyboard);
}
bool scenes::stream::on_input_mouse_move(float x, float y)
{
	return forward_hid_input(from_headset::hid::mouse_move{x, y}, application::get_config().forward_mouse);
}
bool scenes::stream::on_input_button_down(uint8_t button)
{
	return forward_hid_input(from_headset::hid::button_down{button}, application::get_config().forward_mouse);
}
bool scenes::stream::on_input_button_up(uint8_t button)
{
	return forward_hid_input(from_headset::hid::button_up{button}, application::get_config().forward_mouse);
}
bool scenes::stream::on_input_scroll(float h, float v)
{
	return forward_hid_input(from_headset::hid::mouse_scroll{h, v}, application::get_config().forward_mouse);
}
