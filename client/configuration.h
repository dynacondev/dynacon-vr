/*
 * WiVRn VR streaming
 * Copyright (C) 2024  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2024  Patrick Nicolas <patricknicolas@laposte.net>
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

#include "wivrn_discover.h"
#include "wivrn_packets.h"

#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <simdjson.h>
#include <string>
#include <type_traits>
#include <vector>
#include <openxr/openxr.h>

namespace xr
{
class session;
class system;
} // namespace xr

// key <-> member serialization descriptor, defined in configuration.cpp
struct config_field;

enum class feature
{
	microphone,
	hand_tracking,
	eye_gaze,
	face_tracking,
	body_tracking,
};

enum class posture
{
	standing,
	seated,
};

class configuration
{
public:
	struct server_data
	{
		bool autoconnect;
		bool manual;
		bool visible;
		bool compatible;

		wivrn_discover::service service;
	};

	std::map<std::string, server_data> servers;
	float preferred_refresh_rate = 0;
	std::optional<float> minimum_refresh_rate;
	float resolution_scale = 1.2;
	std::optional<wivrn::video_codec> codec;
	uint32_t bitrate_bps = 50'000'000;
	uint8_t bit_depth = 10;

	bool passthrough_enabled = false;
	// XR_META_boundary_visibility, the boundary is always shown outside passthrough
	bool passthrough_boundary_enabled = false;
	bool mic_unprocessed_audio = false;

	// Input forwarding, per device. Off by default; only effective if the server permits it.
	bool forward_keyboard = false;
	bool forward_mouse = false;
	bool forward_gamepad = false;

	// Hands-only mode (default): hands are reported as 26-joint hand
	// tracking only, never as controller input on /user/hand/*. When
	// false, simple_controller and hand_interaction bindings are withheld
	// so the runtime cannot promote hands to controllers; hand joints
	// still stream via XrHandTracker. Set true to restore legacy
	// hand-as-controller emulation (pinch driving controller input).
	bool hands_emulate_controllers = false;

	std::underlying_type_t<wivrn::from_headset::body_part_mask> body_part_mask = ~0;

	bool enable_stream_gui = true;

	// 4x MSAA for the client-side lobby/controllers renderer (transient
	// multisample images resolved into the single-sample swapchain images).
	// WIVRN_MSAA=0/4 env var overrides this. Falls back to single-sample if
	// the device does not support 4x for the swapchain formats.
	bool msaa_enabled = true;

	// Kiosk mode: at startup, when an auto-connect computer is configured,
	// show only a minimal "Waiting for Connection" overlay over passthrough
	// instead of the full lobby. Both thumbsticks reveal the full GUI.
	bool kiosk_mode = true;

	// application launcher: list vs grid, and grid icon size, 0 small 1 medium 2 large
	bool app_list_view = false;
	uint32_t app_icon_size = 0;

	// interface theme, defaults match the built-in "OLED" preset and "Orange" accent
	std::string theme_preset = "OLED";
	std::string theme_accent = "Orange";
	float theme_rounding = 8;
	float theme_card_rounding = 14;
	float theme_font_scale = 1.0;
	float theme_background_alpha = 0.75;

	// XR_FB_composition_layer_settings extension flags
	struct openxr_post_processing_settings
	{
		XrCompositionLayerSettingsFlagsFB super_sampling = 0;
		XrCompositionLayerSettingsFlagsFB sharpening = 0;
	};
	openxr_post_processing_settings openxr_post_processing{};

	// XR_FB_color_space session color space, persisted enum value,
	// -1 = unset (P3 when reported, else first enumerated mode)
	int64_t color_space = -1;

	std::string virtual_keyboard_layout = "QWERTY";

	std::string environment_model = "assets://ground.glb";

	bool override_foveation_enable = false;
	float override_foveation_pitch = -10 * M_PI / 180;
	float override_foveation_distance = 3;

	bool high_power_mode = true;
	uint32_t fps_divider = 1;

	bool usb_network = false;

	// Allow unsafe config values
	bool extended_config = false;

	bool first_run = true;

	std::string locale;

	bool check_feature(feature f) const;
	void set_feature(feature f, bool state);

	static constexpr float default_height_offset_standing = 0;
	static constexpr float default_height_offset_seated = 0.45;

private:
	mutable std::mutex mutex;
	std::map<feature, bool> features;
	std::optional<float> stream_scale;
	float height_offset_standing = default_height_offset_standing;
	float height_offset_seated = default_height_offset_seated;
	posture default_posture = posture::standing;

	// table of scalar settings shared by save()/load(); non-scalar settings are explicit
	static const std::vector<config_field> & config_fields();

public:
	configuration(xr::system &, xr::session &);
	configuration(xr::system &, xr::session &, const std::filesystem::path &);

	void save();

	void set_stream_scale(float);
	float get_stream_scale() const;
	float get_default_stream_scale() const;

	float get_height_offset() const;
	void set_height_offset(float);

	posture get_posture() const;
	void set_posture(posture);

	uint32_t max_bitrate(bool extended) const
	{
		return extended ? 800'000'000u : 200'000'000u;
	}

	uint32_t max_bitrate() const
	{
		return max_bitrate(extended_config);
	}
};
