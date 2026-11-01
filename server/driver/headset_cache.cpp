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

#include "headset_cache.h"

#include "utils/xdg_base_directory.h"

#include "util/u_logging.h"

#include <fstream>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>

namespace wivrn
{

std::filesystem::path headset_cache_file()
{
	return xdg_config_home() / "wivrn" / "last_headset.json";
}

namespace
{
constexpr int cache_version = 1;

template <typename E>
std::string enum_to_string(E value, const char * what)
{
	auto name = magic_enum::enum_name(value);
	if (name.empty())
		throw std::runtime_error(std::string("invalid ") + what);
	return std::string(name);
}

template <typename E>
E enum_from_string(const nlohmann::json & json, const char * what)
{
	auto value = magic_enum::enum_cast<E>(json.get<std::string>());
	if (not value)
		throw std::runtime_error(std::string("invalid ") + what + ": " + json.dump());
	return *value;
}

nlohmann::json settings_to_json(const from_headset::settings_changed & settings)
{
	return {
	        {"preferred_refresh_rate", settings.preferred_refresh_rate},
	        {"minimum_refresh_rate", settings.minimum_refresh_rate},
	        {"fps_divider", settings.fps_divider},
	        {"bitrate_bps", settings.bitrate_bps},
	        {"mirror_gamepad", settings.mirror_gamepad},
	        {"enabled_body_parts", settings.enabled_body_parts},
	};
}

from_headset::settings_changed settings_from_json(const nlohmann::json & json)
{
	from_headset::settings_changed settings;
	settings.preferred_refresh_rate = json.at("preferred_refresh_rate");
	settings.minimum_refresh_rate = json.at("minimum_refresh_rate");
	settings.fps_divider = json.at("fps_divider");
	settings.bitrate_bps = json.at("bitrate_bps");
	settings.mirror_gamepad = json.at("mirror_gamepad");
	settings.enabled_body_parts = json.at("enabled_body_parts");
	return settings;
}

nlohmann::json audio_to_json(const from_headset::headset_info_packet::audio_description & audio)
{
	return {
	        {"num_channels", audio.num_channels},
	        {"sample_rate", audio.sample_rate},
	};
}

from_headset::headset_info_packet::audio_description audio_from_json(const nlohmann::json & json)
{
	return {
	        .num_channels = json.at("num_channels"),
	        .sample_rate = json.at("sample_rate"),
	};
}
} // namespace

void save_headset_cache(const from_headset::headset_info_packet & info)
{
	try
	{
		nlohmann::json json{
		        {"version", cache_version},
		        {"render_eye_width", info.render_eye_width},
		        {"render_eye_height", info.render_eye_height},
		        {"stream_eye_width", info.stream_eye_width},
		        {"stream_eye_height", info.stream_eye_height},
		        {"available_refresh_rates", info.available_refresh_rates},
		        {"settings", settings_to_json(info.settings)},
		        {"hand_tracking", info.hand_tracking},
		        {"eye_gaze", info.eye_gaze},
		        {"palm_pose", info.palm_pose},
		        {"user_presence", info.user_presence},
		        {"passthrough", info.passthrough},
		        {"face_tracking", enum_to_string(info.face_tracking, "face_tracking")},
		        {"body_tracking", enum_to_string(info.body_tracking, "body_tracking")},
		        {"num_generic_trackers", info.num_generic_trackers},
		        {"system_name", info.system_name},
		        {"language", info.language},
		        {"country", info.country},
		        {"variant", info.variant},
		};

		if (info.speaker)
			json["speaker"] = audio_to_json(*info.speaker);
		else
			json["speaker"] = nullptr;

		if (info.microphone)
			json["microphone"] = audio_to_json(*info.microphone);
		else
			json["microphone"] = nullptr;

		nlohmann::json codecs = nlohmann::json::array();
		for (const auto & capability: info.supported_codecs)
		{
			codecs.push_back({
			        {"codec", enum_to_string(capability.codec, "codec")},
			        {"supports_10bit", capability.supports_10bit},
			});
		}
		json["supported_codecs"] = std::move(codecs);

		if (info.bit_depth)
			json["bit_depth"] = *info.bit_depth;
		else
			json["bit_depth"] = nullptr;

		auto file = headset_cache_file();
		std::filesystem::create_directories(file.parent_path());

		auto tmp = file;
		tmp += ".new";
		{
			std::ofstream out(tmp);
			out.exceptions(std::ofstream::failbit);
			out << json.dump(1);
		}

		std::error_code ec;
		std::filesystem::rename(tmp, file, ec);
		if (ec)
			U_LOG_W("Failed to save headset cache: %s", ec.message().c_str());
	}
	catch (const std::exception & e)
	{
		U_LOG_W("Failed to save headset cache: %s", e.what());
	}
}

std::optional<from_headset::headset_info_packet> load_headset_cache()
{
	auto file = headset_cache_file();
	if (not std::filesystem::exists(file))
		return std::nullopt;

	try
	{
		std::ifstream in(file);
		in.exceptions(std::ifstream::failbit);
		nlohmann::json json = nlohmann::json::parse(in);

		if (json.at("version") != cache_version)
		{
			U_LOG_W("Ignoring headset cache with unsupported version");
			return std::nullopt;
		}

		from_headset::headset_info_packet info;
		info.render_eye_width = json.at("render_eye_width");
		info.render_eye_height = json.at("render_eye_height");
		info.stream_eye_width = json.at("stream_eye_width");
		info.stream_eye_height = json.at("stream_eye_height");
		info.available_refresh_rates = json.at("available_refresh_rates").get<std::vector<float>>();
		info.settings = settings_from_json(json.at("settings"));

		if (not json.at("speaker").is_null())
			info.speaker = audio_from_json(json.at("speaker"));
		if (not json.at("microphone").is_null())
			info.microphone = audio_from_json(json.at("microphone"));

		info.hand_tracking = json.at("hand_tracking");
		info.eye_gaze = json.at("eye_gaze");
		info.palm_pose = json.at("palm_pose");
		info.user_presence = json.at("user_presence");
		info.passthrough = json.at("passthrough");
		info.face_tracking = enum_from_string<from_headset::face_type>(json.at("face_tracking"), "face_tracking");
		info.body_tracking = enum_from_string<from_headset::body_type>(json.at("body_tracking"), "body_tracking");
		info.num_generic_trackers = json.at("num_generic_trackers");

		for (const auto & item: json.at("supported_codecs"))
		{
			info.supported_codecs.push_back({
			        .codec = enum_from_string<video_codec>(item.at("codec"), "codec"),
			        .supports_10bit = item.at("supports_10bit"),
			});
		}

		if (not json.at("bit_depth").is_null())
			info.bit_depth = json.at("bit_depth").get<uint8_t>();

		info.system_name = json.at("system_name");
		info.language = json.at("language");
		info.country = json.at("country");
		info.variant = json.at("variant");

		// Plausibility gate: a corrupt cache must never seed a session.
		if (info.render_eye_width == 0 or info.render_eye_height == 0 or
		    info.stream_eye_width == 0 or info.stream_eye_height == 0 or
		    info.available_refresh_rates.empty() or info.supported_codecs.empty() or
		    info.system_name.empty())
		{
			U_LOG_W("Ignoring implausible headset cache");
			return std::nullopt;
		}

		return info;
	}
	catch (const std::exception & e)
	{
		U_LOG_W("Ignoring invalid headset cache: %s", e.what());
		return std::nullopt;
	}
}

} // namespace wivrn
