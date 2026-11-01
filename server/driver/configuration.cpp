/*
 * WiVRn VR streaming
 * Copyright (C) 2022-2024  Guillaume Meunier <guillaume.meunier@centraliens.net>
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

#define JSON_DISABLE_ENUM_SERIALIZATION 1

#include "configuration.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>

#include "util/u_logging.h"

#include "utils/flatpak.h"
#include "utils/xdg_base_directory.h"
#include "wivrn_config.h"

static constexpr size_t config_file_max_size = 256 * 1024;

static auto resolve_path(std::filesystem::path path)
{
	std::error_code ec;
	auto canonical = std::filesystem::canonical(path, ec);

	// path doesn't exist
	if (ec)
		return path;

	return canonical;
}

// [rx, ry, rz] degrees to xyzw quaternion. Fixed-frame rotations about X,
// then Y, then Z (q = qz * qy * qx), so single-axis values do the obvious
// thing and combined angles compose in a documented order.
static std::array<float, 4> euler_deg_to_quat(float rx_deg, float ry_deg, float rz_deg)
{
	constexpr float deg = (float)M_PI / 180.f;
	float cx = std::cos(rx_deg * deg * 0.5f), sx = std::sin(rx_deg * deg * 0.5f);
	float cy = std::cos(ry_deg * deg * 0.5f), sy = std::sin(ry_deg * deg * 0.5f);
	float cz = std::cos(rz_deg * deg * 0.5f), sz = std::sin(rz_deg * deg * 0.5f);
	auto mul = [](std::array<float, 4> a, std::array<float, 4> b) {
		return std::array<float, 4>{
		        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
		        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
		        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
		        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
		};
	};
	std::array<float, 4> qx{sx, 0, 0, cx}, qy{0, sy, 0, cy}, qz{0, 0, sz, cz};
	return mul(mul(qz, qy), qx);
}

static std::filesystem::path config_file;
static std::filesystem::path known_keys_file = resolve_path(xdg_config_home() / "wivrn" / "known_keys.json");
static std::filesystem::path cookie_file = resolve_path(xdg_config_home() / "wivrn" / "cookie");

namespace wivrn
{
NLOHMANN_JSON_SERIALIZE_ENUM(
        video_codec,
        {
                {video_codec(-1), ""},
                {h264, "h264"},
                {h264, "avc"},
                {h265, "h265"},
                {h265, "hevc"},
                {av1, "av1"},
                {av1, "AV1"},
                {raw, "raw"},
        })

NLOHMANN_JSON_SERIALIZE_ENUM(
        service_publication,
        {
                {service_publication(-1), ""},
                {service_publication::none, nullptr},
                {service_publication::none, "none"},
                {service_publication::avahi, "avahi"},
        })

void configuration::set_config_file(const std::filesystem::path & path)
{
	config_file = resolve_path(path);
}
std::filesystem::path configuration::get_config_file()
{
	if (config_file.empty())
		return resolve_path(xdg_config_home() / "wivrn" / "config.json");
	return config_file;
}

static void log_effective_config_once(const nlohmann::json & merged)
{
	// Process-once: the body is idempotent, so a benign race only risks a repeated line
	static bool logged = false;
	if (logged)
		return;
	logged = true;

	if (not merged.is_object())
	{
		U_LOG_I("Effective configuration: not an object, ignoring");
		return;
	}

	std::string keys;
	for (auto it = merged.begin(); it != merged.end(); ++it)
	{
		if (not keys.empty())
			keys += ", ";
		keys += it.key();
	}
	U_LOG_I("Effective configuration keys: %s", keys.empty() ? "(none)" : keys.c_str());
}

// One searched location for wivrn/config.json, shared by the config merge,
// the search log and the ConfigFiles D-Bus property so all three agree.
// `path` is the host-form path for display (empty when unknown); `local` is
// the sandbox-visible path used for direct reads (empty for via-host reads);
// `location` tells the dashboard which side can open it ("host"/"sandbox").
struct config_candidate
{
	std::string id;
	std::string role;
	std::string path;
	std::string location;
	std::string local;
	std::string status = "missing";
	std::optional<std::string> content;
};

static void log_config_search_once(const std::vector<config_candidate> & candidates,
                                   const std::map<std::string, std::string> & first_source,
                                   const std::map<std::string, std::string> & final_source)
{
	// Process-once: like log_effective_config_once, read_configuration runs on
	// every connection so only the first search is reported in full
	static bool logged = false;
	if (logged)
		return;
	logged = true;

	U_LOG_I("Config search (%u locations):", (unsigned)candidates.size());
	for (const auto & candidate: candidates)
	{
		std::string status = candidate.status;
		if (status == "invalid")
			status += " (see error above)";
		else if (status == "skipped")
			status += " (HOST_XDG_CONFIG_HOME and HOME unset)";
		else if (status == "missing" and candidate.local.empty())
			status = "missing or unreadable";
		U_LOG_I("Config search: %s: %s: %s",
		        candidate.role.c_str(),
		        candidate.path.empty() ? "(location unknown)" : candidate.path.c_str(),
		        status.c_str());
	}
	for (const auto & [key, winner]: final_source)
	{
		auto it = first_source.find(key);
		if (it != first_source.end() and it->second != winner)
			U_LOG_I("Config key '%s' from %s overridden by %s", key.c_str(), it->second.c_str(), winner.c_str());
	}
}

static std::vector<config_candidate> collect_config_candidates()
{
	std::vector<config_candidate> candidates;
	auto try_direct = [&](std::string id, std::string role, std::filesystem::path sandbox_path, std::string display, std::string location) {
		config_candidate candidate;
		candidate.id = std::move(id);
		candidate.role = std::move(role);
		candidate.location = std::move(location);
		auto resolved = resolve_path(sandbox_path);
		candidate.local = resolved.string();
		candidate.path = display.empty() ? candidate.local : std::move(display);
		if (std::filesystem::exists(resolved))
		{
			std::ifstream file(resolved);
			candidate.content = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
		}
		candidates.push_back(std::move(candidate));
	};
	bool flatpak = wivrn::is_flatpak();
	// HOST_XDG_CONFIG_HOME only exists when the host exports XDG_CONFIG_HOME.
	// Otherwise the host uses the XDG default, and $HOME inside the sandbox
	// names the same location as on the host, so $HOME/.config is a valid
	// host path even though it is invisible from here.
	std::string host_config_home;
	if (flatpak)
	{
		if (const char * host_xdg = std::getenv("HOST_XDG_CONFIG_HOME"); host_xdg and *host_xdg)
			host_config_home = host_xdg;
		else if (const char * home = std::getenv("HOME"); home and *home)
			host_config_home = (std::filesystem::path(home) / ".config").string();
	}
	// Bundled defaults first, then host files, then sandbox files, so
	// dashboard edits (sandbox per-app config) win. Host paths are
	// invisible from inside the flatpak sandbox: /run/host/* is
	// bind-mounted (host-os/host-etc permissions) and the host XDG
	// config is read from outside the sandbox like model files.
	try_direct("bundle",
	           "bundled defaults",
	           std::filesystem::path(WIVRN_INSTALL_PREFIX "/share") / "wivrn" / "config.json",
	           "",
	           "sandbox");
	if (flatpak)
	{
		try_direct("host-usr-share",
		           "host system data",
		           std::filesystem::path("/run/host/usr/share") / "wivrn" / "config.json",
		           "/usr/share/wivrn/config.json",
		           "host");
		try_direct("host-etc",
		           "host system config",
		           std::filesystem::path("/run/host/etc") / "wivrn" / "config.json",
		           "/etc/wivrn/config.json",
		           "host");
		if (not host_config_home.empty())
		{
			config_candidate candidate;
			candidate.id = "host-xdg";
			candidate.role = "host user config";
			candidate.path = (std::filesystem::path(host_config_home) / "wivrn" / "config.json").string();
			candidate.location = "host";
			if (auto data = wivrn::read_file_via_host(candidate.path, config_file_max_size))
				candidate.content = std::string(reinterpret_cast<const char *>(data->data()), data->size());
			candidates.push_back(std::move(candidate));
		}
		else
		{
			config_candidate candidate;
			candidate.id = "host-xdg";
			candidate.role = "host user config";
			candidate.location = "host";
			candidate.status = "skipped";
			candidates.push_back(std::move(candidate));
		}
	}
	try_direct("etc", flatpak ? "sandbox system config" : "system config", std::filesystem::path("/etc") / "wivrn" / "config.json", "", "sandbox");
	try_direct("xdg", flatpak ? "sandbox user config" : "user config", xdg_config_home() / "wivrn" / "config.json", "", "sandbox");
	return candidates;
}

nlohmann::json configuration::read_configuration()
{
	if (config_file.empty())
	{
		nlohmann::json merged;
		// First and winning source per top-level key, to spot a later file shadowing an earlier one
		std::map<std::string, std::string> key_first_source;
		std::map<std::string, std::string> key_final_source;
		auto candidates = collect_config_candidates();
		for (auto & candidate: candidates)
		{
			if (not candidate.content)
				continue;
			// Display path identifies the source; via-host reads are marked like the log lines
			std::string source = candidate.path;
			if (candidate.local.empty())
				U_LOG_I("Using configuration file %s (via host)", candidate.path.c_str());
			else
				U_LOG_I("Using configuration file %s", candidate.local.c_str());
			if (candidate.local.empty())
				source += " (via host)";
			try
			{
				for (const auto & [key, value]: nlohmann::json::parse(*candidate.content).get<nlohmann::json::object_t>())
				{
					if (not key_first_source.contains(key))
						key_first_source[key] = source;
					key_final_source[key] = source;
					merged[key] = value;
				}
				candidate.status = "used";
			}
			catch (std::exception & e)
			{
				if (candidate.local.empty())
					U_LOG_E("Invalid configuration file %s (via host): %s", candidate.path.c_str(), e.what());
				else
					U_LOG_E("Invalid configuration file %s: %s", candidate.local.c_str(), e.what());
				candidate.status = "invalid";
			}
		}
		log_config_search_once(candidates, key_first_source, key_final_source);
		log_effective_config_once(merged);
		return merged;
	}
	else
	{
		U_LOG_I("Using configuration file %s", config_file.c_str());
		try
		{
			nlohmann::json merged = nlohmann::json::parse(std::ifstream(config_file));
			log_effective_config_once(merged);
			return merged;
		}
		catch (std::exception & e)
		{
			U_LOG_E("Invalid configuration file %s: %s", config_file.c_str(), e.what());
			return nlohmann::json();
		}
	}
}

std::string configuration::config_files_json()
{
	nlohmann::json array = nlohmann::json::array();
	for (auto & candidate: collect_config_candidates())
	{
		if (candidate.content)
		{
			try
			{
				(void)nlohmann::json::parse(*candidate.content).get<nlohmann::json::object_t>();
				candidate.status = "used";
			}
			catch (std::exception &)
			{
				candidate.status = "invalid";
			}
		}
		array.push_back({{"id", candidate.id},
		                 {"path", candidate.path},
		                 {"status", candidate.status},
		                 {"location", candidate.location}});
	}
	return array.dump();
}

configuration::encoder parse_encoder(const nlohmann::json & item)
{
	configuration::encoder e;
	if (item.contains("encoder"))
		e.name = item["encoder"];

#define SET_IF(property)              \
	if (item.contains(#property)) \
		e.property = item[#property];

	SET_IF(codec);
	if (e.codec == wivrn::video_codec(-1))
		throw std::runtime_error("invalid codec value " + item["codec"].get<std::string>());
	SET_IF(options);
	SET_IF(device);
	return e;
}

configuration::configuration()
{
	try
	{
		auto json = read_configuration();

		if (auto it = json.find("grip-surface"); it != json.end())
		{
			grip_surface = *it;
		}

		if (auto it = json.find("encoder"); it != json.end())
		{
			if (it->is_array())
			{
				for (size_t i = 0; i < std::min(encoders.size(), it->size()); ++i)
				{
					encoders[i] = parse_encoder(it->at(i));
				}
			}
			else if (it->is_string())
			{
				std::ranges::fill(encoders, encoder{.name = *it});
			}
			else
			{
				std::ranges::fill(encoders, parse_encoder(*it));
			}
		}

		if (auto it = json.find("application"); it != json.end())
		{
			if (it->is_string())
				application.push_back(*it);
			else
			{
				for (const auto & i: *it)
					application.push_back(i);
			}
		}

		// Orientation helper: [rx, ry, rz] degrees (fixed-frame X, then Y,
		// then Z, see euler_deg_to_quat()) to xyzw quaternion.
		auto parse_orientation = [](const nlohmann::json & o, const char * what) {
			if (not o.is_array() or o.size() != 3)
				throw std::runtime_error(std::string("invalid ") + what + " orientation: expected [rx, ry, rz] degrees");
			return euler_deg_to_quat(o[0], o[1], o[2]);
		};

		if (auto it = json.find("fiducials"); it != json.end())
		{
			for (const auto & item: *it)
			{
				fiducial_entry e;
				if (item.contains("id"))
					e.id = item["id"];
				if (e.id.empty())
				{
					U_LOG_W("fiducial entry without id, skipping");
					continue;
				}
				if (item.contains("tag"))
					e.tag = item["tag"];
				if (item.contains("static"))
					e.is_static = item["static"];
				if (item.contains("sigma-k"))
					e.sigma_k = item["sigma-k"];
				if (item.contains("euro-min-cutoff"))
					e.euro_min_cutoff = item["euro-min-cutoff"];
				if (item.contains("euro-beta"))
					e.euro_beta = item["euro-beta"];
				if (item.contains("hold-inner-mm"))
					e.hold_inner_mm = item["hold-inner-mm"];
				if (item.contains("hold-inner-deg"))
					e.hold_inner_deg = item["hold-inner-deg"];
				if (item.contains("markers"))
				{
					for (const auto & m: item["markers"])
					{
						fiducial_marker mk;
						if (m.contains("marker-data"))
							mk.marker_data = m["marker-data"];
						if (mk.marker_data.empty())
						{
							U_LOG_W("fiducial \"%s\": marker without marker-data, skipping marker", e.id.c_str());
							continue;
						}
						if (m.contains("marker-size-m"))
							mk.marker_size_m = m["marker-size-m"];
						if (m.contains("position"))
							mk.position = m["position"];
						if (m.contains("orientation"))
							mk.orientation = parse_orientation(m["orientation"], "fiducial marker");
						e.markers.push_back(std::move(mk));
					}
				}
				if (e.markers.empty())
				{
					U_LOG_W("fiducial \"%s\": no markers, skipping", e.id.c_str());
					continue;
				}
				if (std::ranges::any_of(fiducials, [&](const fiducial_entry & f) { return f.id == e.id; }))
				{
					U_LOG_W("duplicate fiducial id \"%s\", skipping", e.id.c_str());
					continue;
				}
				fiducials.push_back(std::move(e));
			}
		}

		// Gates the uinput mirror of forwarded input devices. The OpenXR gamepad needs no
		// permission, so it is always exposed.
		if (auto it = json.find("hid-forwarding"); it != json.end())
			hid_forwarding = *it;

		if (auto it = json.find("passthrough"); it != json.end())
		{
			for (const auto & item: *it)
			{
				passthrough_object o;
				if (item.contains("type"))
					o.type = item["type"];
				bool is_boundary = o.type == "3d-boundary" or o.type == "3d-boundary-reversed";
				bool is_depth = o.type == "3d-depth-passthrough";
				if (o.type != "3d-passthrough" and o.type != "3d-passthrough-reversed" and not is_boundary and not is_depth)
				{
					U_LOG_W("passthrough object \"%s\": unknown type \"%s\", skipping (not implemented yet)",
					        item.value("id", o.type).c_str(), o.type.c_str());
					continue;
				}
				if (item.contains("id"))
					o.id = item["id"];
				if (o.id.empty())
				{
					U_LOG_W("passthrough object without id, skipping");
					continue;
				}
				if (item.contains("tag"))
					o.tag = item["tag"];
				if (item.contains("fiducial"))
				{
					const auto & f = item["fiducial"];
					if (f.is_string())
						o.fiducial.push_back(f);
					else
						for (const auto & i: f)
							o.fiducial.push_back(i);
				}
				// Drop dangling references (typos shouldn't nuke the map).
				o.fiducial.erase(
				        std::remove_if(o.fiducial.begin(), o.fiducial.end(), [&](const std::string & id) {
					        bool known = std::ranges::any_of(fiducials, [&](const fiducial_entry & e) { return e.id == id; });
					        if (not known)
						        U_LOG_W("passthrough object \"%s\": unknown fiducial \"%s\", dropping reference", o.id.c_str(), id.c_str());
					        return not known;
				        }),
				        o.fiducial.end());
				if (o.fiducial.empty())
				{
					U_LOG_W("passthrough object \"%s\": no valid fiducials, skipping", o.id.c_str());
					continue;
				}
				if (item.contains("model-path"))
					o.model_path = item["model-path"];
				if (o.model_path.empty() and is_boundary)
				{
					U_LOG_W("passthrough object \"%s\": %s needs model-path (a zone without geometry), skipping", o.id.c_str(), o.type.c_str());
					continue;
				}
				if (item.contains("position"))
					o.position = item["position"];
				if (item.contains("orientation"))
					o.orientation = parse_orientation(item["orientation"], "passthrough object");
				if (item.contains("scale"))
				{
					if (item["scale"].is_number())
						o.scale = item["scale"];
					else if (item["scale"].is_array() and item["scale"].size() > 0)
						o.scale = item["scale"].at(0);
				}
				if (item.contains("feather-px"))
				{
					if (o.type == "3d-passthrough-reversed" or is_boundary)
						U_LOG_W("passthrough object \"%s\": feather-px is not accepted for %s, ignoring", o.id.c_str(), o.type.c_str());
					else
						o.feather_px = item["feather-px"];
				}
				if (item.contains("surface-projected"))
				{
					if (o.type == "3d-passthrough-reversed" or is_boundary or is_depth)
						U_LOG_W("passthrough object \"%s\": surface-projected is not applicable to %s, ignoring", o.id.c_str(), o.type.c_str());
					else
						o.surface_projected = item["surface-projected"];
				}
				if (item.contains("transition-distance"))
				{
					if (not is_boundary)
						U_LOG_W("passthrough object \"%s\": transition-distance is only accepted for 3d-boundary(-reversed), ignoring", o.id.c_str());
					else
					{
						o.transition_distance_m = item["transition-distance"];
						if (o.transition_distance_m < 0)
						{
							U_LOG_W("passthrough object \"%s\": negative transition-distance, clamping to 0", o.id.c_str());
							o.transition_distance_m = 0;
						}
					}
				}
				if (std::ranges::any_of(passthrough_objects, [&](const passthrough_object & p) { return p.id == o.id; }))
				{
					U_LOG_W("duplicate passthrough object id \"%s\", skipping", o.id.c_str());
					continue;
				}
				passthrough_objects.push_back(std::move(o));
			}
		}

		if (auto it = json.find("debug-gui"); it != json.end())
			debug_gui = *it;

		if (auto it = json.find("use-steamvr-lh"); it != json.end())
			use_steamvr_lh = *it;

		if (auto it = json.find("lh-max-extrapolation"); it != json.end())
			lh_max_extrapolation = *it;

		if (auto it = json.find("lh-stick-deadzone"); it != json.end())
			lh_stick_deadzone = *it;

		if (auto it = json.find("bit-depth"); it != json.end())
			bit_depth = *it;

		if (auto it = json.find("suppress-meta-boundary"); it != json.end())
			suppress_meta_boundary = *it;

		if (auto it = json.find("passthrough-until-lock"); it != json.end())
		{
			if (not it->is_string())
				U_LOG_W("passthrough-until-lock must be a fiducial id string, ignoring");
			else
				passthrough_until_lock = it->get<std::string>();
		}
		if (not passthrough_until_lock.empty() and
		    not std::ranges::any_of(fiducials, [&](const fiducial_entry & e) { return e.id == passthrough_until_lock; }))
		{
			U_LOG_W("passthrough-until-lock \"%s\" matches no fiducial id, ignoring", passthrough_until_lock.c_str());
			passthrough_until_lock.clear();
		}

		if (auto it = json.find("passthrough-fade-time"); it != json.end())
		{
			if (not it->is_number())
				U_LOG_W("passthrough-fade-time must be a number of seconds, ignoring");
			else
			{
				passthrough_fade_time_s = it->get<float>();
				if (passthrough_fade_time_s < 0)
				{
					U_LOG_W("passthrough-fade-time negative, clamping to 0");
					passthrough_fade_time_s = 0;
				}
			}
		}

		if (auto it = json.find("tcp-only"); it != json.end())
			tcp_only = *it;

		if (auto it = json.find("reprojection"); it != json.end())
		{
			if (not it->is_object())
				U_LOG_W("reprojection must be an object, ignoring");
			else
			{
				if (auto jt = it->find("enabled"); jt != it->end())
				{
					if (jt->is_boolean())
						reprojection.enabled = jt->get<bool>();
					else
						U_LOG_W("reprojection.enabled must be a boolean, ignoring");
				}
				if (auto jt = it->find("source-reverse-z"); jt != it->end())
				{
					if (jt->is_boolean())
						reprojection.source_reverse_z = jt->get<bool>();
					else
						U_LOG_W("reprojection.source-reverse-z must be a boolean, ignoring");
				}
				if (auto jt = it->find("gpu-priority"); jt != it->end())
				{
					if (not jt->is_string())
						U_LOG_W("reprojection.gpu-priority must be a string, ignoring");
					else
					{
						const auto value = jt->get<std::string>();
						if (value == "medium")
							reprojection.gpu_priority = configuration::reprojection::gpu_priority::medium;
						else if (value == "high")
							reprojection.gpu_priority = configuration::reprojection::gpu_priority::high;
						else if (value == "realtime")
							reprojection.gpu_priority = configuration::reprojection::gpu_priority::realtime;
						else
							U_LOG_W("reprojection.gpu-priority must be \"medium\", \"high\" or \"realtime\", ignoring");
					}
				}
			}
		}

		if (auto it = json.find("port"); it != json.end())
			port = *it;

		if (auto it = json.find("hostname"); it != json.end())
			hostname = *it;

		if (auto it = json.find("publish-service"); it != json.end())
		{
			publication = *it;
			if (publication == service_publication(-1))
				throw std::runtime_error("invalid service publication " + it->get<std::string>());
		}

		if (auto it = json.find("openvr-compat-path"); it != json.end())
		{
			if (it->is_null())
				openvr_compat_path = nullptr;
			else
				openvr_compat_path = *it;
		}
	}
	catch (const std::exception & e)
	{
		U_LOG_E("Configuration file error: %s", e.what());
	}
}

std::string server_cookie()
{
	{
		std::ifstream cookie(cookie_file);
		char buffer[33];
		cookie.read(buffer, sizeof(buffer) - 1);

		if (cookie)
			return {buffer, sizeof(buffer) - 1};
	}

	{
		std::random_device r;
		std::default_random_engine engine(r());

		std::uniform_int_distribution<int> dist(0, 61);

		char buffer[33];
		for (int i = 0; i < 32; i++)
		{
			int c = dist(engine);

			if (c < 10)
				buffer[i] = '0' + c;
			else if (c < 36)
				buffer[i] = 'A' + c - 10;
			else
				buffer[i] = 'a' + c - 36;
		}

		buffer[sizeof(buffer) - 1] = 0;

		std::filesystem::create_directories(cookie_file.parent_path());
		std::ofstream cookie(cookie_file);
		cookie.write(buffer, sizeof(buffer) - 1);

		return buffer;
	}
}

std::string to_iso8601(std::chrono::system_clock::time_point timestamp)
{
	// Truncate to an integer number of seconds to be parsable by strptime
	auto t = std::chrono::time_point_cast<std::chrono::seconds>(timestamp);

#if __cpp_lib_chrono >= 201907L
	return std::format("{:%FT%H:%M:%S%z}", std::chrono::zoned_time{std::chrono::current_zone(), t});
#else
	return std::format("{:%FT%H:%M:%S%z}", t);
#endif
}

template <typename T>
std::string to_iso8601(std::chrono::zoned_time<T> timestamp)
{
	return std::format("{:%FT%H:%M:%S%z}", timestamp);
}

std::optional<std::chrono::system_clock::time_point> from_iso8601(const std::string & timestamp)
{
	tm t;
	if (strptime(timestamp.c_str(), "%FT%H:%M:%S%z", &t) == nullptr)
		return std::nullopt;

	return std::chrono::system_clock::from_time_t(mktime(&t));
}

std::vector<headset_key> known_keys()
{
	if (not std::filesystem::exists(known_keys_file))
		return {};

	try
	{
		std::ifstream file(known_keys_file);
		std::vector<headset_key> keys;
		auto json = nlohmann::json::parse(file);

		for (const auto & key: json)
		{
			headset_key k{
			        .public_key = key["key"],
			        .name = key["name"],
			};

			if (key.contains("last_connection"))
				k.last_connection = from_iso8601(key["last_connection"]);

			keys.push_back(k);
		}

		return keys;
	}
	catch (const std::exception & e)
	{
		U_LOG_E("Invalid key file: %s", e.what());
		return {};
	}
}

static void save_keys(const std::vector<headset_key> & keys)
{
	nlohmann::json json;

	for (const auto & key: keys)
	{
		nlohmann::json json_key;
		json_key["key"] = key.public_key;
		json_key["name"] = key.name;
		if (key.last_connection)
			json_key["last_connection"] = to_iso8601(*key.last_connection);

		json.push_back(json_key);
	}

	std::string json_str = json.dump();

	std::filesystem::path known_keys_file_new = known_keys_file;
	known_keys_file_new += ".new";

	std::ofstream file(known_keys_file_new);
	file.write(json_str.data(), json_str.size());

	std::error_code ec;
	std::filesystem::rename(known_keys_file_new, known_keys_file, ec);

	if (ec)
		U_LOG_E("Failed to save keys: %s", ec.message().c_str());
}

void add_known_key(headset_key key)
{
	std::vector<headset_key> keys = known_keys();

	key.last_connection = std::chrono::system_clock::now();
	if (key.name == "")
		key.name = "Unknown headset";

	int n = 1;
	std::string original_name = key.name;
	while (std::ranges::any_of(keys, [&](const headset_key & k) { return k.name == key.name; }))
	{
		key.name = original_name + " (" + std::to_string(++n) + ")";
	}

	keys.push_back(key);

	save_keys(keys);
}

void remove_known_key(const std::string & key)
{
	std::vector<headset_key> keys = known_keys();
	std::erase_if(keys, [&](const headset_key & k) { return k.public_key == key; });

	save_keys(keys);
}

void rename_known_key(headset_key key)
{
	std::vector<headset_key> keys = known_keys();

	auto key_iter = std::ranges::find(keys, key.public_key, &headset_key::public_key);
	if (key_iter != keys.end())
		key_iter->name = key.name;

	save_keys(keys);
}

void update_last_connection_timestamp(const std::string & key)
{
	std::vector<headset_key> keys = known_keys();

	auto key_iter = std::ranges::find(keys, key, &headset_key::public_key);
	if (key_iter != keys.end())
	{
		key_iter->last_connection = std::chrono::system_clock::now();
		save_keys(keys);
	}
}

} // namespace wivrn
