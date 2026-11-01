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

#pragma once

#include "hostname.h"
#include "wivrn_config.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>

#include "wivrn_packets.h"

namespace wivrn
{

enum class service_publication
{
	none,
	avahi,
};

struct configuration
{
	struct encoder
	{
		std::string name;
		std::optional<wivrn::video_codec> codec;
		std::map<std::string, std::string> options;
		std::optional<std::string> device;
	};

	std::array<encoder, 4> encoders; // left, right, alpha, depth
	std::optional<uint8_t> bit_depth;
	std::optional<std::array<float, 3>> grip_surface;
	std::vector<std::string> application;

	// Fiducial tracking markers (Quest only). Each fiducial resolves its
	// sighted markers to one 6DoF frame, identified by id and placed by
	// the per-instance smoothing filters.
	struct fiducial_marker
	{
		// Exact decoded payload identifying the marker (required).
		std::string marker_data;
		float marker_size_m = 0;
		// Marker-to-fiducial offset, authored tag-in-board: position is
		// the tag center in fiducial meters, orientation the tag axes
		// relative to fiducial axes. Position in meters, orientation as
		// xyzw quaternion (parsed from [rx, ry, rz] degrees).
		std::array<float, 3> position = {0, 0, 0};
		std::array<float, 4> orientation = {0, 0, 0, 1};
	};
	struct fiducial_entry
	{
		// Stable id linking config, protocol and headset UI (required,
		// must be unique).
		std::string id;
		// Display-only label for the headset status UI. Never used for
		// matching or linkage.
		std::string tag;
		bool is_static = true;
		// Resolver/smoothing tuning, carried through to the per-instance
		// filters.
		float sigma_k = 3;
		float euro_min_cutoff = 0.4f;
		float euro_beta = 0.07f;
		float hold_inner_mm = 1;
		float hold_inner_deg = 0.1f;
		std::vector<fiducial_marker> markers;
	};
	std::vector<fiducial_entry> fiducials;
	// Behavior objects placed from solved fiducial frames. Instances fan
	// out per sighted marker entity: the same QR seen twice places the
	// object twice; one object may reference several fiducials (one
	// instance per visible pairing, never fused).
	struct passthrough_object
	{
		// Behavior type. "3d-passthrough" renders the model as a
		// surface-projected passthrough cutout; "3d-passthrough-reversed"
		// reverse-cuts the model from every mask layer instead (game video
		// shows through, depth-tested against covering windows). "3d-boundary"
		// shows game video only inside the model volume (passthrough outside),
		// "3d-boundary-reversed" inverts it. Boundary meshes must be closed
		// volumes. "3d-depth-passthrough" reveals the flat feed where the
		// runtime depth map resolves inside the model volume (depth-tested
		// volume, feathered like a normal window). Unknown types are skipped
		// with a warning (forward-compat).
		std::string type = "3d-passthrough";
		std::string id;
		std::string tag;
		// Referenced fiducial ids (dangling refs skip the object).
		std::vector<std::string> fiducial;
		std::string model_path;
		// Fiducial-to-object offset: objectPose = solvedFiducialPose *
		// offset. Uniform scale applies last, objects only.
		std::array<float, 3> position = {0, 0, 0};
		std::array<float, 4> orientation = {0, 0, 0, 1};
		float scale = 1;
		// Passthrough window feather width in screen pixels (render-only,
		// applied live). Mask stacks group by this value so each object
		// feathers independently. Not accepted for
		// 3d-passthrough-reversed and 3d-boundary(-reversed)
		// (warned + ignored).
		float feather_px = 24;
		// Surface-projected mapping: the headset projects camera imagery
		// onto this object's mesh instead of the flat reconstruction feed.
		// Enabled by default; opt out per object with false (plus a live
		// in-headset override). Ignored for 3d-passthrough-reversed and
		// 3d-boundary(-reversed) (warned).
		bool surface_projected = true;
		// Boundary fade width in meters (3d-boundary(-reversed) only):
		// the scene cross-fades over +/- this distance around the mesh
		// surface (symmetric smoothstep, 0.5 at the surface). 0 = hard
		// step at the surface. Ignored for other types (warned).
		float transition_distance_m = 0.3f;
	};
	std::vector<passthrough_object> passthrough_objects;

	// Meta (OS) boundary visibility while our passthrough is shown: true
	// (default) requests suppression, false leaves the OS boundary
	// visible. Pushed to the headset (no in-headset toggle); a request
	// only, the runtime may refuse. The boundary always shows in opaque
	// VR regardless of this setting.
	bool suppress_meta_boundary = true;

	// Full flat passthrough until the named fiducial locks (first live
	// solve latches, never reset on loss). Empty (default) disables.
	// Only applies when it matches a fiducial id in the config;
	// dangling values warn + ignore.
	std::string passthrough_until_lock;

	// Smoothstep fade from full passthrough to VR once the until-lock
	// target latches, in seconds (default 2). 0 = instant release.
	// Applies to the until-lock transition only, nothing else.
	float passthrough_fade_time_s = 2.0f;

	struct reprojection
	{
		// Master switch for server-side depth reprojection: every
		// frame carrying depth is reprojected from its known
		// generation pose to the latest predicted pose.
		bool enabled = true;
		// Depth encoding of the submitting app: true = reverse-Z
		// (1 at near, X-Plane 12 Vulkan). No in-band signal carries
		// this, so the default encodes the pipeline assumption;
		// normal-Z apps set it false explicitly.
		bool source_reverse_z = true;
		// Device-global queue priority for our compositor, transfer
		// and encode queues: high prefers our submits when the app
		// saturates the GPU. This is the only lever that arbitrates
		// across our VkDevice and the app's device; medium requests
		// nothing (driver default).
		enum class gpu_priority
		{
			medium,
			high,
			// Privilege-gated (CAP_SYS_NICE): falls back to high,
			// then default, when denied.
			realtime,
		};
		gpu_priority gpu_priority = gpu_priority::high;
	};
	reprojection reprojection;

	bool debug_gui = false;
	bool use_steamvr_lh = false;
	std::optional<int64_t> lh_max_extrapolation;
	std::optional<float> lh_stick_deadzone;
	bool hid_forwarding = false;
	bool tcp_only = false;
	int port = wivrn::default_port;
	std::string hostname = wivrn::hostname();
	service_publication publication = service_publication::avahi;

	// monostate: default value, string: user defined, nullptr: disabled
	std::variant<std::monostate, std::string, std::nullptr_t> openvr_compat_path;

	static void set_config_file(const std::filesystem::path &);
	static std::filesystem::path get_config_file();

	static nlohmann::json read_configuration();

	// JSON array describing every searched config file location, in merge
	// order: [{"id", "path", "status", "location"}]. See ConfigFiles in
	// dbus/io.github.wivrn.Server.xml. Re-reads the files (cheap).
	static std::string config_files_json();
	configuration();
};

std::string server_cookie();

struct headset_key
{
	std::string public_key;
	std::string name;
	std::optional<std::chrono::system_clock::time_point> last_connection;
};

std::vector<headset_key> known_keys();
void add_known_key(headset_key key);
void remove_known_key(const std::string & key);
void rename_known_key(headset_key key);
void update_last_connection_timestamp(const std::string & key);

} // namespace wivrn
