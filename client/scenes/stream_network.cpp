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

#include "stream.h"

#include "application.h"
#include "utils/i18n.h"
#include "utils/named_thread.h"

#include <algorithm>
#include <fstream>
#include <spdlog/spdlog.h>
#include <uni_algo/case.h>

void scenes::stream::process_packets()
{
#ifdef __ANDROID__
	application::instance().setup_jni();
#endif
	while (state_ != state::shutdown)
	{
		try
		{
			network_session->poll(*this, std::chrono::milliseconds(500));
		}
		catch (std::exception & e)
		{
			spdlog::info("Exception in network thread, exiting: {}", e.what());
			exit();
		}
	}
}

void scenes::stream::operator()(to_headset::server_message && message)
{
	switch (message.kind)
	{
		case to_headset::server_message::kind::toast:
		case to_headset::server_message::kind::toast_urgent: {
			auto toast = gui_toast.lock();
			toast->emplace(message.msg, message.kind == to_headset::server_message::kind::toast_urgent);

			gui_status_last_change = instance.now();
			break;
		}

		case to_headset::server_message::kind::error: {
			auto queue = stream_error_queue.lock();
			queue->emplace(std::move(message.msg));

			break;
		}
	}
}

void scenes::stream::operator()(to_headset::video_stream_data_shard && shard)
{
	std::shared_lock lock(decoder_mutex);
	uint8_t idx = shard.stream_item_idx;
	// Per-stream payload accounting (network thread, atomics): smoothed
	// into per-stream bitrates in accumulate_metrics (render thread).
	if (idx < stream_rx_bytes.size())
		stream_rx_bytes[idx].fetch_add(shard.payload.size(), std::memory_order_relaxed);
	if (idx >= decoders.size())
	{
		// We don't know (yet?) about this stream, ignore packet
		return;
	}
	decoders[idx].decoder->push_shard(std::move(shard));
}

void scenes::stream::operator()(to_headset::feature_control && control)
{
	switch (control.f)
	{
		case wivrn::to_headset::feature_control::hid_input:
			hid_forwarding = control.state;
			if (not control.state and (application::get_config().forward_keyboard or application::get_config().forward_mouse or application::get_config().forward_gamepad))
			{
				auto toast = gui_toast.lock();
				toast->emplace(_("The server does not allow forwarded input devices"), true);
				gui_status_last_change = instance.now();
			}
			return;
		case wivrn::to_headset::feature_control::microphone:
			if (audio_handle)
				audio_handle->set_mic_state(control.state);
			return;
	}
}

void scenes::stream::operator()(to_headset::audio_stream_description && desc)
{
	audio_handle.emplace(desc, *network_session, instance);
}

void scenes::stream::operator()(to_headset::video_stream_description && desc)
{
	setup(desc);

	if (not tracking_thread)
	{
		tracking_thread = utils::named_thread("tracking_thread", &stream::tracking, this);
	}
}

void scenes::stream::operator()(to_headset::refresh_rate_change && rate)
{
	spdlog::info("refresh rate change request: {}", rate.hz);
	session.set_refresh_rate(rate.hz);
	std::shared_lock lock(decoder_mutex);
	if (video_stream_description)
		video_stream_description->refresh_rate = rate.hz;
}

void scenes::stream::operator()(to_headset::stream_tab_change && tab)
{
	next_gui_status = tab.tab;
}

void scenes::stream::operator()(to_headset::timesync_query && query)
{
	network_session->send_stream(from_headset::timesync_response{
	        .query = query.query,
	        .response = instance.now(),
	});
}

void scenes::stream::operator()(audio_data && data)
{
	if (audio_handle)
		(*audio_handle)(std::move(data));
}

void scenes::stream::send_feedback(const wivrn::from_headset::feedback & feedback)
{
	try
	{
		network_session->send_control(wivrn::from_headset::feedback{feedback});
	}
	catch (std::exception & e)
	{
		spdlog::warn("Exception while sending feedback packet: {}", e.what());
	}
}

void scenes::stream::operator()(to_headset::application_list && l)
{
	apps(std::move(l));
}

void scenes::stream::operator()(to_headset::application_icon && icon)
{
	apps(std::move(icon));
}

void scenes::stream::operator()(to_headset::running_applications && apps)
{
	*running_applications.lock() = std::move(apps);
	stream_apps_known = true;
}

void scenes::stream::operator()(to_headset::fiducial_map && map)
{
	spdlog::info("Received fiducial map with {} fiducials, {} passthrough objects", map.fiducials.size(), map.objects.size());
	for (const auto & fid: map.fiducials)
	{
		std::string label = fid.tag.empty() ? fid.id : fid.tag;
		spdlog::info("Fiducial \"{}\": {} marker(s) ({})", label, fid.markers.size(), fid.is_static ? "static" : "moving");
	}
	for (const auto & obj: map.objects)
	{
		std::string label = obj.tag.empty() ? obj.id : obj.tag;
		if (obj.type == "3d-boundary" or obj.type == "3d-boundary-reversed")
		{
			if (obj.model_hash.empty())
				spdlog::info("Passthrough object \"{}\" ({}): no model", label, obj.type);
			else
				spdlog::info("Passthrough object \"{}\" ({}): model {} ({} bytes) transition {:.2f}m",
				             label, obj.type, obj.model_hash.substr(0, 8), obj.model_size,
				             (double)obj.transition_distance_m);
		}
		else if (obj.model_hash.empty())
			spdlog::info("Passthrough object \"{}\" ({}): no model", label, obj.type);
		else
			spdlog::info("Passthrough object \"{}\" ({}): model {} ({} bytes)",
			             label, obj.type, obj.model_hash.substr(0, 8), obj.model_size);
	}
	if (not map.passthrough_until_lock.empty())
		spdlog::info("passthrough-until-lock: full passthrough until \"{}\" locks (fade {:.1f}s)",
		             map.passthrough_until_lock, (double)std::max(0.f, map.passthrough_fade_time_s));
	*fiducial_map.lock() = std::move(map);

	// Server-pushed Meta boundary default (top-level config, no
	// in-headset toggle): applied live on every map receipt. Startup
	// default comes from the untouched client setting until the first map
	// arrives; the map is always sent, even empty.
	{
		bool suppress;
		{
			auto locked = fiducial_map.lock();
			suppress = locked->suppress_meta_boundary;
		}
		static bool last_suppress = false;
		static bool have_last = false;
		if (not have_last or suppress != last_suppress)
		{
			have_last = true;
			last_suppress = suppress;
			session.set_passthrough_boundary_enabled(not suppress);
			spdlog::info("Meta boundary in passthrough: {}", suppress ? "suppressed (server)" : "shown (server)");
		}
	}

	// Request models missing from the local cache
	std::error_code ec;
	auto cache_dir = application::get_config_path() / "fiducial_models";
	static bool cache_dir_logged = false;
	if (not cache_dir_logged)
	{
		cache_dir_logged = true;
		spdlog::info("Fiducial model cache dir: {}", cache_dir.string());
	}
	std::filesystem::create_directories(cache_dir, ec);
	if (ec)
	{
		spdlog::warn("Cannot create fiducial model cache dir: {}", ec.message());
		return;
	}

	std::vector<std::pair<std::string, uint64_t>> missing;
	size_t cached = 0;
	{
		auto map = fiducial_map.lock();
		for (const auto & obj: map->objects)
		{
			if (obj.model_hash.empty() or obj.model_size == 0)
				continue;
			if (std::filesystem::exists(fiducial_model_path(obj.model_hash), ec))
			{
				++cached;
				continue;
			}
			if (fiducial_downloads.contains(obj.model_hash))
				continue;
			missing.emplace_back(obj.model_hash, obj.model_size);
		}
	}

	if (cached)
		spdlog::info("Fiducial models already cached: {}", cached);

	for (const auto & [hash, size]: missing)
	{
		fiducial_downloads[hash] = fiducial_download{};
		network_session->send_control(from_headset::fiducial_model_request{
		        .model_hash = hash,
		});
		spdlog::info("Requesting fiducial model {} ({} bytes)", hash, size);
	}
}

std::filesystem::path scenes::stream::fiducial_model_path(const std::string & hash)
{
	return application::get_config_path() / "fiducial_models" / (hash + ".glb");
}

void scenes::stream::operator()(to_headset::fiducial_model_chunk && chunk)
{
	// Sanity caps: chunk counts larger than a 64MB model in 1-byte chunks
	if (chunk.chunk_count == 0 or chunk.chunk_count > 65536 or chunk.data.empty())
	{
		spdlog::warn("Ignoring invalid fiducial model chunk for {}", chunk.model_hash);
		return;
	}
	if (chunk.chunk_index >= chunk.chunk_count)
	{
		spdlog::warn("Ignoring out-of-range fiducial model chunk for {}", chunk.model_hash);
		return;
	}

	uint64_t expected_size = 0;
	{
		auto map = fiducial_map.lock();
		auto obj = std::ranges::find(map->objects, chunk.model_hash, &to_headset::passthrough_object::model_hash);
		if (obj == map->objects.end() or obj->model_size == 0)
		{
			spdlog::warn("Ignoring fiducial model chunk for unknown hash {}", chunk.model_hash);
			return;
		}
		expected_size = obj->model_size;
	}

	auto & download = fiducial_downloads[chunk.model_hash];
	if (download.chunk_count == 0)
		download.chunk_count = chunk.chunk_count;
	else if (download.chunk_count != chunk.chunk_count)
	{
		spdlog::warn("Ignoring fiducial model chunk with mismatched count for {}", chunk.model_hash);
		return;
	}

	if (download.chunks.contains(chunk.chunk_index))
		return;
	download.chunks[chunk.chunk_index] = std::move(chunk.data);

	if (download.chunks.size() < download.chunk_count)
		return;

	// All chunks received: assemble in order
	std::vector<std::byte> data;
	data.reserve(expected_size);
	for (uint32_t i = 0; i < download.chunk_count; ++i)
	{
		auto it = download.chunks.find(i);
		if (it == download.chunks.end())
			return;
		data.insert(data.end(), it->second.begin(), it->second.end());
	}

	if (data.size() != expected_size)
	{
		spdlog::warn("Fiducial model {} size mismatch (got {}, expected {}), discarding",
		             chunk.model_hash,
		             data.size(),
		             expected_size);
		fiducial_downloads.erase(chunk.model_hash);
		return;
	}

	std::error_code ec;
	auto path = fiducial_model_path(chunk.model_hash);
	std::filesystem::create_directories(path.parent_path(), ec);
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	if (not file)
	{
		spdlog::warn("Cannot write fiducial model cache file {}", path.string());
		fiducial_downloads.erase(chunk.model_hash);
		return;
	}
	file.write(reinterpret_cast<const char *>(data.data()), data.size());
	file.close();
	spdlog::info("Cached fiducial model {} ({} bytes) at {}", chunk.model_hash, data.size(), path.string());
	fiducial_downloads.erase(chunk.model_hash);
}

void scenes::stream::start_application(std::string appid)
{
	network_session->send_control(wivrn::from_headset::start_app{
	        .app_id = std::move(appid),
	});
}
