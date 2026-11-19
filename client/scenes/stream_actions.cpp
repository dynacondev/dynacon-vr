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

#include "application.h"
#include "stream.h"
#include <algorithm>
#include <spdlog/spdlog.h>

void scenes::stream::read_actions()
{
	from_headset::inputs inputs;

	// Extended hand mode: pinch and grasp analogs are never forwarded.
	// Instead their max feeds a single snapped binary output on the pinch
	// value channel (driving select downstream exactly as pinch does, but
	// thresholded). Debounce is left to the runtime.
	const auto & config = application::get_config();
	const bool extended = config.hand_mode == hand_controller_mode::extended;
	const float threshold = std::clamp(config.pinch_emit_threshold, 0.0f, 1.0f);

	struct pinch_accum
	{
		float best = 0;
		XrTime timestamp = 0;
		bool seen = false;
	};
	pinch_accum accum_left, accum_right;

	if (not is_gui_interactable())
	{
		for (const auto & [id, action, action_type]: input_actions)
		{
			switch (action_type)
			{
				case XR_ACTION_TYPE_BOOLEAN_INPUT: {
					auto value = application::read_action_bool(action);
					if (value and not (extended and (id == device_id::LEFT_PINCH_READY or id == device_id::RIGHT_PINCH_READY or id == device_id::LEFT_GRASP_READY or id == device_id::RIGHT_GRASP_READY)))
						inputs.values.push_back({id, (float)value->second, value->first});
				}
				break;

				case XR_ACTION_TYPE_FLOAT_INPUT: {
					auto value = application::read_action_float(action);
					if (not value)
						break;
					pinch_accum * accum = nullptr;
					if (id == device_id::LEFT_PINCH_VALUE or id == device_id::LEFT_GRASP_VALUE)
						accum = &accum_left;
					else if (id == device_id::RIGHT_PINCH_VALUE or id == device_id::RIGHT_GRASP_VALUE)
						accum = &accum_right;
					if (extended and accum)
					{
						// Order is not guaranteed (unordered source map):
						// accumulate the max, emit once after the loop.
						if (not accum->seen or value->second > accum->best)
						{
							accum->best = value->second;
							accum->timestamp = value->first;
						}
						accum->seen = true;
						break;
					}
					inputs.values.push_back({id, value->second, value->first});
				}
				break;

				case XR_ACTION_TYPE_VECTOR2F_INPUT: {
					auto value = application::read_action_vec2(action);
					if (value)
					{
						inputs.values.push_back({id, value->second.x, value->first});
						inputs.values.push_back({(device_id)((int)id + 1), value->second.y, value->first});
					}
				}
				break;

				case XR_ACTION_TYPE_POSE_INPUT:
				default:
					break;
			}
		}

		if (extended)
		{
			auto emit = [&](const pinch_accum & accum, device_id pinch_id) {
				if (not accum.seen)
					return;
				inputs.values.push_back({pinch_id, accum.best >= threshold ? 1.0f : 0.0f, accum.timestamp});
			};
			emit(accum_left, device_id::LEFT_PINCH_VALUE);
			emit(accum_right, device_id::RIGHT_PINCH_VALUE);
		}
	}

	try
	{
		network_session->send_stream(std::move(inputs));
	}
	catch (std::exception & e)
	{
		spdlog::warn("Exception while sending inputs packet: {}", e.what());
	}
}

void scenes::stream::operator()(to_headset::haptics && haptics)
{
	auto range = haptics_actions.equal_range(haptics.id);
	for (auto it = range.first; it != range.second; ++it)
	{
		XrAction action = it->second.action;
		float old = it->second.amplitude;
		it->second.amplitude = haptics.amplitude;
		// Some runtimes may be slow to process actions
		// Skip it if not necessary
		if (old == 0 and haptics.amplitude == 0)
			continue;

		if (haptics.amplitude > 0)
			application::haptic_start(action, XR_NULL_PATH, haptics.duration.count(), haptics.frequency, std::min(1.0f, haptics.amplitude));
		else
			application::haptic_stop(action, XR_NULL_PATH);
	}
}
