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

#include "wivrn_packets.h"

#include <filesystem>
#include <optional>

namespace wivrn
{

// Path of the single last-headset spec cache, next to known_keys.json.
std::filesystem::path headset_cache_file();

// Persist the latest headset spec for early-start boots.
void save_headset_cache(const from_headset::headset_info_packet & info);

// Load the cached spec, or nullopt when missing/invalid.
// Strict: any unknown version or implausible value rejects the file and
// the server falls back to waiting for a headset.
std::optional<from_headset::headset_info_packet> load_headset_cache();

} // namespace wivrn
