/*
 * WiVRn VR streaming
 * Copyright (C) 2026  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2026  Patrick Nicolas <patricknicolas@laposte.net>
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

#include <openxr/openxr.h>

#include <cstdint>
#include <vector>

namespace xr
{
// Display names for XrColorSpaceFB values (XR_FB_color_space)
inline const char * color_space_name(XrColorSpaceFB color_space)
{
	switch (color_space)
	{
		case XR_COLOR_SPACE_UNMANAGED_FB:
			return "Unmanaged";
		case XR_COLOR_SPACE_REC2020_FB:
			return "Rec 2020";
		case XR_COLOR_SPACE_REC709_FB:
			return "Rec 709";
		case XR_COLOR_SPACE_RIFT_CV1_FB:
			return "Rift CV1";
		case XR_COLOR_SPACE_RIFT_S_FB:
			return "Rift S";
		case XR_COLOR_SPACE_QUEST_FB:
			return "Quest";
		case XR_COLOR_SPACE_P3_FB:
			return "P3";
		case XR_COLOR_SPACE_ADOBE_RGB_FB:
			return "Adobe RGB";
		case XR_COLOR_SPACE_MAX_ENUM_FB:
			break;
	}
	return "Unknown";
}

// Resolve a persisted color space value against the runtime-enumerated list:
// returns the persisted value when still supported, otherwise P3 (Meta's
// recommended mastering space) when reported, otherwise the first enumerated
// mode. The extension has no getter and the runtime default is not
// queryable, so fresh installs cannot preserve the untouched runtime default.
inline XrColorSpaceFB resolve_color_space(int64_t configured, const std::vector<XrColorSpaceFB> & supported)
{
	for (auto mode: supported)
	{
		if ((int64_t)mode == configured)
			return mode;
	}
	for (auto mode: supported)
	{
		if (mode == XR_COLOR_SPACE_P3_FB)
			return mode;
	}
	if (not supported.empty())
		return supported.front();
	return XR_COLOR_SPACE_UNMANAGED_FB;
}
} // namespace xr
