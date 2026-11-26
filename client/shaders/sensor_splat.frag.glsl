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

#version 450

// Splat pass-through fragment: stores the splatted planar display depth
// (meters) into the transient R16F image. No sampling, no UBO — the
// depth test already resolved visibility. Blending off (replace).
// Verdict from the TL provoking vertex: 1 = discard (no-data or
// behind-camera: honest FAR hole), 2 = flat background stretch
// (disocclusion covered by the quad-max depth, never foreground),
// else the smooth interpolated depth.

layout(location = 0) in float v_depth;
layout(location = 1) flat in int v_flag;
layout(location = 2) flat in float v_bgmax;

layout(location = 0) out float out_depth;

void main()
{
	if (v_flag == 1)
		discard;
	if (v_flag == 2)
	{
		out_depth = v_bgmax;
		return;
	}
	out_depth = v_depth;
}
