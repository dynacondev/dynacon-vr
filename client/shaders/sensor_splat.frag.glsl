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
// Per-triangle verdict from the TL provoking vertex (triangles are
// emitted consecutively per quad, so parity selects: tri 0 = (a,b,d) /
// v_bg0, tri 1 = (a,c,d) / v_bg1): 0 = smooth interpolated depth,
// 1 = discard (no-data/behind-camera: honest FAR hole), 2 = flat
// triple-max background (disocclusion cover, never foreground).

layout(location = 0) in float v_depth;
layout(location = 1) flat in int v_mask;
layout(location = 2) flat in float v_bg0;
layout(location = 3) flat in float v_bg1;

layout(location = 0) out float out_depth;

void main()
{
	int tri = gl_PrimitiveID & 1;
	int verdict = (v_mask >> (tri * 2)) & 3;
	if (verdict == 1)
		discard;
	if (verdict == 2)
	{
		float bg = v_bg0;
		if (tri == 1)
			bg = v_bg1;
		out_depth = bg;
		return;
	}
	out_depth = v_depth;
}
