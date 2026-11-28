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

// Splat pass-through fragment: stores the flat quad-max planar display
// depth (meters, from the TL provoking vertex) into the transient R16F
// image. Quads with no valid in-front corner arrive as <= 0 and discard
// to honest FAR holes (no-data/behind-camera). No sampling, no UBO —
// the depth test already resolved visibility. Blending off (replace).

layout(location = 0) flat in float v_bg;

layout(location = 0) out float out_depth;

void main()
{
	if (v_bg <= 0.0)
		discard;
	out_depth = v_bg;
}
