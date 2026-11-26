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
// Killed triangles (flat flag from the TL provoking vertex) discard.

layout(location = 0) in float v_depth;
layout(location = 1) flat in int v_kill;

layout(location = 0) out float out_depth;

void main()
{
	if (v_kill != 0)
		discard;
	out_depth = v_depth;
}
