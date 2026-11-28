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

// Common-layer geometry fragment: R16F linear-meters output for MIN
// blending over the large-finite FAR clear (65504.0).
//
// clip.w IS the planar eye distance in meters (d): the server color-eye
// projection is a perspective projection whose third row maps -z_view
// to clip.w. Rastering clip.w writes d directly, so the verdict
// compares layer meters against virt meters with no frustum
// parameters on either side.
//
// CULL_NONE (like the silhouette raster): windows reveal from both
// sides, and MIN blending keeps near-wins order-independent for
// overlapping windows. No depth attachment: MIN over the FAR clear
// resolves near-wins without one.
layout(location = 0) in float eye_depth;

layout(location = 0) out vec4 out_color;

void main()
{
	float d = eye_depth;
	if (d <= 1e-6)
		discard;
	out_color = vec4(d);
}
