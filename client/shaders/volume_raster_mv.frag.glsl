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

// Multiview volume raster fragment: furthest-point encode of the
// back-face eye distance as sensor-compatible D in [0,1] (same
// near/far, same infinite-far branch as the runtime D16). Push block
// widened to the MVP pair so the VERT/FRAG blocks agree byte-for-byte
// (see mask_mv.frag precedent — a mismatched FRAG block would read
// mvp[1] as near_z). Fragment reads only near_z/far_z.
layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	float near_z;
	float far_z;
}
pc;

layout(location = 0) in float eye_depth;

layout(location = 0) out vec4 out_color;

void main()
{
	float d = eye_depth;
	if (d <= 1e-6)
		discard;
	float n = pc.near_z;
	float f = pc.far_z;
	float D;
	if (!(f > n) || f > 1e9)
	{
		// Infinite far plane (what the Quest runtime reports):
		// D = 0 at near, -> 1 at far.
		D = 1.0 - n / d;
	}
	else
	{
		D = f * (d - n) / (d * (f - n));
	}
	out_color = vec4(clamp(D, 0.0, 1.0));
}
