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

// Trivial circle falloff inside the billboard quad (quad space: the dot
// edge is at 1.0 by construction). 1px AA band; subpixel dots fade by
// coverage instead of popping (no shimmer at distance).
layout(location = 0) in vec2 vUv;
layout(location = 1) in float vW;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
	vec4 view_params; // x: view W px, y: view H px, z: proj[1][1]
}
pc;

void main()
{
	float d = length(vUv);
	float pitch = max(pc.color_pitch.w, 1e-4);
	float px_per_m = (pc.view_params.y * 0.5) * abs(pc.view_params.z) / max(vW, 1e-3);
	float r_px = 0.15 * pitch * px_per_m;
	float e = 1.0 / max(r_px, 1e-3);
	float dot_ = (1.0 - smoothstep(1.0 - e, 1.0, d)) * clamp(r_px, 0.0, 1.0);
	if (dot_ <= 0.001)
		discard;
	out_color = vec4(pc.color_pitch.rgb, dot_);
}
