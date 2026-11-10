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
// coverage instead of popping (no shimmer at distance). Occlusion is a
// single center-depth compare against the prepass (whole circles live or
// die — never half-chopped by geometry edges).
layout(location = 0) in vec2 vUv;
layout(location = 1) in float vW;
layout(location = 2) in vec2 vCenterUV;
layout(location = 3) in float vCenterDepth;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D depthTex;
// Proximity sphere field (furthest exit, same mapping): whole-circle
// near-field gate alongside the occlusion test below.
layout(set = 0, binding = 1) uniform sampler2D sphereTex;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
	vec4 view_params; // x: view W px, y: view H px, z: proj[1][1], w: dot radius (cells)
}
pc;

void main()
{
	// Reversed-Z (nearer = larger): visible iff the center sits ahead of
	// the prepass surface. Own face passes strictly via the 2mm nudge;
	// eps guards D16 quantization only. Range note: the 1/w depth mapping
	// makes this tolerance worth ~2cm at 2m but much looser far out —
	// distant dots barely occlude, inherent to the mapping, not the eps.
	if (vCenterDepth + 1e-4 < texture(depthTex, vCenterUV).r)
		discard;
	// Proximity gate (same keep-nearer form): the dot center must sit
	// ahead of the furthest sphere exit, i.e. inside ANY tracked sphere.
	// Outside-sphere eyes are an accepted over-reveal (range test later).
	if (vCenterDepth + 1e-4 < texture(sphereTex, vCenterUV).r)
		discard;
	float d = length(vUv);
	float pitch = max(pc.color_pitch.w, 1e-4);
	float px_per_m = (pc.view_params.y * 0.5) * abs(pc.view_params.z) / max(vW, 1e-3);
	float r_px = pc.view_params.w * pitch * px_per_m;
	float e = 1.0 / max(r_px, 1e-3);
	float dot_ = (1.0 - smoothstep(1.0 - e, 1.0, d)) * clamp(r_px, 0.0, 1.0);
	if (dot_ <= 0.001)
		discard;
	out_color = vec4(pc.color_pitch.rgb, dot_);
}
