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

#extension GL_EXT_multiview : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

// Sensor warp + upsample in one pass: the old-main rotation homography
// (E1/Suv/Q rows/Rew/Rcw, capture<->display relative rotation) maps the
// depth-stream (sync eye) UV to the runtime sensor UV. Bilinear tap of
// the 320px sensor smooths the low-res input (fake intermediates at
// discontinuities accepted; edge-aware upsampling is a later upgrade).
// Off-canvas warp writes far (1.0): fail closed, never reveals.
// Identity H reproduces the direct tap. Plain scalar math only (see
// AGENTS.md: no mat3/ternary on this driver path).

layout(push_constant) uniform Push
{
	vec4 h0;
	vec4 h1;
	vec4 h2;
	vec4 h3;
	vec4 h4;
	vec4 h5;
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;

void main()
{
	// Branchless eye select (mix, never a ternary/mat3).
	float vi = float(gl_ViewIndex);
	float m = step(0.5, vi);
	vec4 c0 = mix(pc.h0, pc.h3, m);
	vec4 c1 = mix(pc.h1, pc.h4, m);
	vec4 c2 = mix(pc.h2, pc.h5, m);

	float px = c0.x * uv.x + c1.x * uv.y + c2.x;
	float py = c0.y * uv.x + c1.y * uv.y + c2.y;
	float pz = c0.z * uv.x + c1.z * uv.y + c2.z;
	float ok = step(1e-6, pz);
	vec2 s = vec2(px, py) / max(pz, 1e-6);
	ok *= step(0.0, s.x) * step(s.x, 1.0) * step(0.0, s.y) * step(s.y, 1.0);

	float d = texture(sensor_tex, vec3(s, vi)).r;
	// mix(), not a ternary: off-canvas reads back far.
	out_color = vec4(mix(1.0, d, ok));
}
