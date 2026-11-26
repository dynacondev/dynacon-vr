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

// Splat colorize: false-colors the transient splatted planar depth
// (R16F, meters, display space) with the sensor ramp. No-data reads as
// FAR red through the ramp itself (clear value 65504 saturates t to 1),
// so no explicit no-data branch is needed. Standard 4-byte eye push,
// like all viz fragment shaders.

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray splat_tex;

layout(set = 0, binding = 1) uniform params
{
	vec2 near_far;
	float opacity;
	float pad;
} u;

layout(push_constant) uniform pc
{
	int eye;
} push;

void main()
{
	float d = texture(splat_tex, vec3(v_uv, float(push.eye))).r;
	float n = u.near_far.x;
	float f = u.near_far.y;
	// Infinite far (unbounded or classic): normalize against the
	// reconstructed distance itself (t = 1 - near/d).
	float denom;
	if (!(f > n) || f > 1e9)
		denom = max(d, 1e-6);
	else
		denom = f - n;
	float t = clamp((d - n) / denom, 0.0, 1.0);
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);

	out_color = vec4(col, u.opacity);
}
