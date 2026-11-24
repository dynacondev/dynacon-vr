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

// 6DOF sensor warp fragment: false-colors the warped depth exactly like
// depth_viz (green near -> red far), NEAREST throughout so the viz shows
// true outcomes. Includes the anti-streak discard: forward-warped
// triangles spanning a depth discontinuity interpolate fiction (the
// "rubber sheet" between two surfaces that beats the true background in
// the depth test because it is nearer). Re-tapping the sensor at the
// interpolated UV disagrees with the interpolated depth exactly on those
// pixels -> discard -> the hole stays cleared = far. No hole-filling.

layout(location = 0) in vec2 v_suv;
layout(location = 1) in float v_depth;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 2) uniform FrameParams
{
	float near_z;
	float far_z;
	float opacity;
	float pad;
}
u;

layout(push_constant) uniform pc
{
	int eye;
}
push;

float linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

void main()
{
	float n = u.near_z;
	float f = u.far_z;

	// Anti-streak: the interpolated depth must agree with the sensor
	// texel actually under the interpolated UV. v_depth == 0 marks
	// no-data vertices (collapsed in the vertex stage); those fragments
	// only exist on degenerate tris and are discarded here too.
	float d = texture(sensor_tex, vec3(v_suv, float(push.eye))).r;
	float s = linearize_depth(d, n, f);
	if (v_depth <= 0.0 || d >= 0.9999 || abs(s - v_depth) > max(0.02, 0.05 * v_depth))
		discard;

	bool infinite = !(f > n) || f > 1e9;
	float denom = infinite ? max(s, 1e-6) : (f - n);
	float t = clamp((s - n) / denom, 0.0, 1.0);
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);

	out_color = vec4(col, u.opacity);
}
