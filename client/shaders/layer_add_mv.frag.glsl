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

// Layer add: linearize the warped-sensor sample for the MIN-blend LOAD
// back into the raw common intermediate. UNGATED by design: where the
// compare test failed sens >= layer, so MIN is a no-op there; where the
// sensor revealed, its depth joins the layer for the global verdict.
// Plain scalar math only (no mat3/ternary on this driver path).

layout(push_constant) uniform Push
{
	// x = near (m), y = far (m, may be inf).
	vec2 near_far;
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;

float linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		// Infinite far (Quest reports +inf). D=0 -> near, D=1 -> far.
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

void main()
{
	float vi = float(gl_ViewIndex);
	vec3 buv = vec3(uv, vi);
	float d = texture(sensor_tex, buv).r;
	float sens_lin = linearize_depth(d, pc.near_far.x, pc.near_far.y);
	out_color = vec4(sens_lin);
}
