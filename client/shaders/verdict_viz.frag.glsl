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

#extension GL_GOOGLE_include_directive : require

#include "depth_viz_common.glsl.inc"

// Verdict debug view: samples the binary R8 verdict (1 = passthrough
// reveal, 0 = VR wins) and paints green / red. Geometry-free pixels
// (layer FAR >= 65000: no-window frames, reversed-only cuts, hidden,
// debug cutouts) read opaque black regardless of the verdict value,
// so empty frames never read all-green. Far-vs-far reads red (VR wins),
// same as the verdict.
//
// No Y-flip: both targets share the top-left UV convention of the
// fullscreen triangle.
//
// Plain scalar math only (no mat3/ternary on this driver path).

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray verdict_tex;
layout(set = 0, binding = 1) uniform sampler2DArray layer_tex;

layout(push_constant) uniform pc
{
	int eye;
	float near_z;
	float far_z;
	float opacity;
} push;

void main()
{
	float e = float(push.eye);
	float v = texture(verdict_tex, vec3(v_uv, e)).r;
	float l = texture(layer_tex, vec3(v_uv, e)).r;

	float empty = step(65000.0, l);
	vec3 col = mix(vec3(0.9, 0.15, 0.1), vec3(0.1, 0.9, 0.2), v);
	col = mix(col, vec3(0.0), empty);

	out_color = vec4(col, push.opacity);
}
