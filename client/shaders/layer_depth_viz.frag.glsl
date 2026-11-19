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

// Composited-layer debug false-color: samples the owned R16F common
// depth layer (linear meters, 2-layer array, half merge res) and maps
// 0..2.046m green->red to match the VR-depth viz, so the two are
// directly comparable. Far windows (>2.041m, same cut as the verdict)
// saturate red; FAR clear (65504, geometry-free pixels incl.
// no-window frames, reversed-only cuts, hidden and debug cutouts at
// >=65000) reads opaque black, never transparent.
//
// No Y-flip: the layer shares the top-left UV convention of the
// fullscreen triangle it was rasterized with.
//
// Plain scalar math only (no mat3/ternary on this driver path).

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray layer_tex;

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
	float l = texture(layer_tex, vec3(v_uv, float(push.eye))).r;

	float empty = step(65000.0, l);
	float d = clamp(l, 0.0, 2.046);
	float t = d / 2.046;
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);
	col = mix(col, vec3(0.0), empty);

	out_color = vec4(col, u.opacity);
}
