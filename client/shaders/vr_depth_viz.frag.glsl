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

// VR-depth debug false-color: samples the decoded SBS virtual-depth
// frame (single image, both eyes side by side) with the decoder's own
// sampler and maps 0..2.046m green->red. Same decode convention as the
// verdict: green channel (smallest chroma-crosstalk gain under the 709
// conversion), 2mm/LSB, valid codes 0-1020 (0-2.040m); holes, sky and
// beyond-range pin to 1023, anything past 1020 decodes as far (black).
// Far reads opaque black (same as the layer and verdict viz modes),
// never transparent.
//
// No Y-flip: the SBS image shares the top-left UV convention of the
// fullscreen triangle (verdict path), unlike the runtime D16 which is
// bottom-up.
//
// Plain scalar math only (no mat3/ternary on this driver path).

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D virt_tex;

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
	float e = float(push.eye);
	vec2 vuv = vec2(v_uv.x * 0.5 + e * 0.5, v_uv.y);
	float vd = texture(virt_tex, vuv).g;

	float empty = step(1020.5 / 1023.0, vd);
	float d = clamp(vd * 2.046, 0.0, 2.046);
	float t = d / 2.046;
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);
	col = mix(col, vec3(0.0), empty);

	out_color = vec4(col, u.opacity);
}
