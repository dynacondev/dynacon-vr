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

// VR-depth viz with the composited raster pasted on top: samples the
// decoded SBS virtual-depth frame (single image, both eyes side by
// side) with the frame handle's own generation sampler and maps 0..2.046m green->red
// (far past 1020 reads black, same decode convention as the verdict:
// green channel, 2mm/LSB, server clamp + guard band), then replaces
// every pixel where the owned R16F common depth layer holds a window
// (linear meters, 2-layer array, half merge res) with the same
// green->red ramp of the raster depth. Scale mismatches between the
// two paths read directly off the shared ramp: same wall, different
// color = the paths disagree by that ramp distance.
//
// No Y-flip on either tap: the SBS image shares the top-left UV
// convention of the fullscreen triangle (verdict path), and so does
// the raster layer it was co-rendered with.
//
// Plain scalar math only (no mat3/ternary on this driver path).

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D virt_tex;

layout(set = 0, binding = 1) uniform sampler2DArray layer_tex;

layout(set = 0, binding = 2) uniform params
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

	float vempty = step(1020.5 / 1023.0, vd);
	float v = clamp(vd * 2.046, 0.0, 2.046);
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), v / 2.046);
	col = mix(col, vec3(0.0), vempty);

	float l = texture(layer_tex, vec3(v_uv, e)).r;
	float present = 1.0 - step(65000.0, l);
	float d = clamp(l, 0.0, 2.046);
	vec3 lcol = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), d / 2.046);
	col = mix(col, lcol, present);

	out_color = vec4(col, u.opacity);
}
