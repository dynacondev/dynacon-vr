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

// Splat compare: raster-vs-sensor validation over the splatted planar
// depth (R16F, meters, display space) against the owned R16F common
// depth layer. Green = agree, red ramp = |layer - sensor| over 0..2m.
// No-data on either side (layer FAR >= 65000, splat clear 65504) reads
// opaque black. Flat quads participate as background depth, so
// disocclusions compare instead of quarantining; only fully-invalid
// windows (no-data/behind-camera in all four corners) leave FAR seams. Standard 4-byte eye push,
// like all viz fragment shaders.

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray layer_tex;

layout(set = 0, binding = 1) uniform sampler2DArray splat_tex;

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
	float l = texture(layer_tex, vec3(v_uv, e)).r;
	float ds = texture(splat_tex, vec3(v_uv, e)).r;

	float absent = step(65000.0, l);
	float no_return = step(65000.0, ds);
	float noval = max(absent, no_return);

	float lc = min(l, 10.0);
	float dc = min(ds, 10.0);
	float t = clamp(abs(lc - dc) * 0.5, 0.0, 1.0);
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);
	col = mix(col, vec3(0.0), noval);

	out_color = vec4(col, u.opacity);
}
