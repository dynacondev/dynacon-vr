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

// Raster-vs-sensor validation view: compares the owned R16F common depth
// layer (linear meters, 2-layer array, half merge res) against the
// runtime D16 environment depth (linearized in-shader with the UBO
// near/far, same math as depth_viz.frag), per eye. Green = agree,
// red ramp = |layer - sensor| over 0..2m. Both-far reads green (caps at
// 10m match, same far-vs-far convention as the verdict).
// Geometry-free pixels (layer FAR >= 65000: no-window frames,
// reversed-only cuts, hidden, debug cutouts) and sensor no-returns
// (D >= 0.9999) read opaque black — neither side can validate there.
//
// Eye-UV caveat: the layer is rasterized on the server video-eye
// timebase, the sensor is live. Normalized-UV taps assume the same eye,
// so trust only static scenes (ideally fronto-parallel, centre-weighted);
// contours and motion show registration red, not value error.
//
// No Y-flip on the layer (top-left convention of the fullscreen
// triangle it was rasterized with); the sensor texture is bottom-up,
// sampled flipped like depth_viz.frag.
//
// Plain scalar math only (no mat3/ternary on this driver path).

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray layer_tex;

layout(set = 0, binding = 1) uniform sampler2DArray sensor_tex;

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

float linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		// Infinite far plane. D=0 -> near, D=1 -> far.
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

void main()
{
	float e = float(push.eye);
	float l = texture(layer_tex, vec3(v_uv, e)).r;
	float ds = texture(sensor_tex, vec3(v_uv.x, 1.0 - v_uv.y, e)).r;

	float absent = step(65000.0, l);
	float no_return = step(0.9999, ds);
	float noval = max(absent, no_return);

	float d = linearize_depth(ds, u.near_far.x, u.near_far.y);
	float lc = min(l, 10.0);
	float dc = min(d, 10.0);
	float t = clamp(abs(lc - dc) * 0.5, 0.0, 1.0);
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);
	col = mix(col, vec3(0.0), noval);

	out_color = vec4(col, u.opacity);
}
