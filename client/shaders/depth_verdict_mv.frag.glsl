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

// Exact global verdict: R16F common layer (linear meters, windows +
// hidden + bbox + revealed sensor) vs the server virtual-depth D canvas
// at W. reveal = !(present && virt strictly nearer): exact, no bias
// anywhere global — crisp occlusion edges are intentional. Tie keeps the
// window (the step reads linV >= L as no-cut). The layer FAR clear
// (65504) reads as absent: reveal 1 there, so geometry-free pixels
// (hidden reveals and debug cutouts, which never write the layer) keep
// whatever the merged union says. Both taps NEAREST: the binary verdict
// (never the depths) bilinear-upsamples in the apply pass, so no depth
// halo.
// Global VR occlusion applies to all groups here; reversed cuts stay
// per-group (feather alpha). Plain scalar math only.

layout(push_constant) uniform Push
{
	// x = near (m), y = far (m, may be inf).
	vec2 near_far;
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray layer_tex;
layout(set = 0, binding = 1) uniform sampler2DArray virt_tex;

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
	float l = texture(layer_tex, buv).r;
	float v = texture(virt_tex, buv).r;

	float virt_lin = linearize_depth(v, pc.near_far.x, pc.near_far.y);
	float layer_present = 1.0 - step(65000.0, l);
	float vr_wins = 1.0 - step(l, virt_lin);
	float reveal = 1.0 - layer_present * vr_wins;
	out_color = vec4(reveal);
}
