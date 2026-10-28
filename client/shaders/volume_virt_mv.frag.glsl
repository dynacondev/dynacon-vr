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

// Virtual-depth SBS (640x320, one decoded stream-3 frame) to
// sensor-compatible D (R16, 320px, per-eye layers). Fullscreen _mv:
// one draw broadcasts to both layers, gl_ViewIndex selects the SBS half.
// Note: decoder samplers are linear, so taps at the SBS seam (u ~= 0.5)
// blend L/R eyes over ~1 texel — subsumed by the downstream feather,
// no inset applied.
//
// keeps the hot compare pass in D space (sensor < min(volume, virtual))
// and the compare push constants untouched (already at the 128B minimum
// guarantee): meters->D runs here, where near/far ride a tiny push.
//
// Sampling: green channel. The decoder's Y'CbCr sampler converts to RGB
// (709 full-range here); with the server's neutral chroma (128) all
// channels equal luma, and green has the smallest chroma-crosstalk gain
// (R: +1.57Cr', B: +1.86Cb' vs G: -0.19Cb' -0.47Cr'), so lossy chroma
// wobble perturbs depth least through green.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D virt_tex;

layout(push_constant) uniform Push
{
	float near_z;
	float far_z;
}
pc;

void main()
{
	float vi = float(gl_ViewIndex);
	vec2 vuv = vec2(uv.x * 0.5 + vi * 0.5, uv.y);
	float vd = texture(virt_tex, vuv).g;

	float D;
	// 253.5/255 threshold, not 254.5: the forced 709-full-range Y'CbCr
	// conversion reads neutral chroma ~0.65 LSB low through green, and
	// lossy ringing near silhouettes costs a few more LSBs. Sacrificing
	// the top 1.5cm of range (2.535m vs 2.55m) buys real margin against
	// phantom-near speckles that would wrongly kill far passthrough.
	if (vd >= 253.5 / 255.0)
	{
		// far/unknown/sky/hole: no virtual occluder (min no-op).
		D = 1.0;
	}
	else
	{
		float m = max(vd * 2.55, 1e-3);
		float n = pc.near_z;
		float f = pc.far_z;
		if (!(f > n) || f > 1e9)
		{
			// Infinite far plane (what the Quest runtime reports):
			// D = 0 at near, -> 1 at far.
			D = 1.0 - n / m;
		}
		else
		{
			D = f * (m - n) / (m * (f - n));
		}
	}
	out_color = vec4(clamp(D, 0.0, 1.0));
}
