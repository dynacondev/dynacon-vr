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

// Fused virtual-depth decode + exact global verdict at W (half merge
// res). One fullscreen pass, no intermediate D canvas: the SBS layout
// is trivial (left eye x[0..0.5), select half by gl_ViewIndex).
//
// layer_tex: R16F common layer (linear meters, MIN of the positive
// window draws, FAR clear 65504 reads as absent). NEAREST: the binary
// verdict (never the depths) bilinear-upsamples in the apply pass, so
// no depth halo.
//
// virt_tex: decoded stream-3 SBS frame (single image, both eyes side
// by side), sampled with the decoder's own YCbCr sampler. Green
// channel: smallest chroma-crosstalk gain under the 709 conversion,
// so lossy chroma wobble perturbs depth least through green (same
// convention as the old volume_virt decode).
//
// SBS encoding: 10-bit 2mm/LSB luma, server-clamped at 2.040m (code
// 1020) so beyond-range VR reads as near and occludes; 1021+ is a
// lossy-overshoot guard band, decoded as far. No near/far push: SBS
// codes are already meters (code * 0.002), so neither side needs a
// frustum to agree.
//
// verdict: reveal = !(present && virt strictly nearer). Exact, no
// bias anywhere — crisp occlusion edges are intentional. Tie keeps
// the window. Geometry-free pixels (reversed-only cuts, hidden,
// debug cutouts never write the layer) read FAR and keep whatever
// the merged union says. Global: applies to every group at once,
// after all alpha is combined; reversed cuts stay per-group feather
// alpha and are unaffected here.
//
// Layout note: the decoder image is normalized to GENERAL at frame
// top on the main command buffer and sampled read-only here under
// the same steady-state discipline as the video path. Transition
// frames sit the verdict out scene-side (depth_verdict_safe: neutral
// merged submit, fail-open), so this pass only ever samples
// steady-state GENERAL — including on the async mask path.
//
// Plain scalar math only (no mat3/ternary on this driver path).
layout(set = 0, binding = 0) uniform sampler2DArray layer_tex;
layout(set = 0, binding = 1) uniform sampler2D virt_tex;

void main()
{
	float vi = float(gl_ViewIndex);
	vec3 buv = vec3(uv, vi);
	float l = texture(layer_tex, buv).r;

	vec2 vuv = vec2(uv.x * 0.5 + vi * 0.5, uv.y);
	float vd = texture(virt_tex, vuv).g;

	float virt_m;
	if (vd >= 1020.5 / 1023.0)
	{
		// Lossy overshoot past the server clamp: no virtual
		// occluder (compare no-op).
		virt_m = 1e9;
	}
	else
	{
		virt_m = max(vd * 2.046, 1e-3);
	}

	float layer_present = 1.0 - step(65000.0, l);
	float vr_wins = 1.0 - step(l, virt_m);
	float reveal = 1.0 - layer_present * vr_wins;
	out_color = vec4(reveal);
}
