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

// AA-spike twin of mask_mv.frag: analytic edge coverage from the explicit
// barycentric varying. coverage = min over smoothstep(0, fwidth, bary):
// 1.0 interior, ~0.5 on the silhouette edge. REPLACE blend, no depth
// (positives-only tier-0/tier-1 groups; reversed groups stay on the binary
// mask_mv path). Push block byte-identical to mask_mv (132B). View-agnostic
// (no gl_ViewIndex use), so no multiview extension is required here — the
// vertex stage selects the eye.
//
// SPIKE NOTES (device run decides):
// - Single-sample raster shades one side of a shared interior edge
//   (tie-break), so watertight interiors may show a ~0.5 hairline where
//   the binary path wrote 1.0.
// - REPLACE with no depth is order-dependent on overlap: a grey foreground
//   edge REPLACEs (dims) a background 1.0 interior where binary 1-over-1
//   was invisible. Disjoint/exterior-dominated windows are unaffected.
// - Tier-0 binary punches (marker cutouts, sensor reveal) stamp hard
//   values over the grey ramp with no reconciling blur (tier-1 reconciles
//   via its blur chain).
layout(location = 0) in vec3 vBary;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	float opacity;
}
pc;

layout(location = 0) out vec4 out_color;

void main()
{
	// eps keeps smoothstep defined when fwidth is 0 (degenerate UV).
	vec3 fw = fwidth(vBary) + vec3(1e-4);
	vec3 a3 = smoothstep(vec3(0.0), fw, vBary);
	float cov = min(min(a3.x, a3.y), a3.z);
	out_color = vec4(pc.opacity * cov);
}
