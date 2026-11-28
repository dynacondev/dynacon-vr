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

// Explicit silhouette band inner fragment: linear coverage ramped only
// inside the edge's own front triangle (stencil == 1 proves single
// front-facing coverage; containment proves it is THIS edge's
// surface). Front-over-behind and occluded quads fall outside the
// triangle, so valid geometry behind keeps binary with no depth test,
// no bias, no tuning — the test is resolution-exact by construction.
// Inner-draw blend is MIN (nearest-edge-wins into the 1 interior;
// overlapping quads take the lowest cov, i.e. the nearest edge). Closed
// boundary (== 0 paints) keeps the shared edge row gap-free; the
// fallback (discard) preserves binary underneath (fail-safe).
// Degenerate triangles (needle/collinear: all s_i ~ 0) paint —
// over-reveal by design, never a pinhole (non-forced silhouette edges
// always carry a non-degenerate front triangle: area0*area1 < 0 forces
// the picked half to nonzero area, so only forced boundary edges can
// hit this path).
// Same tent as the outer shader: identical cov(d), so the two stencil-
// disjoint halves join continuously at the edge row.
layout(location = 0) flat in vec2 fA;
layout(location = 1) flat in vec2 fB;
layout(location = 2) flat in float fSide;
layout(location = 3) flat in float fOpacity;
layout(location = 4) flat in vec2 fC;

layout(location = 0) out vec4 out_color;

void main()
{
	vec2 p = gl_FragCoord.xy;
	// Same-side test, unnormalized (orientation sign == fSide).
	float s0 = (fB.x - fA.x) * (p.y - fA.y) - (fB.y - fA.y) * (p.x - fA.x);
	float s1 = (fC.x - fB.x) * (p.y - fB.y) - (fC.y - fB.y) * (p.x - fB.x);
	float s2 = (fA.x - fC.x) * (p.y - fC.y) - (fA.y - fC.y) * (p.x - fC.x);
	if (fSide * s0 < 0.0 || fSide * s1 < 0.0 || fSide * s2 < 0.0)
		discard;
	vec2 e = fB - fA;
	float len2 = dot(e, e);
	vec2 t = len2 > 1e-18 ? e / sqrt(len2) : vec2(1.0, 0.0);
	vec2 n = vec2(-t.y, t.x);
	float d = fSide * dot(p - fA, n);
	float cov = clamp(d + 0.5, 0.0, 1.0);
	out_color = vec4(fOpacity * cov);
}
