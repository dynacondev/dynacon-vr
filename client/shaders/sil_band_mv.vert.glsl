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

// Explicit silhouette band vertex shader: one instance per baked loop
// edge (inline positions a/b/c0/c1, per-instance rate), 4 verts as a
// triangle strip forming a ~2px screen-space quad centred on the edge
// with 1px square-cap extensions past both endpoints (gaps become
// overlaps: every ambiguity over-reveals, never pinholes).
//
// Per-eye exact: facing comes from the calling view's MVP
// (gl_ViewIndex selects, plain integer indexing, no mat3/ternary).
// Non-silhouette edges collapse to a behind-far point (zero
// fragments, static instance count, no compaction pass). Behind-camera
// edges (any w <= 0) collapse too: the binary raster clips those
// correctly, the band just stays out of its way (aliasing, no hole).
// Degenerate screen edges (< 1e-6 px) collapse (NaN-guard).
//
// c1 == c0 (bitwise, inline copies of one vertex) marks a forced emit:
// boundary, non-manifold or inconsistent winding. Fail-safe direction.
layout(location = 0) in vec3 a;
layout(location = 1) in vec3 b;
layout(location = 2) in vec3 c0;
layout(location = 3) in vec3 c1;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	float opacity;
	float vp_w;
	float vp_h;
}
pc;

layout(location = 0) flat out vec2 fA;
layout(location = 1) flat out vec2 fB;
layout(location = 2) flat out float fSide;
layout(location = 3) flat out float fOpacity;

void main()
{
	mat4 M = pc.mvp[gl_ViewIndex];
	vec4 ca = M * vec4(a, 1.0);
	vec4 cb = M * vec4(b, 1.0);
	vec4 cc0 = M * vec4(c0, 1.0);
	vec4 cc1 = M * vec4(c1, 1.0);
	vec2 vp = vec2(pc.vp_w, pc.vp_h);
	bool forced = all(equal(c0, c1));
	float mw = min(min(ca.w, cb.w), min(cc0.w, cc1.w));
	vec2 pa = (ca.xy / ca.w * 0.5 + 0.5) * vp;
	vec2 pb = (cb.xy / cb.w * 0.5 + 0.5) * vp;
	vec2 p0 = (cc0.xy / cc0.w * 0.5 + 0.5) * vp;
	vec2 p1 = (cc1.xy / cc1.w * 0.5 + 0.5) * vp;
	float area0 = (pb.x - pa.x) * (p0.y - pa.y) - (pb.y - pa.y) * (p0.x - pa.x);
	float area1 = (pa.x - pb.x) * (p1.y - pb.y) - (pa.y - pb.y) * (p1.x - pb.x);
	bool sil = forced || (mw > 1e-6 && area0 * area1 < 0.0);
	vec2 e = pb - pa;
	float len = length(e);
	if (!sil || len < 1e-6)
	{
		// Collapse: behind-far point, zero fragments.
		gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
		fA = vec2(0.0);
		fB = vec2(0.0);
		fSide = 1.0;
		fOpacity = 0.0;
		return;
	}
	vec2 t = e / len;
	vec2 n = vec2(-t.y, t.x);
	vec2 front = area0 > 0.0 ? p0 : p1;
	float s = dot(front - pa, n) >= 0.0 ? 1.0 : -1.0;
	// Strip corners 0..3: (a,-) (a,+) (b,-) (b,+).
	bool is_b = gl_VertexIndex >= 2;
	bool is_plus = (gl_VertexIndex == 1 || gl_VertexIndex == 3);
	vec2 base = is_b ? pb : pa;
	vec4 base_clip = is_b ? cb : ca;
	vec2 px = base + n * (is_plus ? 1.0 : -1.0) + t * (is_b ? 1.0 : -1.0);
	vec2 ndc_off = (px - base) / vp * 2.0;
	gl_Position = vec4(base_clip.xy + ndc_off * base_clip.w, base_clip.zw);
	fA = pa;
	fB = pb;
	fSide = s;
	fOpacity = pc.opacity;
}
