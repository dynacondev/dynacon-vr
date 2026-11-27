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

// AA twin of mask_mv.frag: analytic edge coverage from the explicit
// barycentric varying, evaluated on a 2x2 ordered subpixel grid (+-0.25px)
// and averaged. Screen-space gradients are analytic: object-space
// barycentric gradients (static vertex attributes) transformed through
// the per-fragment projective Jacobian (MVP rows + interpolated
// (ndc.xy, 1/w)). Every lane uses only its own triangle's data, so
// quads straddling a shared edge are exact by construction — no dFdx
// anywhere. The ramp is evaluated in pixel units (bary divided by its
// screen-space gradient magnitude: signed pixel-distance to each edge),
// edge-centered smoothstep(-0.5, +0.5) — point-symmetric, so watertight
// neighbours' complements sum to exactly 1.0 at every tap regardless
// of triangle heights. The rasterizer runs conservative-overestimate,
// so the neighbour across a shared edge shades this pixel too; ADD
// (blend ONE, ONE, saturated by the UNORM store) sums interior
// straddles to 1.0, while a lone silhouette fragment keeps its smooth
// single-sided ramp — continuous in subpixel phase. No depth (order
// independent); backfaces are culled at rasterizer level (front-side
// viewing only), so behind-geometry can never saturate the ramp.
// Reversed cuts stay binary MIN.
// Push block declares the full 140B incl. inv_view (reciprocal
// viewport scale); binary mask_mv declares the 132B prefix subset.
layout(location = 0) noperspective in vec3 vBary;
layout(location = 1) noperspective in vec3 vProj;
layout(location = 2) flat in vec3 vG1;
layout(location = 3) flat in vec3 vG2;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	float opacity;
	float inv_view_w;
	float inv_view_h;
}
pc;

layout(location = 0) out vec4 out_color;

float smoothcov(vec3 q)
{
	// Pixel-unit edge distances: the uniform 1px ramp reaches 1.0 half
	// a pixel inside each edge, 0 half a pixel outside (silhouette
	// keeps a smooth single-sided ramp with a soft outer toe,
	// fail-safe). Under quad/primitive mixing the old dFdx form fell
	// back to ~0.5-flat; the analytic form has no mixing. Degenerate
	// input drives den to its floor, q0 saturates, and the per-edge
	// min drives coverage to 0 (fail-safe: zero-area tris emit no
	// fragments anyway).
	vec3 a3 = smoothstep(vec3(-0.5), vec3(0.5), q);
	return min(min(a3.x, a3.y), a3.z);
}

vec3 tap(vec3 q, vec2 e1, vec2 e2, vec2 e0, vec2 o)
{
	return q + vec3(dot(o, e1), dot(o, e2), dot(o, e0));
}

void main()
{
	// Projective Jacobian at the fragment: A0/A1 are the clip rows
	// corrected by the perspective divide (all from the fragment's own
	// interpolated position — no derivatives). N is the object-space
	// triangle normal recovered from the static gradients (scale drops
	// out of every quotient below).
	mat4 M = pc.mvp[gl_ViewIndex];
	vec3 R0 = vec3(M[0][0], M[1][0], M[2][0]);
	vec3 R1 = vec3(M[0][1], M[1][1], M[2][1]);
	vec3 R3 = vec3(M[0][3], M[1][3], M[2][3]);
	float nx = vProj.x, ny = vProj.y, al = vProj.z;
	vec3 A0 = R0 - nx * R3;
	vec3 A1 = R1 - ny * R3;
	vec3 n6 = cross(vG1, vG2);
	vec3 N = n6 / max(dot(n6, n6), 1e-24);
	float s = al * dot(cross(A0, A1), N);
	// Sign-preserving floor (mandatory: guards the 0*Inf of guarded
	// degenerates and near-silhouette projections).
	float mag = abs(s);
	s = (s < 0.0f ? -1.0f : 1.0f) * max(mag, 1e-12);
	vec3 tx = cross(A1, N) / s;
	vec3 ty = cross(N, A0) / s;
	// Screen-space barycentric gradients (per pixel), then pixel
	// distances and their unit directions per edge.
	vec2 iv = vec2(pc.inv_view_w, pc.inv_view_h);
	vec2 g1s = vec2(dot(vG1, tx), dot(vG1, ty)) * iv;
	vec2 g2s = vec2(dot(vG2, tx), dot(vG2, ty)) * iv;
	vec2 g0s = -(g1s + g2s);
	vec3 den = max(vec3(length(g1s), length(g2s), length(g0s)), vec3(1e-6));
	float b0 = 1.0 - vBary.x - vBary.y;
	// NOTE the component order: vBary.x is B0 (1 at p0) while vG1 is
	// grad-B1 (1 at p1), so numerators are (y, b0, x) to pair each
	// bary with its OWN gradient magnitude. min() is order-insensitive
	// but this division is not — rotating it reintroduces
	// height-dependent seams (verified numerically: -12% dips).
	vec3 q0 = vec3(vBary.y, b0, vBary.x) / den;
	vec2 e1 = g1s / den.x;
	vec2 e2 = g2s / den.y;
	vec2 e0 = g0s / den.z;
	float cov = smoothcov(tap(q0, e1, e2, e0, vec2(-0.25, -0.25)))
	          + smoothcov(tap(q0, e1, e2, e0, vec2(0.25, -0.25)))
	          + smoothcov(tap(q0, e1, e2, e0, vec2(-0.25, 0.25)))
	          + smoothcov(tap(q0, e1, e2, e0, vec2(0.25, 0.25)));
	out_color = vec4(pc.opacity * cov * 0.25);
}
