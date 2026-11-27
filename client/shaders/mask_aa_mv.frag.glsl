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

// AA twin of mask_mv.frag: analytic edge coverage from the explicit
// barycentric varying, evaluated on a 2x2 ordered subpixel grid (+-0.25px)
// and averaged. The ramp is evaluated in pixel units (bary divided by
// its screen-space gradient magnitude: signed pixel-distance to each
// edge), edge-centered smoothstep(-0.5, +0.5) — point-symmetric, so
// watertight neighbours' complements sum to exactly 1.0 at every tap
// regardless of triangle heights (provided each 2x2 quad is covered
// by a single triangle; subpixel-dense quads fall back per the note
// below). The rasterizer runs conservative-overestimate, so the
// neighbour across a shared edge shades this pixel too; ADD (blend ONE,
// ONE, saturated by the UNORM store) sums interior straddles to 1.0,
// while a lone silhouette fragment keeps its smooth single-sided ramp —
// continuous in subpixel phase, no quantization steps. No depth (order
// independent); backfaces are culled at rasterizer level (front-side
// viewing only), so behind-geometry can never saturate the ramp.
// Reversed cuts stay binary MIN.
// Push block byte-identical to mask_mv (132B). View-agnostic (no
// gl_ViewIndex use), so no multiview extension is required here — the
// vertex stage selects the eye.
layout(location = 0) noperspective in vec3 vBary;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	float opacity;
}
pc;

layout(location = 0) out vec4 out_color;

float smoothcov(vec3 q)
{
	// Pixel-unit edge distances: the uniform 1px ramp reaches 1.0 half
	// a pixel inside each edge, 0 half a pixel outside (silhouette
	// keeps a smooth single-sided ramp with a soft outer toe,
	// fail-safe). Under quad/primitive mixing (subpixel-dense quads)
	// den is arbitrary: overestimated widths flatten toward 0.5 that
	// ADD whitens; underestimated widths go binary but still complement
	// by sign; lone dense silhouettes may alias ~0.5px.
	vec3 a3 = smoothstep(vec3(-0.5), vec3(0.5), q);
	return min(min(a3.x, a3.y), a3.z);
}

void main()
{
	// Screen-space barycentric gradients: one pixel of travel.
	vec3 gx = dFdx(vBary);
	vec3 gy = dFdy(vBary);
	// Per-edge gradient magnitude; q = bary/magnitude is the signed
	// pixel-distance to that edge. The 1e-6 floor only keeps the divide
	// defined for degenerate (zero-gradient, i.e. zero-area: a linear
	// bary constant over nonzero area forces all verts onto one line)
	// input, which the rasterizer culls — it is NOT fail-safe-white:
	// constant corner bary yields 0.5 gray, an outside fringe yields 0.
	// NaN bary (singular plane solve, driver-dependent) propagates
	// unsanitized, as in the previous revisions.
	vec3 den = max(vec3(length(vec2(gx.x, gy.x)), length(vec2(gx.y, gy.y)), length(vec2(gx.z, gy.z))), vec3(1e-6));
	vec3 gpi = (gx + gy) / den;
	vec3 gmi = (gx - gy) / den;
	vec3 q0 = vBary / den;
	float cov = smoothcov(q0 - 0.25 * gpi)
	          + smoothcov(q0 + 0.25 * gmi)
	          + smoothcov(q0 - 0.25 * gmi)
	          + smoothcov(q0 + 0.25 * gpi);
	out_color = vec4(pc.opacity * cov * 0.25);
}
