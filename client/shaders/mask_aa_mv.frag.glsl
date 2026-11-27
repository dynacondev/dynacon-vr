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
// and averaged. The ramp is edge-centered — smoothstep(-0.5*fw, +0.5*fw)
// is point-symmetric, so watertight neighbours' complements sum to exactly
// 1.0 at every tap (a one-sided 0..fw ramp would sum to ~0 at the edge
// center instead). The rasterizer runs conservative-overestimate, so the
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

float smoothcov(vec3 b, vec3 fw)
{
	// Edge-centered ramp: point-symmetric, so complements across a
	// shared edge sum to exactly 1.0. Half-width 0.5*fw keeps the 1px
	// ramp (silhouette reaches 1.0 half a pixel inside, 0 half a pixel
	// outside — a soft outer toe, fail-safe).
	vec3 a3 = smoothstep(-0.5 * fw, 0.5 * fw, b);
	return min(min(a3.x, a3.y), a3.z);
}

void main()
{
	// eps keeps smoothstep defined when fwidth is 0 (degenerate UV).
	// Upper clamp at 1.0: on subpixel-dense tessellation adjacent quad
	// pixels sit in different triangles, so dFdx measures inter-triangle
	// jumps instead of the gradient and fw explodes — flattening the
	// ramp into a near-constant ~0.5 that ADD saturates everywhere.
	// Bary units are self-normalizing per triangle, so capping the ramp
	// at 1 bary unit restores spatial selectivity (outside offsets read
	// true 0) while leaving normal triangles bit-identical.
	vec3 fw = min(fwidth(vBary) + vec3(1e-4), vec3(1.0));
	// Screen-space barycentric gradients: one pixel of travel.
	vec3 gx = dFdx(vBary);
	vec3 gy = dFdy(vBary);
	vec3 gp = gx + gy;
	vec3 gm = gx - gy;
	float cov = smoothcov(vBary - 0.25 * gp, fw)
	          + smoothcov(vBary + 0.25 * gm, fw)
	          + smoothcov(vBary - 0.25 * gm, fw)
	          + smoothcov(vBary + 0.25 * gp, fw);
	out_color = vec4(pc.opacity * cov * 0.25);
}
