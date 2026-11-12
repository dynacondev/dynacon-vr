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

// Mask merge: per-pixel MAX of up to 8 submitted mask images
// (feather groups + debug cutout group + hidden group). Every input
// covers the same FOV, so one shared UV addresses all of them; the
// linear sampler's bilinear tap IS the upscale for half/quarter-res
// inputs (same as the compositor stretch it replaces).
//
// MAX (any-mask-reveals-wins) preserves the two safety theorems:
// hidden-wins-reversed (hidden only adds coverage) and over-reveal
// (max >= every input, so no window can shrink). It differs from the
// old saturating-ADD union only in overlap bands (0.6+0.6 saturated
// to 1.0, MAX gives 0.6) — the double-count was a bug, not a feature.
//
// 8 STATIC bindings (not a sampler array): no descriptor-indexing
// extension, no dynamic indexing for Adreno to miscompile, no
// per-fragment early-out (loop-free, derivatives stay uniform).
// Unused slots bind an already-active input's view again (same view
// in several slots is legal); a uniform push-constant count skips
// taps past the active inputs (uniform control flow — skipped
// texture() calls never execute — so cost scales with real inputs,
// not with the 8 slots).
// RGB is zero: downstream the REPLACE blend preserves dst color and
// the veil bias touches alpha only, so mask RGB is ignored.
layout(set = 0, binding = 0) uniform sampler2DArray src0;
layout(set = 0, binding = 1) uniform sampler2DArray src1;
layout(set = 0, binding = 2) uniform sampler2DArray src2;
layout(set = 0, binding = 3) uniform sampler2DArray src3;
layout(set = 0, binding = 4) uniform sampler2DArray src4;
layout(set = 0, binding = 5) uniform sampler2DArray src5;
layout(set = 0, binding = 6) uniform sampler2DArray src6;
layout(set = 0, binding = 7) uniform sampler2DArray src7;

// Active input count (1..8), pushed per draw. Uniform across the
// draw, so the branches below are uniform control flow: skipped
// taps never execute (no derivatives in this shader, nothing to
// diverge). Static indexing throughout — Adreno-safe.
layout(push_constant) uniform Push
{
	int count;
}
pc;

void main()
{
	vec3 buv = vec3(uv, float(gl_ViewIndex));
	float a = texture(src0, buv).a;
	if (pc.count > 1)
		a = max(a, texture(src1, buv).a);
	if (pc.count > 2)
		a = max(a, texture(src2, buv).a);
	if (pc.count > 3)
		a = max(a, texture(src3, buv).a);
	if (pc.count > 4)
		a = max(a, texture(src4, buv).a);
	if (pc.count > 5)
		a = max(a, texture(src5, buv).a);
	if (pc.count > 6)
		a = max(a, texture(src6, buv).a);
	if (pc.count > 7)
		a = max(a, texture(src7, buv).a);
	out_color = vec4(0.0, 0.0, 0.0, a);
}
