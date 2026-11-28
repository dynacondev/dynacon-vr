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

// Explicit silhouette band fragment: linear coverage from the signed
// pixel distance to the loop edge, positive inside. clamp(d + 0.5) is
// the 1px tent: 1.0 a pixel inside, 0.0 a pixel outside, affine so any
// downstream bilinear upscale reproduces it exactly. Outer-draw blend
// is MAX (nearest-edge-wins over the 0 background; overlapping quads
// take the highest cov, i.e. the nearest edge). No textures, no
// derivatives, ~6 ALU; interior pixels never execute this shader (no
// band quad covers them), so internal triangulation is invisible by
// construction and box-min erode downstream sees the same binary
// interiors as before.
layout(location = 0) flat in vec2 fA;
layout(location = 1) flat in vec2 fB;
layout(location = 2) flat in float fSide;
layout(location = 3) flat in float fOpacity;

layout(location = 0) out vec4 out_color;

void main()
{
	vec2 e = fB - fA;
	float len2 = dot(e, e);
	vec2 t = len2 > 1e-18 ? e / sqrt(len2) : vec2(1.0, 0.0);
	vec2 n = vec2(-t.y, t.x);
	float d = fSide * dot(gl_FragCoord.xy - fA, n);
	float cov = clamp(d + 0.5, 0.0, 1.0);
	out_color = vec4(fOpacity * cov);
}
