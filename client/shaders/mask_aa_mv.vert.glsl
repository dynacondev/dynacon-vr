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

// AA twin of mask_mv.vert: adds the triangle's barycentric plane
// coefficients (location 1: c1, c2 with B_i(p) = g_i.p + c_i; location
// 2-3: object-space gradients g1, g2 — all identical on the 3 verts of
// a triangle, meshes upload de-indexed). vP carries the object-space
// position with default (perspective-correct) interpolation, which
// reconstructs the exact 3D surface point per fragment — the shader
// evaluates true projective barycentrics from it. Push block declares
// the full 140B (binary mask_mv declares the 132B prefix subset).
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 vc;
layout(location = 2) in vec3 grad1;
layout(location = 3) in vec3 grad2;

layout(location = 0) flat out vec2 vC;
layout(location = 1) out vec3 vP;
layout(location = 2) flat out vec3 vG1;
layout(location = 3) flat out vec3 vG2;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	// First-acquisition fade: global silhouette opacity, ramps 0 -> 1.
	float opacity;
	float inv_view_w;
	float inv_view_h;
}
pc;

void main()
{
	vC = vc;
	vP = pos;
	vG1 = grad1;
	vG2 = grad2;
	gl_Position = pc.mvp[gl_ViewIndex] * vec4(pos, 1.0);
}
