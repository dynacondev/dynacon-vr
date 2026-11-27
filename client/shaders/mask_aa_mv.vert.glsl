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

// AA twin of mask_mv.vert: adds explicit per-corner barycentrics
// (location 1) plus static object-space barycentric gradients
// (locations 2-3, identical on all 3 verts of a triangle). Meshes upload
// de-indexed (3 verts per triangle, corners (1,0,0), (0,1,0), (0,0,1))
// so shared verts need no splitting. Barycentric is noperspective
// (screen-linear): the fragment evaluates coverage at screen-space
// subpixel offsets, for which perspective-correct interpolation would
// be wrong. vProj carries (ndc.xy, 1/w): both are affine in screen
// pixels, so noperspective interpolation is exact per fragment. Push
// block declares the full 140B (binary mask_mv declares the 132B
// prefix subset: ranges may exceed the declared block, shared-prefix
// offsets agree).
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 bary;
layout(location = 2) in vec3 grad1;
layout(location = 3) in vec3 grad2;

layout(location = 0) noperspective out vec3 vBary;
layout(location = 1) noperspective out vec3 vProj;
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
	vBary = bary;
	vG1 = grad1;
	vG2 = grad2;
	vec4 clip = pc.mvp[gl_ViewIndex] * vec4(pos, 1.0);
	gl_Position = clip;
	vProj = vec3(clip.xy / clip.w, 1.0 / clip.w);
}
