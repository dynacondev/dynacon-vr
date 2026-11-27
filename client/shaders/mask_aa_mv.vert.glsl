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

// AA twin of mask_mv.vert: adds an explicit per-corner barycentric
// attribute (location 1). Meshes upload de-indexed (3 verts per triangle,
// corners (1,0,0), (0,1,0), (0,0,1)) so shared verts need no splitting.
// Barycentric is noperspective (screen-linear): the fragment evaluates
// coverage at screen-space subpixel offsets, for which
// perspective-correct interpolation would be wrong. Push block
// byte-identical to mask_mv (132B): the shared VERT|FRAG range trap
// applies — never widen this block.
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 bary;

layout(location = 0) noperspective out vec3 vBary;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	// First-acquisition fade: global silhouette opacity, ramps 0 -> 1.
	float opacity;
}
pc;

void main()
{
	vBary = bary;
	gl_Position = pc.mvp[gl_ViewIndex] * vec4(pos, 1.0);
}
