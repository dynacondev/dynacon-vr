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

// Multiview twin of mask.vert.glsl: identical transform, per-eye MVP
// selected by gl_ViewIndex. One draw broadcasts to both layers, halving
// the hidden renderpass begins (8 per-eye to 4 multiview). Plain integer
// indexing only (see AGENTS.md: no mat3/ternary on this driver path).
layout(location = 0) in vec3 pos;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	// First-acquisition fade: global silhouette opacity, ramps 0 -> 1.
	// Read by the fragment stage; the blur chain is linear in alpha so
	// the whole window (interior and feather band) fades uniformly.
	float opacity;
}
pc;

void main()
{
	gl_Position = pc.mvp[gl_ViewIndex] * vec4(pos, 1.0);
}
