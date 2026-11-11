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

// Multiview twin of volume.vert.glsl: identical transform, per-eye MVP
// selected by gl_ViewIndex. One draw broadcasts to both layers, halving
// the volume/VR raster begins. Plain integer indexing only (see
// AGENTS.md: no mat3/ternary on this driver path). The fragment stage
// reads only near_z/far_z, so the push mirrors volume_push with an
// MVP pair (136B — same >128B note as the hidden/silhouette MV pushes;
// Adreno 740 provides 256B).
layout(location = 0) in vec3 pos;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	float near_z;
	float far_z;
}
pc;

layout(location = 0) out float eye_depth;

void main()
{
	vec4 clip = pc.mvp[gl_ViewIndex] * vec4(pos, 1.0);
	gl_Position = clip;
	// Planar eye distance (-z view): clip.w for the perspective
	// projection the MVP came from. The fragment stage converts it
	// to sensor-compatible D.
	eye_depth = clip.w;
}
