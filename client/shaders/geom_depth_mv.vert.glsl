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

// Common depth-layer vertex shader: per-eye MVP selected by
// gl_ViewIndex. One draw broadcasts to both layers. Plain integer
// indexing only (see AGENTS.md: no mat3/ternary on this driver path).
// The fragment stage linearizes nothing: clip.w is already meters (see
// the frag derivation), so the push carries an MVP pair only (128B —
// exactly the Vulkan-guaranteed minimum, no maxPushConstantsSize throw
// needed; the layout range stays VERT-only and the fragment declares
// no push block, so the mask_mv offset trap is structurally
// impossible).
layout(location = 0) in vec3 pos;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
}
pc;

layout(location = 0) out float eye_depth;

void main()
{
	vec4 clip = pc.mvp[gl_ViewIndex] * vec4(pos, 1.0);
	gl_Position = clip;
	// Planar eye distance (-z view): clip.w for the perspective
	// projection the MVP came from. Same convention as mask_mv.
	eye_depth = clip.w;
}
