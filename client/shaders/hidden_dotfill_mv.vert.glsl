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

// Multiview twin of hidden_dotfill.vert.glsl: identical depth-fill
// raster, per-eye MVP selected by gl_ViewIndex. Same push layout as the
// dots multiview pair (subset use). Plain integer indexing only (see
// AGENTS.md: no mat3/ternary on this driver path).
layout(location = 0) in vec3 pos;

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	vec4 color_pitch;
	vec4 view_params[2];
}
pc;

void main()
{
	gl_Position = pc.mvp[gl_ViewIndex] * vec4(pos, 1.0);
}
