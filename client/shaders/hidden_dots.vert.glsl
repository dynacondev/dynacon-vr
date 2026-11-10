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

// Plain MVP plus pitch-normalized position and view depth for the
// fragment-stage round dots. No placement rebuild, no normals.
layout(location = 0) in vec3 pos;

layout(location = 0) out vec3 vQ;
layout(location = 1) out float vViewZ;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
	vec4 view_params; // x: view W px, y: view H px, z: proj[1][1]
}
pc;

void main()
{
	vQ = pos / max(pc.color_pitch.w, 1e-4);
	vec4 clip = pc.mvp * vec4(pos, 1.0);
	// Perspective clip w is -z_eye (w row is (0,0,-1,0) in every
	// perspective variant including reversed-Z): positive view distance
	// in meters straight from w, no matrix needed.
	vViewZ = clip.w;
	gl_Position = clip;
}
