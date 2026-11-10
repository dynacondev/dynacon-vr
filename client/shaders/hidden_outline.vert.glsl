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

// Inverted-hull shell vertex stage: plain MVP plus a radial push of each
// vertex away from the model-space shell center. The occluder prepass runs
// the same shader with thickness 0 (exact original silhouette); the rim
// pass expands ~2cm so its backfaces peek out around the occluder depth.
// No varyings: the fragment stage is a flat color.
layout(location = 0) in vec3 pos;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color; // rgb: outline color (read by the fragment stage)
	vec4 center_thickness; // xyz: model-space shell center (m), w: expand (m)
}
pc;

void main()
{
	vec3 d = pos - pc.center_thickness.xyz;
	float l = length(d);
	vec3 p = pos + (l > 1e-6 ? d / l * pc.center_thickness.w : vec3(0.0));
	gl_Position = pc.mvp * vec4(p, 1.0);
}
