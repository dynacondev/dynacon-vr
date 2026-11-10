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

layout(location = 0) in vec3 pos;

layout(location = 0) out vec3 vWorld;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 obj_pos_scale; // xyz: world position (m), w: uniform scale
	vec4 obj_quat; // xyzw orientation
	vec4 grid_col_scale; // rgb: grid color, w: grid spacing (m)
	vec4 cam_pos_width; // xyz: eye world position (m), w: line width (px)
}
pc;

vec3 qrot(vec4 q, vec3 v)
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main()
{
	vec3 wp = pc.obj_pos_scale.xyz + qrot(pc.obj_quat, pos * pc.obj_pos_scale.w);
	vWorld = wp;
	gl_Position = pc.mvp * vec4(pos, 1.0);
}
