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

// Plain MVP plus the object-space position for the fragment-stage dot
// pattern. No placement rebuild, no center math: the dots live in model
// space, which is world space up to the (uniform, ~1) placement scale.
layout(location = 0) in vec3 pos;

layout(location = 0) out vec3 vObj;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
}
pc;

void main()
{
	vObj = pos;
	gl_Position = pc.mvp * vec4(pos, 1.0);
}
