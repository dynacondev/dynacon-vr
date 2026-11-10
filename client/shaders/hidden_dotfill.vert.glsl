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

// Depth-fill pass for the hidden dot overlay: plain MVP raster of the
// mesh triangles into the depth attachment (color writes off). The dot
// billboards then depth-test against it, so occluded faces never show
// through. Same push layout as the dots pair (subset use).
layout(location = 0) in vec3 pos;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color_pitch;
	vec4 view_params;
}
pc;

void main()
{
	gl_Position = pc.mvp * vec4(pos, 1.0);
}
