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

// Flat guardian outline color, opaque. Deliberately no lighting, no
// derivatives, no varyings: every covered pixel costs ~nothing, which is
// the point (the triplanar grid this replaces burned ~80 ALU + fwidths
// per pixel over the full silhouette).
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color; // rgb: outline color
	vec4 center_thickness;
}
pc;

void main()
{
	out_color = vec4(pc.color.rgb, 1.0);
}
