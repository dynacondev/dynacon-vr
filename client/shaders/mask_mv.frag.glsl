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

// Multiview silhouette fragment: flat silhouette (opacity
// replicated to all channels), push block widened to the multiview pair
// so the shared 132B layout reads opacity at the right offset (a
// mismatched single-MVP block here would read mvp[1][0][0] as alpha —
// orientation-dependent garbage through the DST_ALPHA blend).
layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	float opacity;
}
pc;

layout(location = 0) out vec4 out_color;

void main()
{
	out_color = vec4(pc.opacity);
}
