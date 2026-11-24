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

// Warp mesh-placement debug: outputs the interpolated sensor UV as
// color, bypassing depth decode, linearize and the anti-streak discard
// entirely. If this gradient swims correctly with head motion, vertex
// placement (mesh, matrices, UBO, descriptors) is proven and the bug
// is in depth values or the discard. If it is frozen/wrong, placement
// itself is broken. Same vertex shader, set layout and push shape as
// the warp pass (bindings simply go unused).

layout(location = 0) in vec2 v_suv;

layout(location = 0) out vec4 out_color;

void main()
{
	out_color = vec4(v_suv.x, v_suv.y, 0.5, 1.0);
}
