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

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

// Mask-only apply: merged contributor alpha (NEAREST — the feather
// bands are already filtered) straight into the submit image, no
// verdict tap. Used when no virtual-depth frame paired (toggle off,
// decoder dead, transition frames): the verdict would clear to 1 and
// multiply by 1, so skipping the whole verdict pass + barrier and
// copying merged directly is pixel-identical with one tap instead of
// two. RGB stays zero (downstream the REPLACE blend preserves dst
// color, same as the verdict apply it replaces).
//
// Only binding 0 is declared: the pipeline shares the 2-binding
// apply layout (binding 1 is simply never written or sampled here).
layout(set = 0, binding = 0) uniform sampler2DArray merged_tex;

void main()
{
	vec3 buv = vec3(uv, float(gl_ViewIndex));
	float a = texture(merged_tex, buv).a;
	out_color = vec4(0.0, 0.0, 0.0, a);
}
