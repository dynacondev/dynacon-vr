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

// Window reveal field: sensor verdict (binding 0, R8 at W) minus the
// per-group reversed cut (binding 1, R16F linear meters). Binary verdict;
// the shared group blur feathers the edge.
// Cut-before-add is structural: the sensor verdict only ever adds
// coverage, and the global virt verdict (merge side) gates everything by
// VR depth afterwards — reversed cuts stay per-group here.
// The rev cut is presence-based (no nearer-test: the old window-depth
// term no longer exists in the depth group). Empty rev (far clear)
// never cuts. Tie-keeps-window is structural: a binary win has no depth
// race to break.

layout(set = 0, binding = 0) uniform sampler2DArray win_tex;
layout(set = 0, binding = 1) uniform sampler2DArray rev_tex;

void main()
{
	float vi = float(gl_ViewIndex);
	vec3 buv = vec3(uv, vi);
	float w = texture(win_tex, buv).r;
	float r = texture(rev_tex, buv).r;
	float w_on = step(0.5, w);
	float rev_on = 1.0 - step(65000.0, r);
	out_color = vec4(w_on * (1.0 - rev_on));
}
