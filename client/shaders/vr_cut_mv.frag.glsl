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

// VR cut: keep (1) unless virtual is strictly nearer than the rasterized
// known geometry (passthrough + hidden + hands, reversed excluded).
// Both inputs are sensor-compatible D in [0,1] at the cut working res
// (min(group submit, virtual eye)), so no reprojection and no push
// constants: the same eye UV taps both. Empty (far clear, 1.0) on
// either side keeps (fail open to reveal): the soft sensor pass, not
// this cut, decides unmodeled content. Binary verdict; the shared
// group blur feathers the edge exactly like a silhouette cut.

layout(set = 0, binding = 0) uniform sampler2DArray virt_tex;
layout(set = 0, binding = 1) uniform sampler2DArray exact_tex;

void main()
{
	float vi = float(gl_ViewIndex);
	vec3 buv = vec3(uv, vi);
	float v = texture(virt_tex, buv).r;
	float e = texture(exact_tex, buv).r;
	float v_present = 1.0 - step(0.9999, v);
	float e_present = 1.0 - step(0.9999, e);
	float vr_wins = (1.0 - step(e, v)) * v_present * e_present;
	out_color = vec4(1.0 - vr_wins);
}
