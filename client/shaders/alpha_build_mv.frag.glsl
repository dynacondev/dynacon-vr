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

// Window reveal field from crisp (undilated) depths: full passthrough
// where a known window stands, VR is not strictly nearer, and no
// reversed cut wins. This runs BEFORE the exactOcc dilation, so VR and
// reversed cuts keep their full edge while the later sensor compare
// uses the forgiving (dilated) occluder. Cut-before-add is structural:
// the sensor verdict MAXes onto this field afterwards.
// Binary verdict; the shared group blur feathers the edge.
// Empty exact (far clear, 1.0) yields 0: nothing to reveal (this differs
// from the retired vr_cut keep field, whose seed already held
// silhouettes). No reversed draws rasterizes to far everywhere, so the
// reversed test is a no-op then.

layout(set = 0, binding = 0) uniform sampler2DArray win_tex;
layout(set = 0, binding = 1) uniform sampler2DArray virt_tex;
layout(set = 0, binding = 2) uniform sampler2DArray rev_tex;

void main()
{
	float vi = float(gl_ViewIndex);
	vec3 buv = vec3(uv, vi);
	float w = texture(win_tex, buv).r;
	float v = texture(virt_tex, buv).r;
	float r = texture(rev_tex, buv).r;
	float w_present = 1.0 - step(0.9999, w);
	float v_present = 1.0 - step(0.9999, v);
	float r_present = 1.0 - step(0.9999, r);
	float vr_wins = (1.0 - step(w, v)) * v_present * w_present;
	float rev_wins = (1.0 - step(w, r)) * r_present * w_present;
	out_color = vec4(w_present * (1.0 - vr_wins) * (1.0 - rev_wins));
}
