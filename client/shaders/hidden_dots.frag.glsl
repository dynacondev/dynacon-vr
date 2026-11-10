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

// Fixed world-size dot grid: the dot function runs on all three axis
// pairs and takes the minimum, so every face is covered with no normals,
// no axis selection, no derivatives. Radius is cell-relative (0.18), so
// dots scale with the pitch and stay perspective-correct for free.
// Non-dot pixels discard (no depth to punch holes in — pure bandwidth
// win, ~90% of the silhouette skips blending).
layout(location = 0) in vec3 vObj;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
}
pc;

void main()
{
	vec3 q = vObj / max(pc.color_pitch.w, 1e-4);
	vec3 d2 = vec3(
	        length(fract(q.xy - 0.5) - 0.5),
	        length(fract(q.xz - 0.5) - 0.5),
	        length(fract(q.yz - 0.5) - 0.5));
	float m = min(d2.x, min(d2.y, d2.z));
	float dot_ = 1.0 - smoothstep(0.16, 0.20, m);
	if (dot_ <= 0.001)
		discard;
	out_color = vec4(pc.color_pitch.rgb, dot_);
}
