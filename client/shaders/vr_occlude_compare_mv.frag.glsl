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

// VR-over-real occlusion for passthrough windows: eye-space test of the
// virtual depth against the rasterized object depth (both sensor-compatible
// D in [0,1], 1.0 = far/empty). Keeps 1 where the object stays, cuts to 0
// where VR wins. The merge below MINs this into the tier seed (LOAD,
// preserves silhouettes + reversed cuts), so the cut feathers exactly like
// a silhouette edge.
//
// Both canvases live in display-eye UV space (per-eye depth resolution,
// 2 layers), so no sensor reprojection is needed: the same eye UV taps
// both. Binary virt < object test, evaluated 2x2 per texel (quarter-texel
// eye-UV offsets, averaged) for true geometric coverage — the same AA
// rationale as volume_compare_mv.frag: per-texel softening can only fade
// whole texels, only a finer evaluation rate moves contour vertices between
// texels and de-staircases diagonals. Both taps are LINEAR-filtered (host
// sampler), so the verdict slides sub-texel.
//
// Guards: object empty (far clear 1.0) never cuts — the seed is already 0
// outside the silhouette, and far-plane geometry (~0.99) is kept. Virtual
// far (1.0 = far/unknown/sky/hole from volume_virt_mv.frag) never cuts.
// No epsilon on the comparison itself: the 4-level average IS the
// antialiasing.

layout(set = 0, binding = 0) uniform sampler2DArray object_tex;
layout(set = 0, binding = 1) uniform sampler2DArray virt_tex;

void main()
{
	float vi = float(gl_ViewIndex);
	const float q = 0.25 / float(textureSize(object_tex, 0).x);
	float keep = 0.0;
	for (int i = 0; i < 4; ++i)
	{
		vec2 uvi = uv + (vec2(float(i & 1), float((i >> 1) & 1)) - 0.5) * (2.0 * q);
		float v = texture(object_tex, vec3(uvi, vi)).r;
		float w = texture(virt_tex, vec3(uvi, vi)).r;
		float obj_present = 1.0 - step(0.9999, v);
		float virt_present = 1.0 - step(0.9999, w);
		float vr_wins = (1.0 - step(v, w)) * obj_present * virt_present;
		keep += 1.0 - vr_wins;
	}
	out_color = vec4(keep * 0.25);
}
