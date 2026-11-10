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

// Screen-round dots over the silhouette (billboard illusion, no geometry):
// each axis pair's nearest lattice center is projected to screen space and
// the fragment measures pixel distance to it, so every dot is a true
// circle facing the viewer however the face is angled — grazing smear is
// structurally impossible. Centers stay glued to the surface lattice
// (world-anchored, no swim), radius is a fixed world size via the
// view-depth scale (perspective-correct spheres). Grazing faces show sparser
// rows instead of smears (accepted density variation). No normals, no
// derivatives; non-dot pixels discard (no depth to punch holes in).
layout(location = 0) in vec3 vQ;
layout(location = 1) in float vViewZ;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
	vec4 view_params; // x: view W px, y: view H px, z: proj[1][1]
}
pc;

// Nearest lattice center of one axis pair (cells): integer lattice
// (the historic dot convention), symmetric basin [C-0.5, C+0.5).
vec2 pair_center(vec2 q)
{
	return floor(q + 0.5);
}

float pair_dist(vec3 cobj)
{
	vec4 c = pc.mvp * vec4(cobj, 1.0);
	vec2 px = (c.xy / max(c.w, 1e-6) * 0.5 + 0.5) * pc.view_params.xy;
	return length(px - gl_FragCoord.xy);
}

void main()
{
	float pitch = max(pc.color_pitch.w, 1e-4);
	vec3 c0 = vec3(pair_center(vQ.xy), vQ.z) * pitch;
	vec3 c1 = vec3(pair_center(vQ.xz).x, vQ.y, pair_center(vQ.xz).y) * pitch;
	vec3 c2 = vec3(vQ.x, pair_center(vQ.yz)) * pitch;
	float d = min(pair_dist(c0), min(pair_dist(c1), pair_dist(c2)));
	// World radius 0.045 cells, converted with the view-depth scale; 1px
	// AA band (fades tiny distant dots gracefully instead of shimmering).
	float px_per_m = (pc.view_params.y * 0.5) * abs(pc.view_params.z) / max(vViewZ, 1e-3);
	float r_px = 0.045 * pitch * px_per_m;
	float dot_ = 1.0 - smoothstep(max(r_px - 1.0, 0.0), r_px, d);
	if (dot_ <= 0.001)
		discard;
	out_color = vec4(pc.color_pitch.rgb, dot_);
}
