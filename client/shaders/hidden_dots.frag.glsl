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
// the fragment's facing axis pair (from the object-space face normal,
// derived from screen derivatives of vQ — no new varyings) has its
// nearest lattice center projected to screen space, and the fragment
// measures pixel distance to it, so every dot is a true circle facing
// the viewer however the face is angled. Only the facing pair is drawn:
// the other pairs track the fragment along their free axis, which draws
// full-length hairlines through every dot instead of dots. Centers stay
// glued to the surface lattice (world-anchored, no swim), radius is a
// fixed world size via the view-depth scale (perspective-correct
// spheres). Grazing faces show sparser rows instead of smears (accepted
// density variation). No depth; non-dot pixels discard.
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
	// Facing pair only: object-space face normal from screen derivatives
	// of vQ (pitch is a uniform scale, so it preserves direction; the
	// derivatives run here in uniform control flow, before the branch).
	// The facing plane's pair snaps both in-plane axes to the lattice
	// while the off-plane coordinate is constant on the face — a true
	// fixed dot grid. The other pairs would track the fragment along
	// their free axis (center shares the fragment's screen row/column),
	// drawing full-length hairlines; min() over all three unions those
	// lines with the dots, which is the smear this replaces.
	vec3 n = normalize(cross(dFdx(vQ), dFdy(vQ)));
	vec3 an = abs(n);
	float d;
	if (an.x > an.y && an.x > an.z)
		d = pair_dist(vec3(vQ.x, pair_center(vQ.yz)) * pitch); // YZ plane
	else if (an.y > an.z)
		d = pair_dist(vec3(pair_center(vQ.xz).x, vQ.y, pair_center(vQ.xz).y) * pitch); // XZ plane
	else
		d = pair_dist(vec3(pair_center(vQ.xy), vQ.z) * pitch); // XY plane
	// World radius 0.15 cells (diameter ~30% of pitch), converted with
	// the view-depth scale; 1px AA band (fades tiny distant dots
	// gracefully instead of shimmering). Radii far below ~0.08 cells go
	// subpixel at room range: no dot rasterizes and only exact line
	// cores survive, so keep this well above that floor.
	float px_per_m = (pc.view_params.y * 0.5) * abs(pc.view_params.z) / max(vViewZ, 1e-3);
	float r_px = 0.15 * pitch * px_per_m;
	float dot_ = 1.0 - smoothstep(max(r_px - 1.0, 0.0), r_px, d);
	if (dot_ <= 0.001)
		discard;
	out_color = vec4(pc.color_pitch.rgb, dot_);
}
