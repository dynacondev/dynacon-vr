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

#extension GL_GOOGLE_include_directive : require

#include "depth_viz_common.glsl.inc"

// Forward splat vertex shader: indexed triangle mesh over the sensor
// texel grid (vertices at texel centers, two triangles per quad with TL
// listed first in both). Each vertex unprojects its own texel with the
// planar convention, rigid-transforms capture->display and projects with
// a finite display frustum. Every quad emits its flat corner-max display
// depth from the TL vertex (provoking vertex for both triangles): the max
// is background by construction on a depth step, so disocclusions read as
// background instead of rubber-sheet stretches, and approximately the
// surface elsewhere. (A sub-texel foreground wire sharing its quad with
// background reads as background: the max erases it.) Quads with no
// valid in-front corner (no-data/behind-camera only) emit <= 0 and the
// fragment stage discards them to honest FAR holes. A per-texel flow
// shift (display-eye head velocity x extra ms from the UBO) moves
// content forward along its own image velocity before projection:
// translation flow ~ v/Z per texel, rotation flow uniform, zero along
// the rotation axis. Shared vertices compute bit-identical positions,
// so no T-cracks, and the hardware depth test resolves overlaps nearest-wins. texelFetch with integer
// bounds keeps the Y-flip exact and sidesteps vertex-stage sampler
// questions.
//
// The push block below rides the splat pipeline layout's vertex range.
// Plain scalar math only (no mat3/ternary on this driver path).

layout(set = 0, binding = 0) uniform sampler2DArray depth_tex;

layout(set = 0, binding = 1) uniform splat_params
{
	vec4 rot_rows[6]; // per eye (push.eye * 3): display->capture rotation rows
	vec4 sens_tan[2]; // per eye sensor frustum tangents: L, R, U, D
	vec4 disp_tan[2]; // per eye display (target) frustum tangents: L, R, U, D
	vec2 near_far;
	float opacity;
	vec4 trans[2]; // per eye: capture->display translation in display-eye space
	vec4 flow_lin[2]; // per eye: translational flow displacement, display-eye frame
	vec4 flow_ang[2]; // per eye: angular flow displacement (axis*angle), display-eye frame
} u;

layout(push_constant) uniform pc
{
	int eye;
	int sens_w;
	int sens_h;
} push;

layout(location = 0) flat out float v_bg;

// Capture->display rotation rows: transpose of the stored rows.
vec3 c2d_mul(int e, vec3 Xc)
{
	vec3 r;
	r.x = dot(vec3(u.rot_rows[e * 3 + 0].x, u.rot_rows[e * 3 + 1].x, u.rot_rows[e * 3 + 2].x), Xc);
	r.y = dot(vec3(u.rot_rows[e * 3 + 0].y, u.rot_rows[e * 3 + 1].y, u.rot_rows[e * 3 + 2].y), Xc);
	r.z = dot(vec3(u.rot_rows[e * 3 + 0].z, u.rot_rows[e * 3 + 1].z, u.rot_rows[e * 3 + 2].z), Xc);
	return r;
}

void main()
{
	int e = push.eye;
	int tx = gl_VertexIndex % push.sens_w;
	int ty = gl_VertexIndex / push.sens_w;
	int tx1 = min(tx + 1, push.sens_w - 1);
	int ty1 = min(ty + 1, push.sens_h - 1);
	vec4 st = u.sens_tan[e];
	float n = u.near_far.x;
	float f = u.near_far.y;

	// Quad depths (TL = own texel). Integer Y-flip: framebuffer top row
	// (ty = 0) is texture row H-1 (same convention as depth_viz.frag).
	// Edge quads clamp (tx1/ty1), replicating the edge texel (nearest).
	float ds[4] = float[4](texelFetch(depth_tex, ivec3(tx, push.sens_h - 1 - ty, e), 0).r,
	                       texelFetch(depth_tex, ivec3(tx1, push.sens_h - 1 - ty, e), 0).r,
	                       texelFetch(depth_tex, ivec3(tx, push.sens_h - 1 - ty1, e), 0).r,
	                       texelFetch(depth_tex, ivec3(tx1, push.sens_h - 1 - ty1, e), 0).r);

	// Quad corner positions (texel centers).
	float xs[4];
	float ys[4];
	xs[0] = st.x + (st.y - st.x) * (float(tx) + 0.5) / float(push.sens_w);
	ys[0] = st.z + (st.w - st.z) * (float(ty) + 0.5) / float(push.sens_h);
	xs[1] = st.x + (st.y - st.x) * (float(tx1) + 0.5) / float(push.sens_w);
	ys[1] = ys[0];
	xs[2] = xs[0];
	ys[2] = st.z + (st.w - st.z) * (float(ty1) + 0.5) / float(push.sens_h);
	xs[3] = xs[1];
	ys[3] = ys[2];

	// No per-quad decision: every quad with a valid in-front corner emits
	// its flat corner-max display depth. That max is background by
	// construction on a depth step (so disocclusions read as background,
	// never foreground) and approximately the surface elsewhere. Quads
	// whose corners behind the display eye contribute nothing; quads
	// with no valid in-front corner at all (no-data/behind-camera only)
	// emit <= 0 and the fragment stage discards them to honest FAR
	// holes. Positions always stay valid.
	float zbmax = 0.0; // display-space corner max, for the flat background
	for (int k = 0; k < 4; ++k)
	{
		// Far-clamped so sky pixels can't produce NaN steps.
		if (ds[k] < 0.9999)
		{
			float z = min(viz_linearize_depth(ds[k], n, f), 20.0);
			vec3 Xd = c2d_mul(e, vec3(xs[k], ys[k], -1.0) * z) + u.trans[e].xyz;
			// Per-texel flow correction: shift content forward along
			// its own display-eye velocity (translation flow ~ v/Z per
			// texel, rotation flow uniform over the image, zero along
			// the rotation axis). Exact identity with a zero flow displacement.
			// (UBO holds head displacement; the minus applies the
			// content correction.)
			Xd = Xd - (u.flow_lin[e].xyz + cross(u.flow_ang[e].xyz, Xd));
			if (Xd.z < -1e-6)
				zbmax = max(zbmax, -Xd.z);
		}
	}
	v_bg = zbmax;

	// Own texel reprojected. Shared vertices compute bit-identical
	// positions, so no T-cracks; the hardware depth test resolves
	// overlaps nearest-wins. The own depth is the offset-0 tap above.
	float ox = st.x + (st.y - st.x) * (float(tx) + 0.5) / float(push.sens_w);
	float oy = st.z + (st.w - st.z) * (float(ty) + 0.5) / float(push.sens_h);
	float z = min(viz_linearize_depth(ds[0], n, f), 20.0);
	vec3 Xd0 = c2d_mul(e, vec3(ox, oy, -1.0) * z) + u.trans[e].xyz;
	Xd0 = Xd0 - (u.flow_lin[e].xyz + cross(u.flow_ang[e].xyz, Xd0));
	float d = max(-Xd0.z, 1e-6);
	vec4 dt = u.disp_tan[e];
	float du = (Xd0.x / (-Xd0.z) - dt.x) / (dt.y - dt.x);
	float dv = (dt.z - Xd0.y / (-Xd0.z)) / (dt.z - dt.w);
	float near = u.near_far.x;
	// Projection depth range (ordering only for the D16 depth test;
	// content clamps to 20m at emission above, well inside this).
	float far = 50.0;
	gl_Position = vec4((2.0 * du - 1.0) * d, (2.0 * dv - 1.0) * d, far * (d - near) / (far - near), d);
}
