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

// Forward splat vertex shader: indexed triangle mesh over the sensor
// texel CORNERS ((sens_w+1) x (sens_h+1) vertices, one quad per texel
// square, two triangles per quad with TL listed first in both). Each
// vertex unprojects its corner with the planar convention (corner
// positions span the full sensor frustum, no half-texel inset),
// sampling the depth of its BR texel — the texel up-left of the corner
// — clamped at the top/left sensor borders (edge replication, nearest:
// no invented depths). Axis-aligned true edges between differing
// texels therefore coincide with straight mesh edges instead of
// zigzagging around texel centers. Each vertex rigid-transforms
// capture->display and projects with a finite display frustum. The TL
// vertex of each quad (provoking vertex for both triangles) emits the
// quad verdict over its four corners' depths: smooth interpolate when
// clean, discard on no-data/behind-camera (honest FAR holes), flat
// display-space corner-max background stretch on a depth step so
// disocclusions read as background instead of rubber-sheet stretches.
// Kept quads interpolate smoothly (shared vertices compute bit-identical
// positions, so no T-cracks) and the hardware depth test resolves
// overlaps nearest-wins. texelFetch with integer bounds keeps the Y-flip
// exact and sidesteps vertex-stage sampler questions.
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
	float cull_ratio; // max/min depth ratio over a quad before background stretch
	vec4 trans[2]; // per eye: capture->display translation in display-eye space
} u;

layout(push_constant) uniform pc
{
	int eye;
	int sens_w; // sensor TEXEL dims; the corner grid is (sens_w+1) x (sens_h+1)
	int sens_h;
} push;

layout(location = 0) out float v_depth;
layout(location = 1) flat out int v_flag;
layout(location = 2) flat out float v_bgmax;

float linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		// Infinite far plane. D=0 -> near, D=1 -> far.
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

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
	int W = push.sens_w;
	int H = push.sens_h;
	int gw = W + 1; // corner-grid row stride
	int cx = gl_VertexIndex % gw;
	int cy = gl_VertexIndex / gw;
	vec4 st = u.sens_tan[e];
	float n = u.near_far.x;
	float f = u.near_far.y;

	// Four corners of the quad this vertex is TL of: (cx,cy) plus offsets
	// (1,0),(0,1),(1,1). Integer corner positions span the full sensor
	// frustum (cx = 0 is the left tangent, cx = W the right). Each corner
	// samples its BR texel (the texel up-left of it), clamped at the
	// top/left sensor borders (edge replication, nearest). Either order
	// of clamp and Y-flip gives the same result here (the flip maps the
	// ±1-extended range onto itself), but clamp-then-flip keeps the
	// texture row H-1-py on the framebuffer edge row by construction.
	// Note the verdict side effect: clamping also spreads a border
	// texel's depth into every window touching that border, so a no-data
	// top/left edge texel discards up to 4 quads (interior parity is 4
	// before and after; bottom/right are unchanged). Border vertices
	// (cx = W or cy = H) extrapolate one corner past the frustum edge;
	// their verdicts are never read (no quad down-right of them), so
	// finite-but-outside is fine.
	float xs[4];
	float ys[4];
	float ds[4];
	for (int k = 0; k < 4; ++k)
	{
		int dx = 0;
		int dy = 0;
		if (k == 1 || k == 3)
			dx = 1;
		if (k >= 2)
			dy = 1;
		xs[k] = st.x + (st.y - st.x) * float(cx + dx) / float(W);
		ys[k] = st.z + (st.w - st.z) * float(cy + dy) / float(H);
		int px = min(max(cx + dx - 1, 0), W - 1);
		int py = min(max(cy + dy - 1, 0), H - 1);
		// Integer Y-flip: framebuffer top row (py = 0) is texture row
		// H-1 (same convention as depth_viz.frag).
		ds[k] = texelFetch(depth_tex, ivec3(px, H - 1 - py, e), 0).r;
	}

	// Quad verdict over the four corners' depths: 0 = smooth interpolate,
	// 1 = discard (no-data or behind-camera: honest FAR hole),
	// 2 = background stretch (depth step: emit the flat corner-max so
	// background covers the disocclusion, never foreground). Decision
	// logic unchanged from the center grid; only the sampling window
	// moved (up-left-shifted corner window instead of the own-texel
	// neighborhood). Positions always stay valid; the fragment stage
	// applies the verdict.
	bool unusable = false;
	bool behind = false;
	float zmin = 1e30;
	float zmax = 0.0;
	float zbmax = 0.0; // display-space quad max, for background stretch
	for (int k = 0; k < 4; ++k)
	{
		// Far-clamped so sky pixels can't produce NaN steps or blow
		// up the stretch ratio.
		if (ds[k] >= 0.9999)
		{
			unusable = true;
		}
		else
		{
			float z = min(linearize_depth(ds[k], n, f), 20.0);
			zmin = min(zmin, z);
			zmax = max(zmax, z);
			vec3 Xd = c2d_mul(e, vec3(xs[k], ys[k], -1.0) * z) + u.trans[e].xyz;
			if (Xd.z >= -1e-6)
				behind = true;
			else
				zbmax = max(zbmax, -Xd.z);
		}
	}
	if (unusable || behind)
		v_flag = 1;
	else if (zmax / max(zmin, 1e-6) > u.cull_ratio)
		v_flag = 2;
	else
		v_flag = 0;
	// Stretched pixels carry display-space depth (like the smooth path's
	// v_depth), not the capture-space quad max.
	v_bgmax = zbmax;

	// Own corner reprojected (kept triangles interpolate these; stretched
	// triangles ignore it for the flat background, discarded triangles
	// never reach the fragment shader). The own paired depth is the
	// offset-0 corner sample above.
	float ox = st.x + (st.y - st.x) * float(cx) / float(W);
	float oy = st.z + (st.w - st.z) * float(cy) / float(H);
	float z = min(linearize_depth(ds[0], n, f), 20.0);
	vec3 Xd0 = c2d_mul(e, vec3(ox, oy, -1.0) * z) + u.trans[e].xyz;
	float d = max(-Xd0.z, 1e-6);
	vec4 dt = u.disp_tan[e];
	float du = (Xd0.x / (-Xd0.z) - dt.x) / (dt.y - dt.x);
	float dv = (dt.z - Xd0.y / (-Xd0.z)) / (dt.z - dt.w);
	float near = u.near_far.x;
	float far = 50.0;
	gl_Position = vec4((2.0 * du - 1.0) * d, (2.0 * dv - 1.0) * d, far * (d - near) / (far - near), d);
	v_depth = d;
}
