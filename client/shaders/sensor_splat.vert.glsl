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
// texel grid (vertices at texel centers, static index buffer with two
// triangles per quad, TL listed first in both). Each vertex unprojects
// its own texel with the planar convention, rigid-transforms
// capture->display and projects with a finite display frustum. Quads
// straddling a depth step emit a background-stretch verdict (flat
// quad-max depth from the TL corner, which is the provoking vertex for
// both triangles) so disocclusions read as background instead of
// rubber-sheet stretches or holes; no-data and behind-camera quads are
// discarded (honest FAR holes). Kept quads interpolate smoothly (shared
// vertices compute bit-identical positions, so no T-cracks) and the
// hardware depth test resolves overlaps nearest-wins. texelFetch with
// integer bounds keeps the Y-flip exact and sidesteps vertex-stage
// sampler questions.
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
	int sens_w;
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
	int tx = gl_VertexIndex % push.sens_w;
	int ty = gl_VertexIndex / push.sens_w;
	int tx1 = min(tx + 1, push.sens_w - 1);
	int ty1 = min(ty + 1, push.sens_h - 1);
	vec4 st = u.sens_tan[e];
	float n = u.near_far.x;
	float f = u.near_far.y;

	// Quad depths (TL = own texel). Integer Y-flip: framebuffer top row
	// (ty = 0) is texture row H-1 (same convention as depth_viz.frag).
	float d00 = texelFetch(depth_tex, ivec3(tx, push.sens_h - 1 - ty, e), 0).r;
	float d10 = texelFetch(depth_tex, ivec3(tx1, push.sens_h - 1 - ty, e), 0).r;
	float d01 = texelFetch(depth_tex, ivec3(tx, push.sens_h - 1 - ty1, e), 0).r;
	float d11 = texelFetch(depth_tex, ivec3(tx1, push.sens_h - 1 - ty1, e), 0).r;

	// Quad verdict: 0 = smooth interpolate, 1 = discard (no-data or
	// behind-camera: honest FAR hole), 2 = background stretch (depth
	// step: emit the flat quad-max so background covers the
	// disocclusion, never foreground). Positions always stay valid;
	// the fragment stage applies the verdict.
	bool unusable = (d00 >= 0.9999) || (d10 >= 0.9999) || (d01 >= 0.9999) || (d11 >= 0.9999);
	float zmin = 1e30;
	float zmax = 0.0;
	float zbmax = 0.0; // display-space quad max, for background stretch
	vec3 Xd[4];
	bool behind = false;
	if (!unusable)
	{
		float ds[4] = float[4](d00, d10, d01, d11);
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
		for (int k = 0; k < 4; ++k)
		{
			// Far-clamped so sky pixels can't produce NaN steps or blow
			// up the stretch ratio.
			float z = min(linearize_depth(ds[k], n, f), 20.0);
			zmin = min(zmin, z);
			zmax = max(zmax, z);
			Xd[k] = c2d_mul(e, vec3(xs[k], ys[k], -1.0) * z) + u.trans[e].xyz;
			if (Xd[k].z >= -1e-6)
				behind = true;
			else
				zbmax = max(zbmax, -Xd[k].z);
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
	// never reach the fragment shader).
	float xs = st.x + (st.y - st.x) * (float(tx) + 0.5) / float(push.sens_w);
	float ys = st.z + (st.w - st.z) * (float(ty) + 0.5) / float(push.sens_h);
	float z = min(linearize_depth(d00, n, f), 20.0);
	vec3 Xd0 = c2d_mul(e, vec3(xs, ys, -1.0) * z) + u.trans[e].xyz;
	float d = max(-Xd0.z, 1e-6);
	vec4 dt = u.disp_tan[e];
	float du = (Xd0.x / (-Xd0.z) - dt.x) / (dt.y - dt.x);
	float dv = (dt.z - Xd0.y / (-Xd0.z)) / (dt.z - dt.w);
	float near = u.near_far.x;
	float far = 50.0;
	gl_Position = vec4((2.0 * du - 1.0) * d, (2.0 * dv - 1.0) * d, far * (d - near) / (far - near), d);
	v_depth = d;
}
