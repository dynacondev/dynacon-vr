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

// Forward splat vertex shader: one point per sensor texel, unprojected
// with the planar convention, rigid-transformed capture->display and
// projected with a finite display frustum. Hardware depth test resolves
// visibility (nearest wins, order-independent); true disocclusions get
// no splat and read as FAR clear. texelFetch with integer bounds keeps
// the Y-flip exact and sidesteps vertex-stage sampler questions.
// No-data and behind-camera points collapse to a clipped position.
//
// Plain scalar math only (no mat3/ternary on this driver path). The 12B
// push block below rides the splat pipeline layout's vertex range; the
// pass-2 fragment shaders declare the standard 4-byte eye push.

layout(set = 0, binding = 0) uniform sampler2DArray depth_tex;

layout(set = 0, binding = 1) uniform splat_params
{
	vec4 rot_rows[6]; // per eye (push.eye * 3): display->capture rotation rows
	vec4 sens_tan[2]; // per eye sensor frustum tangents: L, R, U, D
	vec4 disp_tan[2]; // per eye display (target) frustum tangents: L, R, U, D
	vec2 near_far;
	float opacity;
	float pad;
	vec4 trans[2]; // per eye: capture->display translation in display-eye space
} u;

layout(push_constant) uniform pc
{
	int eye;
	int sens_w;
	int sens_h;
} push;

layout(location = 0) out float v_depth;

float linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		// Infinite far plane. D=0 -> near, D=1 -> far.
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

void main()
{
	int e = push.eye;
	int tx = gl_VertexIndex % push.sens_w;
	int ty = gl_VertexIndex / push.sens_w;

	// Integer Y-flip: framebuffer top row (ty = 0) is texture row H-1
	// (same convention as the depth_viz.frag 1-v tap).
	float d_norm = texelFetch(depth_tex, ivec3(tx, push.sens_h - 1 - ty, e), 0).r;
	if (d_norm >= 0.9999)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		v_depth = 0.0;
		gl_PointSize = 1.0;
		return;
	}

	// Texel centers in capture-tangent space.
	float su = (float(tx) + 0.5) / float(push.sens_w);
	float sv = (float(ty) + 0.5) / float(push.sens_h);
	vec4 st = u.sens_tan[e];
	float xs = mix(st.x, st.y, su);
	float ys = mix(st.z, st.w, sv);

	float z = linearize_depth(d_norm, u.near_far.x, u.near_far.y);
	vec3 Xc = vec3(xs, ys, -1.0) * z;

	// Capture->display: transpose of the stored rows, plus translation.
	vec3 Xd;
	Xd.x = dot(vec3(u.rot_rows[e * 3 + 0].x, u.rot_rows[e * 3 + 1].x, u.rot_rows[e * 3 + 2].x), Xc) + u.trans[e].x;
	Xd.y = dot(vec3(u.rot_rows[e * 3 + 0].y, u.rot_rows[e * 3 + 1].y, u.rot_rows[e * 3 + 2].y), Xc) + u.trans[e].y;
	Xd.z = dot(vec3(u.rot_rows[e * 3 + 0].z, u.rot_rows[e * 3 + 1].z, u.rot_rows[e * 3 + 2].z), Xc) + u.trans[e].z;
	if (Xd.z >= -1e-6)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		v_depth = 0.0;
		gl_PointSize = 1.0;
		return;
	}

	// Finite display projection. Framebuffer top (sv = 0) is NDC y = -1,
	// matching the fullscreen-triangle convention (v_uv.y = 0 -> y = -1).
	vec4 dt = u.disp_tan[e];
	float xd = Xd.x / (-Xd.z);
	float yd = Xd.y / (-Xd.z);
	float du = (xd - dt.x) / (dt.y - dt.x);
	float dv = (dt.z - yd) / (dt.z - dt.w);
	float d = -Xd.z;
	float n = u.near_far.x;
	float f = 50.0;
	gl_Position = vec4((2.0 * du - 1.0) * d, (2.0 * dv - 1.0) * d, f * (d - n) / (f - n), d);
	v_depth = d;
	// Depth-adaptive footprint: a texel's linear footprint scales ~zc/zd
	// (area ~square; axial-depth proxy — exact for forward motion,
	// approximate under rotation/obliquity), so the point covers the
	// local magnification instead of a fixed size. Clamped: residual
	// pinholes past 6px and up to ~3px foreground fattening at max grow
	// are inherent. Adreno 740 supports far larger points; worst case on
	// a limited device is graceful pinholes.
	float grow = clamp(z / d, 0.25, 4.0);
	gl_PointSize = clamp(3.0 * grow, 1.0, 6.0);
}
