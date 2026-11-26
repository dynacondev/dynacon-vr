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

// Custom 3DOF rotation-only reprojection of the runtime D16 environment
// depth into the predicted display view: the CPU copy of what the runtime
// timewarp does with a capture-posed layer. Per output pixel the display
// ray (target fov) is rotated into capture-eye space (R = Qc^-1 * Qd,
// uploaded per eye) and projected into the sensor frustum; the depth
// value itself passes through approximately unchanged (exact for
// radial distance under pure rotation about the eye; the runtime D16
// convention and the identity headset<->sensor rotational extrinsic
// this assumes are debug-viz stakes, see the record site). Out-of-frustum and behind-camera
// pixels read fully transparent (no data there), matching the runtime's
// treatment of unwrapped areas.
//
// False color, linearization and the 1-v texture flip are byte-identical
// to depth_viz.frag: at zero head motion with coincident frusta and
// extrinsics this renders the same image as the direct tap.
//
// Plain scalar math only (no mat3/ternary on this driver path): the
// rotation rides 3 vec4 UBO rows per eye, applied as dot products.

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray depth_tex;

layout(set = 0, binding = 1) uniform warp_params
{
	vec4 rot_rows[6]; // per eye (push.eye * 3): rotation rows, xyz used
	vec4 sens_tan[2]; // per eye sensor frustum tangents: L, R, U, D
	vec4 disp_tan[2]; // per eye display (target) frustum tangents: L, R, U, D
	vec2 near_far;
	float opacity;
	float pad;
} u;

layout(push_constant) uniform pc
{
	int eye;
} push;

// Same math as depth_viz.frag: D in [0,1], infinite farZ branch.
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
	vec4 dt = u.disp_tan[e];

	// Display ray (unnormalized: ratios cancel in the projection).
	float xt = mix(dt.x, dt.y, v_uv.x);
	float yt = mix(dt.z, dt.w, v_uv.y);
	vec3 d = vec3(xt, yt, -1.0);

	// Into capture-eye space (scalar dot products, never a mat3).
	vec3 c;
	c.x = dot(u.rot_rows[e * 3 + 0].xyz, d);
	c.y = dot(u.rot_rows[e * 3 + 1].xyz, d);
	c.z = dot(u.rot_rows[e * 3 + 2].xyz, d);

	// Behind or at the capture eye: no data.
	if (c.z >= -1e-6)
	{
		out_color = vec4(0.0, 0.0, 0.0, 0.0);
		return;
	}

	vec4 st = u.sens_tan[e];
	float xs = c.x / (-c.z);
	float ys = c.y / (-c.z);
	float su = (xs - st.x) / (st.y - st.x);
	float sv = (st.z - ys) / (st.z - st.w);
	// NaN-safe frustum guard (a degenerate frustum would make 0/0,
	// which passes ordered comparisons and samples garbage).
	if (!(su >= 0.0 && su <= 1.0 && sv >= 0.0 && sv <= 1.0))
	{
		out_color = vec4(0.0, 0.0, 0.0, 0.0);
		return;
	}

	float d_norm = texture(depth_tex, vec3(su, 1.0 - sv, float(e))).r;

	float n = u.near_far.x;
	float f = u.near_far.y;
	float dist = linearize_depth(d_norm, n, f);
	// Infinite far (unbounded or classic): normalize against the
	// reconstructed distance itself (t = 1 - near/d).
	bool infinite = !(f > n) || f > 1e9;
	float denom = infinite ? max(dist, 1e-6) : (f - n);
	float t = clamp((dist - n) / denom, 0.0, 1.0);
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);

	out_color = vec4(col, u.opacity);
}
