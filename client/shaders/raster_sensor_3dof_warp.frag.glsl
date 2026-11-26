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

// Raster-vs-sensor with the custom warp sensor tap
// (sensor_3dof_warp.frag twin): the layer is tapped directly by output
// UV (no pose involved: its content is server-rasterized), the runtime
// D16 sensor is gathered capture->predicted with fixed-point 6DOF
// refinement when tune.z >= 0.5 (3DOF seed only otherwise). The 6DOF seed
// is min-dilated over a ±4 sensor-texel cross (≈, max-pitch axis);
// acceptance is convergence-only.
// Compare ramp, no-data treatment and linearization match
// raster_sensor_viz.frag: failed warp pixels read as sensor no-return
// (opaque black).
//
// Plain scalar math only (no mat3/ternary on this driver path).

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray layer_tex;

layout(set = 0, binding = 1) uniform sampler2DArray sensor_tex;

layout(set = 0, binding = 2) uniform warp_params
{
	vec4 rot_rows[6]; // per eye (push.eye * 3): display->capture rotation rows
	vec4 sens_tan[2]; // per eye sensor frustum tangents: L, R, U, D
	vec4 disp_tan[2]; // per eye display (target) frustum tangents: L, R, U, D
	vec2 near_far;
	float opacity;
	float pad;
	vec4 trans[2]; // per eye: display-eye origin in capture-eye space
	vec4 misc[2];  // per eye: x = refine step cap, y = display texel pitch
	vec4 tune;     // x = accept gate (texels), y = reserved,
	               // z = refinement enable (0 = 3DOF seed, 2 = iterate), w reserved
} u;

layout(push_constant) uniform pc
{
	int eye;
} push;

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

// Gather + fixed-point refinement. Returns display-space depth;
// ok=false means FAR/no-data. tune.z < 0.5 skips refinement.
float warp_depth(int e, float xt, float yt, out bool ok)
{
	ok = false;
	vec3 d = vec3(xt, yt, -1.0);
	vec3 c;
	c.x = dot(u.rot_rows[e * 3 + 0].xyz, d);
	c.y = dot(u.rot_rows[e * 3 + 1].xyz, d);
	c.z = dot(u.rot_rows[e * 3 + 2].xyz, d);

	if (c.z >= -1e-6)
		return 0.0;

	vec4 st = u.sens_tan[e];
	float xs = c.x / (-c.z);
	float ys = c.y / (-c.z);
	float su = (xs - st.x) / (st.y - st.x);
	float sv = (st.z - ys) / (st.z - st.w);
	// NaN-safe frustum guard (degenerate frustum makes 0/0, which
	// passes ordered comparisons and would sample garbage).
	if (!(su >= 0.0 && su <= 1.0 && sv >= 0.0 && sv <= 1.0))
		return 0.0;

	float s = texture(sensor_tex, vec3(su, 1.0 - sv, float(e))).r;
	if (s >= 0.9999)
		return 0.0;
	float z = linearize_depth(s, u.near_far.x, u.near_far.y);
	if (u.tune.z < 0.5)
	{
		ok = true;
		return z;
	}
	// 6DOF: min-dilated seed. Start the walk from the near surface
	// when the neighborhood straddles an edge: background depths are
	// translation-insensitive fixed points, so a background seed at a
	// truth-foreground pixel converges wrong with err ~0 and no
	// straddle (guards blind by construction). Relocation is
	// load-bearing (value alone still traps). Arms individually
	// guarded: the sampler is clamp-to-edge. Ties keep the center.
	// Iterate in capture space (far clamped so sky pixels can't
	// produce NaN steps), accept in display space.
	float arm = u.misc[e].x * 0.5;
	float bx = xs;
	float by = ys;
	float bz = z;
	{
		float ax = xs + arm;
		float au = (ax - st.x) / (st.y - st.x);
		if (au >= 0.0 && au <= 1.0)
		{
			float sa = texture(sensor_tex, vec3(au, 1.0 - sv, float(e))).r;
			if (sa < 0.9999)
			{
				float az = linearize_depth(sa, u.near_far.x, u.near_far.y);
				if (az < bz)
				{
					bx = ax;
					bz = az;
				}
			}
		}
	}
	{
		float ax = xs - arm;
		float au = (ax - st.x) / (st.y - st.x);
		if (au >= 0.0 && au <= 1.0)
		{
			float sa = texture(sensor_tex, vec3(au, 1.0 - sv, float(e))).r;
			if (sa < 0.9999)
			{
				float az = linearize_depth(sa, u.near_far.x, u.near_far.y);
				if (az < bz)
				{
					bx = ax;
					bz = az;
				}
			}
		}
	}
	{
		float ay = ys + arm;
		float av = (st.z - ay) / (st.z - st.w);
		if (av >= 0.0 && av <= 1.0)
		{
			float sa = texture(sensor_tex, vec3(su, 1.0 - av, float(e))).r;
			if (sa < 0.9999)
			{
				float az = linearize_depth(sa, u.near_far.x, u.near_far.y);
				if (az < bz)
				{
					by = ay;
					bz = az;
				}
			}
		}
	}
	{
		float ay = ys - arm;
		float av = (st.z - ay) / (st.z - st.w);
		if (av >= 0.0 && av <= 1.0)
		{
			float sa = texture(sensor_tex, vec3(su, 1.0 - av, float(e))).r;
			if (sa < 0.9999)
			{
				float az = linearize_depth(sa, u.near_far.x, u.near_far.y);
				if (az < bz)
				{
					by = ay;
					bz = az;
				}
			}
		}
	}
	xs = bx;
	ys = by;
	float zc = min(bz, 20.0);
	vec3 Xc = vec3(xs, ys, -1.0) * zc;
	float stp = u.misc[e].x;
	for (int k = 0; k < 2; ++k)
	{
		vec3 Xd = c2d_mul(e, Xc) + u.trans[e].xyz;
		if (Xd.z >= -1e-6)
			return 0.0;
		float exx = xt - Xd.x / (-Xd.z);
		float eyy = yt - Xd.y / (-Xd.z);
		xs += clamp(exx, -stp, stp);
		ys += clamp(eyy, -stp, stp);
		su = (xs - st.x) / (st.y - st.x);
		sv = (st.z - ys) / (st.z - st.w);
		if (!(su >= 0.0 && su <= 1.0 && sv >= 0.0 && sv <= 1.0))
			return 0.0;
		s = texture(sensor_tex, vec3(su, 1.0 - sv, float(e))).r;
		if (s >= 0.9999)
			return 0.0;
		z = min(linearize_depth(s, u.near_far.x, u.near_far.y), 20.0);
		Xc = vec3(xs, ys, -1.0) * z;
	}
	vec3 Xd = c2d_mul(e, Xc) + u.trans[e].xyz;
	if (Xd.z >= -1e-6)
		return 0.0;
	float err = length(vec2(xt - Xd.x / (-Xd.z), yt - Xd.y / (-Xd.z))) / max(u.misc[e].y, 1e-9);
	if (err > u.tune.x)
		return 0.0;
	ok = true;
	return max(-Xd.z, 1e-6);
}

void main()
{
	int e = push.eye;
	float l = texture(layer_tex, vec3(v_uv, float(e))).r;

	// Warped sensor tap (capture->predicted, 3DOF or 6DOF by tune.z).
	vec4 dt = u.disp_tan[e];
	float xt = mix(dt.x, dt.y, v_uv.x);
	float yt = mix(dt.z, dt.w, v_uv.y);
	bool ok;
	float dist = warp_depth(e, xt, yt, ok);

	// Same compare as raster_sensor_viz.frag: failed warp pixels read
	// as sensor no-return (opaque black, FAR semantics).
	float absent = step(65000.0, l);
	float noval = absent;
	float dc = 10.0;
	if (ok)
	{
		dc = min(dist, 10.0);
	}
	else
	{
		noval = 1.0;
	}

	float lc = min(l, 10.0);
	float t = clamp(abs(lc - dc) * 0.5, 0.0, 1.0);
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);
	col = mix(col, vec3(0.0), noval);

	out_color = vec4(col, u.opacity);
}
