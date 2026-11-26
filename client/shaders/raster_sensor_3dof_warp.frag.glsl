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

// Raster-vs-sensor with a custom-3DOF-warped sensor tap: the owned R16F
// common depth layer is tapped directly by output UV (no pose involved:
// its content is server-rasterized, and the residual sync-vs-predicted
// offset is owned by the future alpha-layer work), while the runtime D16
// sensor is reprojected capture->predicted with the same rotation-only
// warp as sensor_3dof_warp.frag. Compare ramp, no-data treatment and
// linearization are byte-identical to raster_sensor_viz.frag.
//
// Plain scalar math only (no mat3/ternary on this driver path).

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray layer_tex;

layout(set = 0, binding = 1) uniform sampler2DArray sensor_tex;

layout(set = 0, binding = 2) uniform warp_params
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
	float l = texture(layer_tex, vec3(v_uv, float(e))).r;

	// Rotation-only sensor warp (sensor_3dof_warp.frag twin).
	vec4 dt = u.disp_tan[e];
	float xt = mix(dt.x, dt.y, v_uv.x);
	float yt = mix(dt.z, dt.w, v_uv.y);
	vec3 d = vec3(xt, yt, -1.0);
	vec3 c;
	c.x = dot(u.rot_rows[e * 3 + 0].xyz, d);
	c.y = dot(u.rot_rows[e * 3 + 1].xyz, d);
	c.z = dot(u.rot_rows[e * 3 + 2].xyz, d);

	float ds;
	bool sensor_valid = c.z < -1e-6;
	if (sensor_valid)
	{
		vec4 st = u.sens_tan[e];
		float xs = c.x / (-c.z);
		float ys = c.y / (-c.z);
		float su = (xs - st.x) / (st.y - st.x);
		float sv = (st.z - ys) / (st.z - st.w);
		// NaN-safe frustum guard (degenerate frustum makes 0/0, which
		// passes ordered comparisons and would sample garbage).
		sensor_valid = su >= 0.0 && su <= 1.0 && sv >= 0.0 && sv <= 1.0;
		if (sensor_valid)
			ds = texture(sensor_tex, vec3(su, 1.0 - sv, float(e))).r;
		else
			ds = 1.0;
	}
	else
	{
		ds = 1.0;
	}

	// Same compare as raster_sensor_viz.frag: warped-out pixels read
	// as sensor no-return (opaque black).
	float absent = step(65000.0, l);
	float no_return = sensor_valid ? step(0.9999, ds) : 1.0;
	float noval = max(absent, no_return);

	float dist = linearize_depth(ds, u.near_far.x, u.near_far.y);
	float lc = min(l, 10.0);
	float dc = min(dist, 10.0);
	float t = clamp(abs(lc - dc) * 0.5, 0.0, 1.0);
	vec3 col = mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);
	col = mix(col, vec3(0.0), noval);

	out_color = vec4(col, u.opacity);
}
