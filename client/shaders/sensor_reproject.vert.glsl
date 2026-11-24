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

// 6DOF sensor-depth forward warp (debug viz): each vertex is one native
// sensor texel; unprojects to world through the depth camera at capture
// time, reprojects to the display eye at predicted time. Full translation
// included (rotation-only H + lateral T approximations need not apply).
// Matrices live here in the vertex stage (proven path); the fragment
// stays scalar + taps (Adreno rule: no mat3/mat4 in fragment).

layout(location = 0) in vec4 in_data; // xy: corner UV (texture space), zw: owning texel id

// Texture space: v = 1 is the frustum top (matches depth_viz's 1-v flip:
// the framebuffer top samples texture row 1). slope_y = b + v*(t-b).
layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 1) uniform WarpUbo
{
	mat4 disp_vp; // display clip-from-world at predicted time (explicit finite-far)
	mat4 sens_mount; // world-from-sensor-eye at capture time (D_cap, full 6DOF)
	vec4 sens_tan; // tan(angleLeft, angleRight, angleDown, angleUp)
	vec2 sens_nf; // sensor near, far (meters)
}
w;

layout(location = 0) out vec2 v_suv;
layout(location = 1) out float v_depth;

layout(push_constant) uniform pc
{
	int eye;
}
push;

// Same D convention as depth_viz.frag: GL-style, infinite-far branch.
float linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

void main()
{
	// Deterministic vertex fetch: the owning texel id is exact (small
	// integers), so texelFetch never rounds — a NEAREST tap exactly on
	// a corner UV would be driver-defined (4-way boundary jitter).
	ivec2 t = ivec2(int(in_data.z), int(in_data.w));
	float d = texelFetch(sensor_tex, ivec3(t, push.eye), 0).r;
	v_suv = in_data.xy;
	if (d >= 0.9999)
	{
		// No-return texel: collapse behind the far plane (clipped away,
		// the hole stays cleared = far). No hole-filling by design.
		v_depth = 0.0;
		gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
		return;
	}
	float s = linearize_depth(d, w.sens_nf.x, w.sens_nf.y);
	v_depth = s;
	vec3 dir = vec3(w.sens_tan.x + v_suv.x * (w.sens_tan.y - w.sens_tan.x),
	                w.sens_tan.z + v_suv.y * (w.sens_tan.w - w.sens_tan.z),
	                -1.0);
	vec4 world = w.sens_mount * vec4(dir * s, 1.0);
	gl_Position = w.disp_vp * world;
}
