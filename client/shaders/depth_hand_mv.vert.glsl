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

// Tracked-hand splat vertex shader (3d-depth-passthrough): CPU-skinned
// hand verts (world space, video-frame time) projected into the
// sync-eye grid with the same mapping as sensor_splat.vert (du/dv
// against the target frustum, far-50 depth range for the D16 ordering),
// so sensor and hand fragments share one depth test. Flat background
// depth like the sensor quads (provoking vertex decides the whole
// triangle — no interpolation smear at silhouettes); behind-camera
// verts emit <= 0 and sensor_splat.frag discards them to honest FAR
// holes. One draw per eye into the per-eye transient framebuffer
// (splat pass-1 pattern); the eye rides the push block, gl_ViewIndex
// is unused. Plain scalar math only (no mat3/ternary on this driver
// path); 96B VERT-only push, inside the Vulkan minimum.

layout(location = 0) in vec3 in_position;

layout(push_constant) uniform pc
{
	mat4 view;
	vec4 tan;
	vec2 near_far;
	int eye;
	int pad;
}
push;

layout(location = 0) flat out float v_bg;

void main()
{
	vec3 Xd = (push.view * vec4(in_position, 1.0)).xyz;
	float zbmax = min(-Xd.z, 20.0);
	v_bg = zbmax;

	float d = max(-Xd.z, 1e-6);
	vec4 dt = push.tan;
	float du = (Xd.x / (-Xd.z) - dt.x) / (dt.y - dt.x);
	float dv = (dt.z - Xd.y / (-Xd.z)) / (dt.z - dt.w);
	float near = push.near_far.x;
	float far = 50.0;
	gl_Position = vec4((2.0 * du - 1.0) * d, (2.0 * dv - 1.0) * d, far * (d - near) / (far - near), d);
}
