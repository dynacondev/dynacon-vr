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

// Camera-facing billboard expansion for the baked dot instances (one
// instance = object-space center + face normal, one static quad). The
// quad is square in screen pixels by construction, so dots are round
// from any angle and extend a full radius past the center, so triangle
// edges never slice them. Backfaces collapse here (facing test on the
// baked normal; the layer has no depth test).
layout(location = 0) in vec2 corner; // static quad, -1..1
layout(location = 1) in vec3 center; // per-instance, object space
layout(location = 2) in vec3 normal; // per-instance, object-space face normal

layout(location = 0) out vec2 vUv; // quad space: 1.0 == dot radius
layout(location = 1) out float vW; // clip w (view distance) for the radius/AA

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
	vec4 view_params; // x: view W px, y: view H px, z: proj[1][1], w: dot radius (cells)
}
pc;

void main()
{
	// 2mm toward the camera off the face plane (front faces: the baked
	// normal points at the viewer): own-face depth tests then pass
	// strictly against the prepass raster, while occluded faces still
	// fail. Invisible, dwarfs 1-ulp rasterization differences.
	vec4 clip = pc.mvp * vec4(center + normal * 0.002, 1.0);
	float w = clip.w;
	// Front-facing test: the normal as a direction (w=0) through the
	// MVP has clip w = -z_view(normal), so w < 0 means the normal has
	// +z_view, i.e. points at the camera. Approximate only for faces
	// nearly edge-on far off-axis (there the lateral terms outweigh
	// n_z); those dots are merged streaks either way, so a slightly
	// early/late pop there is accepted.
	float facing = (pc.mvp * vec4(normal, 0.0)).w;
	float pitch = max(pc.color_pitch.w, 1e-4);
	float px_per_m = (pc.view_params.y * 0.5) * abs(pc.view_params.z) / max(w, 1e-3);
	float r_px = pc.view_params.w * pitch * px_per_m;
	vUv = corner;
	vW = w;
	if (w <= 1e-6 || facing >= 0.0)
	{
		// Behind the camera or backfacing: collapse outside clip space.
		gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
		return;
	}
	// corner ±1 maps to ±r_px screen pixels (aspect-correct per axis).
	vec2 px = corner * r_px / (pc.view_params.xy * 0.5);
	gl_Position = vec4(clip.xy + px * w, clip.z, w);
}
