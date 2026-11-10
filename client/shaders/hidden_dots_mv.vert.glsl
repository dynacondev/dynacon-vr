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

#extension GL_EXT_multiview : require

// Multiview twin of hidden_dots.vert.glsl: identical camera-facing
// billboard expansion, per-eye MVP and view params selected by
// gl_ViewIndex. One instanced draw broadcasts to both layers. The vertex
// shader executes once per view (not once per draw), so per-invocation
// single-eye computation is exact — no cross-eye leakage. Plain integer
// indexing only (see AGENTS.md: no mat3/ternary on this driver path).
layout(location = 0) in vec2 corner; // static quad, -1..1
layout(location = 1) in vec3 center; // per-instance, object space
layout(location = 2) in vec3 normal; // per-instance, object-space face normal

layout(location = 0) out vec2 vUv; // quad space: 1.0 == dot radius
layout(location = 1) out float vW; // clip w (view distance) for the radius/AA
layout(location = 2) out vec2 vCenterUV; // dot-center sample point in the depth image
layout(location = 3) out float vCenterDepth; // dot-center depth (same mapping the prepass stores)

layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
	vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
	vec4 view_params[2]; // per eye: x: view W px, y: view H px, z: proj[1][1], w: dot radius (cells)
}
pc;

void main()
{
	mat4 mvp = pc.mvp[gl_ViewIndex];
	vec4 view_params = pc.view_params[gl_ViewIndex];
	// 2mm toward the camera off the face plane (front faces: the baked
	// normal points at the viewer): own-face depth tests then pass
	// strictly against the prepass raster, while occluded faces still
	// fail. Invisible, dwarfs 1-ulp rasterization differences.
	vec4 clip = mvp * vec4(center + normal * 0.002, 1.0);
	float w = clip.w;
	// Front-facing test: the normal as a direction (w=0) through the
	// MVP. The projection w-row is exactly (0,0,-1,0), so clip w is
	// -z_view(normal) with no lateral terms: w < 0 ⟺ the normal points
	// at the camera, exactly. Only caveat is non-uniform model scale
	// (normals want the inverse-transpose): sign can err in a thin
	// wedge around edge-on under strong anisotropy — accepted, those
	// dots are merged streaks either way.
	float facing = (mvp * vec4(normal, 0.0)).w;
	float pitch = max(pc.color_pitch.w, 1e-4);
	float px_per_m = (view_params.y * 0.5) * abs(view_params.z) / max(w, 1e-3);
	float r_px = view_params.w * pitch * px_per_m;
	vUv = corner;
	vW = w;
	// Center sample point for the occlusion compare below (same NDC→UV
	// mapping the viewport uses, so it lands on the prepass texel under
	// the dot center; depth in the mapping the prepass stores).
	vCenterUV = clip.xy / max(w, 1e-6) * 0.5 + 0.5;
	vCenterDepth = clip.z / max(w, 1e-6);
	if (w <= 1e-6 || facing >= 0.0)
	{
		// Behind the camera or backfacing: collapse outside clip space.
		gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
		return;
	}
	// corner ±1 maps to ±r_px screen pixels (aspect-correct per axis).
	vec2 px = corner * r_px / (view_params.xy * 0.5);
	gl_Position = vec4(clip.xy + px * w, clip.z, w);
}
