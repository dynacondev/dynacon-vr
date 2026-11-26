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

#extension GL_GOOGLE_include_directive : require

#include "depth_viz_common.glsl.inc"

// Environment-depth debug false-color: green at nearZ, red at farZ.
//
// The runtime D16 array texture is sampled directly per eye layer at
// the framebuffer UV (no reprojection, no history, no smoothing).
// NEAREST filtering keeps texel edges exact.
//
// Depth values are Vulkan D in [0,1] (D16_UNORM, same semantics as
// XR_KHR_composition_layer_depth): d = near*far/(far - D*(far-near)).
// Unbounded farZ (+inf, what the Quest runtime reports) or
// farZ < nearZ selects the infinite-projection branch.

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray depth_tex;

layout(push_constant) uniform pc
{
	int eye;
	float near_z;
	float far_z;
	float opacity;
} push;

// Depth values are Vulkan D in [0,1] (D16_UNORM, same semantics as
// XR_KHR_composition_layer_depth): d = near*far/(far - D*(far-near)).
// The runtime reports farZ = +inf when it has no far bound (or
// farZ < nearZ for the classic infinite projection): both select the
// infinite branch. Without the infinity check the finite math divides
// inf/inf = NaN and the whole viz goes transparent.

void main()
{
	float n = push.near_z;
	float f = push.far_z;

	// Y-flip: the framebuffer UV runs top-down against the depth
	// texture's bottom-up rows; sample flipped or the viz renders
	// upside down.
	float d_norm = texture(depth_tex, vec3(v_uv.x, 1.0 - v_uv.y, float(push.eye))).r;

	float d = viz_linearize_depth(d_norm, n, f);
	// Infinite far (unbounded or classic): normalize against the
	// reconstructed distance itself (t = 1 - near/d).
	bool infinite = !(f > n) || f > 1e9;
	float denom = infinite ? max(d, 1e-6) : (f - n);
	float t = clamp((d - n) / denom, 0.0, 1.0);
	vec3 col = viz_ramp(t);

	out_color = vec4(col, push.opacity);
}
