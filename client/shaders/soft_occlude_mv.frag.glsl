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

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

// Soft-occlusion binary flatten (single tap, no supersample): the shared
// group blur feathers the edge, so the compare stays a hard binary vote.
// Eye UV remaps through the capture<->display relative rotation (H) plus
// the B2 lateral shift (T/depth). Identity reproduces the direct tap.
//
// Merged occluder = min(volume far, virtual, exact) in sensor D space:
// exact holds passthrough + hidden nearest depth (reversed excluded),
// volume far is the single depth-passthrough far wall, virtual is the
// server SBS depth. Distance-scaled bias applies in linear meters:
// mergedLin -= mergedLin*bias, then sensorLin < mergedLin reveals.
// Empty (1.0) / sensor-far / invalid-warp guards stay hard: unpaired or
// off-canvas frames reveal nothing (fail closed to exact-only).

layout(push_constant) uniform Push
{
	vec4 h0;
	vec4 h1;
	vec4 h2;
	vec4 h3;
	vec4 h4;
	vec4 h5;
	vec4 corr0;
	vec4 corr1;
	// x = bias (0..0.25), y = near (m), z = far (m, may be inf), w unused.
	vec4 bias_near_far;
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 1) uniform sampler2DArray volume_tex;
layout(set = 0, binding = 2) uniform sampler2DArray virt_tex;
layout(set = 0, binding = 3) uniform sampler2DArray exact_tex;

float linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		// Infinite far (Quest reports +inf). D=0 -> near, D=1 -> far.
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

void main()
{
	// Branchless eye select (mix, never a ternary/mat3).
	float vi = float(gl_ViewIndex);
	float m = step(0.5, vi);
	vec4 c0 = mix(pc.h0, pc.h3, m);
	vec4 c1 = mix(pc.h1, pc.h4, m);
	vec4 c2 = mix(pc.h2, pc.h5, m);
	vec4 corr = mix(pc.corr0, pc.corr1, m);
	vec2 T = corr.xy;
	float ka = corr.z;
	float kb = corr.w;
	float bias = pc.bias_near_far.x;
	float n = pc.bias_near_far.y;
	float f = pc.bias_near_far.z;

	// Single center tap (no 2x2 supersample: shared blur feathers).
	float px = c0.x * uv.x + c1.x * uv.y + c2.x;
	float py = c0.y * uv.x + c1.y * uv.y + c2.y;
	float pz = c0.z * uv.x + c1.z * uv.y + c2.z;
	float ok = step(1e-6, pz);
	vec2 s = vec2(px, py) / max(pz, 1e-6);
	ok *= step(0.0, s.x) * step(s.x, 1.0) * step(0.0, s.y) * step(s.y, 1.0);

	vec2 suv0 = s;
	float k_vol = 0.0;
	float v0 = texture(volume_tex, vec3(uv, vi)).r;
	float w0 = texture(virt_tex, vec3(uv, vi)).r;
	float e0 = texture(exact_tex, vec3(uv, vi)).r;
	// Depth-dependent lateral nudge from the rotation-only spot.
	float k_seed = ka * texture(sensor_tex, vec3(suv0, vi)).r + kb;
	float k = min(max(k_seed, ka * v0 + kb), 20.0);
	vec2 suv1 = suv0 + T * k;
	ok *= step(0.0, suv1.x) * step(suv1.x, 1.0) * step(0.0, suv1.y) * step(suv1.y, 1.0);
	float d = texture(sensor_tex, vec3(suv1, vi)).r;
	float v = texture(volume_tex, vec3(uv, vi)).r;
	float w = texture(virt_tex, vec3(uv, vi)).r;
	float e = texture(exact_tex, vec3(uv, vi)).r;

	float occ_d = min(v, min(w, e));
	float sensor_present = 1.0 - step(0.9999, d);
	float occ_present = 1.0 - step(0.9999, occ_d);
	float sensor_lin = linearize_depth(d, n, f);
	float occ_lin = linearize_depth(occ_d, n, f);
	occ_lin -= occ_lin * bias;
	float reveal = (1.0 - step(occ_lin, sensor_lin)) * sensor_present * occ_present * ok;
	out_color = vec4(reveal);
}
