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

// Depth-volume flatten with rotation + B2 lateral translation: the sensor
// tap is remapped through the capture<->display relative rotation, then
// nudged by T/depth for sideways parallax (forward/radial + depth-value
// terms are NOT handled here — that residual is the experiment).
// Identity reproduces the direct tap. Binary sensor<v test with hard
// far / no-data guards, evaluated 2x2 per texel (quarter-texel eye-UV
// offsets, averaged): true geometric coverage, so the contour positions
// sub-texel and diagonals stop staircasing. 12 taps per texel (3 per
// sub-pixel) instead of 3 -- the price of real AA. Both taps are
// LINEAR-filtered (host sampler).
// (History: per-texel softening — fixed and slope-adaptive e bands —
// could only fade whole texels, never move contour vertices between
// them, and the fixed band went transparent on thin interiors. Only a
// finer evaluation rate fixes texel-corner stairs.)
//
// The remap rides push constants as 8 NAMED vec4 members (not a GLSL
// array: push-constant arrays have driver-defined packing, while
// consecutive vec4 members lay out exactly — 16B each, 128B total).
// Per-eye homography columns: eye e uses h[3e..3e+2]. Per-eye lateral
// correction: eye e uses corr[e].xy = T (UV per 1/m), corr[e].z/.w =
// (a, b) with 1/depth = a*D + b (duplicated; D is sensor-compatible).
// (History: the original failure was the mat3 constructor in the math
// below, never the transport — UBO, texel-buffer and sampled-image
// variants all behaved identically once mat3 was gone. Push is kept as
// the simplest form: no buffers, views, copies or barriers per frame.)
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
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 1) uniform sampler2DArray volume_tex;
// Virtual-depth D canvas (same 320px 2-layer shape as volume_tex):
// sensor-compatible D, 1.0 = far/no-virtual. Unpaired frames rebind the
// volume canvas here, so min(v, v) = v reproduces the legacy sensor <
// volume test bit-identically (no null descriptors, no shader variant).
layout(set = 0, binding = 2) uniform sampler2DArray virt_tex;

void main()
{
	// Branchless eye select (mix, never a ternary/mat3): subs within a
	// pixel diverge, so validity stays flag-folded throughout.
	float vi = float(gl_ViewIndex);
	float m = step(0.5, vi);
	vec4 c0 = mix(pc.h0, pc.h3, m);
	vec4 c1 = mix(pc.h1, pc.h4, m);
	vec4 c2 = mix(pc.h2, pc.h5, m);
	vec4 corr = mix(pc.corr0, pc.corr1, m);
	vec2 T = corr.xy;
	float ka = corr.z;
	float kb = corr.w;

	// 2x2 supersample: the verdict is evaluated at four quarter-texel
	// eye-UV positions and averaged, giving true geometric coverage
	// (0/.25/.5/.75/1). Per-texel softening can only fade whole texels;
	// only a finer evaluation rate moves contour vertices between
	// texels, which is what de-staircases diagonals. The warp (H poly,
	// k-guard) is per-sub: sharing center-k would reintroduce the
	// discontinuity catch-22 at sub scale for the exact pixels this
	// fixes.
	const float q = 0.25 / 320.0;
	float inside = 0.0;
	for (int i = 0; i < 4; ++i)
	{
		vec2 uvi = uv + (vec2(float(i & 1), float((i >> 1) & 1)) - 0.5) * (2.0 * q);
		float px = c0.x * uvi.x + c1.x * uvi.y + c2.x;
		float py = c0.y * uvi.x + c1.y * uvi.y + c2.y;
		float pz = c0.z * uvi.x + c1.z * uvi.y + c2.z;
		float ok = step(1e-6, pz);
		vec2 s = vec2(px, py) / max(pz, 1e-6);
		ok *= step(0.0, s.x) * step(s.x, 1.0) * step(0.0, s.y) * step(s.y, 1.0);

		// B2 lateral: one depth-dependent nudge from the rotation-only
		// spot (T maps 1/depth to sensor-UV, k affine in D, no divide).
		// Forward/radial + depth-value terms deliberately absent.
		vec2 suv0 = s;
		float d0 = texture(sensor_tex, vec3(suv0, vi)).r;
		float v = texture(volume_tex, vec3(uvi, vi)).r;
		// Virtual occluder in the same D space: min() folds virtual and
		// volume far walls into one occluder. Unpaired frames rebind the
		// volume canvas (w == v), so the legacy test is bit-identical.
		float w = texture(virt_tex, vec3(uvi, vi)).r;
		// k guard per sub (see above for why it can't be shared).
		float k_sens = ka * d0 + kb;
		float k_vol = (v > 0.0) ? ka * v + kb : 0.0;
		float k = min(max(k_sens, k_vol), 20.0);
		vec2 suv1 = suv0 + T * k;
		ok *= step(0.0, suv1.x) * step(suv1.x, 1.0) * step(0.0, suv1.y) * step(suv1.y, 1.0);
		float d = texture(sensor_tex, vec3(suv1, vi)).r;

		// Binary vote, hard guards (far d, empty v): the 4-level average
		// IS the antialiasing, no band tuning involved. The occluder is
		// the nearer of the volume far wall and the virtual depth (both
		// sensor-compatible D); 1.0 in either is far. The empty-volume
		// gate stays on v alone: virt must never passthrough outside a
		// volume (occ would be min(0, w) = 0 there anyway).
		float occ = min(v, w);
		inside += ok * (1.0 - step(occ, d)) * step(1e-6, v) * (1.0 - step(0.9999, d));
	}
	out_color = vec4(inside * 0.25);
}
