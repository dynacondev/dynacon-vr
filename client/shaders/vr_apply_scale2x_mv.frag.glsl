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

// Verdict apply with fused binary Scale2x upscale: merged contributor
// alpha (full res, NEAREST — the feather bands are already filtered)
// times the binary verdict (W -> full) edge-preservingly upscaled.
//
// The verdict is an exact 0/1 R8 at half merge res (fused SBS-decode +
// compare in vr_verdict_mv). Bilinear-upsampling it would fringe every
// occlusion edge with a 0.5 halo (merged.a * 0.5 soft rim). Scale2x
// keeps the binary edge crisp and preserves sharp corners: each half-res
// verdict texel E with edge-neighbors B (y-1), D (x-1), F (x+1), H (y+1)
// expands to 4 full-res pixels:
//
//   E00 E01      D==B && B!=F && D!=H ? D : E      B==F && ... ? F : E
//   E10 E11  =   D==H && ... ? D : E               H==F && ... ? F : E
//
// with the standard guard (B==H && D==F -> all E) suppressing
// ambiguous-diagonal artifacts. Only the B/D/E/F/H subset is needed;
// A/C/G/I belong to Scale3x/Eagle variants.
//
// Mapping: fullscreen triangle (blur.vert) + full-res viewport, so the
// fragment covering dst texel (i,j) has gl_FragCoord.xy == (i+0.5,
// j+0.5) and ivec2(gl_FragCoord.xy) == (i,j). With W_full == 2*W_half
// the covering verdict texel is exactly dst/2 (integer division) and
// the sub-pixel is dst - src*2. texelFetch is derivative-free, so the
// per-sub-pixel branch divergence is safe (texture() with implicit
// derivatives would be UB there). Coordinates are clamped per tap
// (texelFetch does not clamp); the odd-extent edge (W_full == 2k+1,
// half == k by floor) duplicates the edge texel for dst column 2k —
// harmless, video extents are even in practice.
//
// Fail-open: the no-virt fallback CLEARs the verdict to 1, so all 5
// taps read 1 and every branch returns E == 1 (submit merged directly,
// pixel-identical neutral parity).
//
// RGB stays zero (downstream the REPLACE blend preserves dst color and
// the veil bias touches alpha only). No push constants (sizes from
// textureSize), no YCbCr here (verdict tap is plain R8), plain scalar
// math only (no mat3/ternary on this driver path).
layout(set = 0, binding = 0) uniform sampler2DArray merged_tex;
layout(set = 0, binding = 1) uniform sampler2DArray verdict_tex;

void main()
{
	vec3 buv = vec3(uv, float(gl_ViewIndex));
	float a = texture(merged_tex, buv).a;

	int layer = int(gl_ViewIndex);
	ivec2 dst_texel = ivec2(gl_FragCoord.xy);
	ivec2 src_texel = dst_texel / 2;
	ivec2 sub = dst_texel - src_texel * 2;

	ivec2 vsize = textureSize(verdict_tex, 0).xy;
	ivec2 lo = ivec2(0);
	ivec2 hi = vsize - ivec2(1);

	float e = texelFetch(verdict_tex, ivec3(clamp(src_texel, lo, hi), layer), 0).r;
	float b = texelFetch(verdict_tex, ivec3(clamp(src_texel + ivec2(0, -1), lo, hi), layer), 0).r;
	float d = texelFetch(verdict_tex, ivec3(clamp(src_texel + ivec2(-1, 0), lo, hi), layer), 0).r;
	float f = texelFetch(verdict_tex, ivec3(clamp(src_texel + ivec2(1, 0), lo, hi), layer), 0).r;
	float h = texelFetch(verdict_tex, ivec3(clamp(src_texel + ivec2(0, 1), lo, hi), layer), 0).r;

	// Binarize past the R8 quant (verdict writes exact 0/1, but never
	// trust == on a sampled value).
	bool be = b > 0.5;
	bool de = d > 0.5;
	bool ee = e > 0.5;
	bool fe = f > 0.5;
	bool he = h > 0.5;

	float v = ee ? 1.0 : 0.0;
	if (!(be == he && de == fe))
	{
		if (sub.x == 0 && sub.y == 0)
			v = (de == be && be != fe && de != he) ? (de ? 1.0 : 0.0) : (ee ? 1.0 : 0.0);
		else if (sub.x == 1 && sub.y == 0)
			v = (be == fe && be != de && fe != he) ? (fe ? 1.0 : 0.0) : (ee ? 1.0 : 0.0);
		else if (sub.x == 0 && sub.y == 1)
			v = (de == he && de != be && he != fe) ? (de ? 1.0 : 0.0) : (ee ? 1.0 : 0.0);
		else
			v = (he == fe && he != be && fe != de) ? (fe ? 1.0 : 0.0) : (ee ? 1.0 : 0.0);
	}

	out_color = vec4(0.0, 0.0, 0.0, a * v);
}
