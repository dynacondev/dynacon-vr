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

// Foveated virtual-depth decode + exact global verdict at FULL merge
// res. One fullscreen pass, no intermediate depth canvas: the verdict
// compares the uniform raster layer against the foveated SBS depth
// through the shared foveation tables (same tables as color: same
// gaze center and source size on every depth-carrying warped frame).
//
// layer_tex: R16F common layer (linear meters, MIN of the positive
// window draws, FAR clear 65504 reads as absent). NEAREST: sampled at
// exact fragment centers, no halo.
//
// virt_tex: decoded stream-3 SBS frame (single image, both eyes side
// by side), sampled with the frame handle's own generation sampler at exact
// packed texel centers. Green channel: smallest chroma-crosstalk gain
// under the 709 conversion, so lossy chroma wobble perturbs depth
// least through green (same convention as the old volume_virt decode).
// A center tap returns that texel — but note the depth payload writes
// constant neutral chroma, so no chroma filtering can perturb this
// path today; a future chroma-modulating payload would need to
// re-check the LINEAR case. Either way the foveated packing is read
// exactly, never interpolated. Interpolating depths would blend
// foreground/background into false mid-depths (halo); the broadcast
// used here (many uniform pixels sharing one packed texel) is exact
// by construction.
//
// Tables: compact foveation runs uploaded per verdict frame (no CPU
// expand, no 2*(W+H) index map). Layout of t[] (uint32s):
//   t[0..3] = render_w, render_h, packed_w, packed_h
//   t[4..7] = per-axis run counts [nx0, nx1, ny0, ny1]
//   x0 runs @ 8, x1 @ 8+MAX, y0 @ 8+2*MAX, y1 @ 8+3*MAX
// (MAX = FOVEA_MAX_RUNS below, must match kFoveaMaxRuns in
// union_merge.cpp). Runs describe packed texels: entry i holds n_i
// texels covering n_i * ratio_i render pixels (ratio_i =
// |center-i|+1, center = count/2, middle run 1:1) — the shared color
// tables, same as the old expand path. Empty axis (count 0) falls
// back to a uniform map over the packed size (straight /
// pre-foveation data). SSBO (not UBO/push): kept as a buffer so the
// verdict descriptor layout is unchanged; contents are ~1KB, not 34KB.
//
// SBS encoding: 10-bit 2mm/LSB luma, valid codes 0-1020 (0-2.040m);
// holes, sky and beyond-range pin to 1023 (far); 1021-1023 is a guard
// band absorbing lossy wobble. Anything past 1020 decodes as far. No
// near/far push: SBS codes are already meters (code * 0.002), so
// neither side needs a frustum to agree. The raster layer uses the
// same cut: l >= 2.041m (1020.5 * 0.002) is far.
//
// verdict: absent (FAR clear) keeps merged; far-vs-far is a VR win
// (reveal 0); virt-far alone keeps the window (reveal 1); otherwise
// reveal = !(virt strictly nearer). Exact, no bias anywhere — the
// boundary follows the sharper input per pixel (partial smoothness:
// the full-res raster varies inside one packed texel). Tie keeps the
// window.
// Geometry-free pixels (reversed-only cuts, hidden, debug cutouts
// never write the layer) read FAR and keep whatever
// the merged union says. Global: applies to every group at once,
// after all alpha is combined; reversed cuts stay per-group feather
// alpha and are unaffected here.
//
// Layout note: the decoder image is normalized to GENERAL at frame
// top on the main command buffer and sampled read-only here under
// the same steady-state discipline as the video path. Transition
// frames sit the verdict out scene-side (depth_verdict_safe: neutral
// merged submit, fail-open), so this pass only ever samples
// steady-state GENERAL — including on the async mask path.
//
// Plain scalar math only (no mat3/ternary on this driver path). The
// table fetch is derivative-free (integer indexing, no sampling), so
// no derivative concerns anywhere in this shader.
layout(set = 0, binding = 0) uniform sampler2DArray layer_tex;
layout(set = 0, binding = 1) uniform sampler2D virt_tex;
layout(set = 0, binding = 2) readonly buffer Fovea
{
	uint t[];
};

// Fixed run capacity per axis (must match kFoveaMaxRuns in
// union_merge.cpp).
#define FOVEA_MAX_RUNS 64
#define FOVEA_HDR 8

// Merge pixel -> packed texel through the compact runs. Replicates
// expand_foveation_axis (union_merge.cpp) exactly: uniform
// merge->render floor, then the render->packed walk (run i covers
// runs[i]*ratio_i render pixels at ratio_i = |center-i|+1). Returns
// the packed index clamped to [0, packed_n-1]; undershoot (runs
// covering fewer than render_n pixels) pads with the last texel,
// all-zero runs read 0 — same as the old r2p pad.
int fovea_axis(int u, int render_n, int merge_n, int packed_n, int count, int base)
{
	if (render_n <= 0 || merge_n <= 0)
		return 0;
	if (packed_n <= 0)
		return 0;
	if (count <= 0)
	{
		int v = u * packed_n / merge_n;
		return min(v, packed_n - 1);
	}
	int r = u * render_n / merge_n;
	r = min(r, render_n - 1);
	int center = count / 2;
	int pos = 0;
	int packed = 0;
	for (int i = 0; i < FOVEA_MAX_RUNS; ++i)
	{
		if (i >= count)
			break;
		int n = int(t[base + i]);
		int ratio = abs(center - i) + 1;
		int span = n * ratio;
		if (r < pos + span)
		{
			int k = (r - pos) / ratio;
			return min(packed + k, packed_n - 1);
		}
		pos += span;
		packed += n;
	}
	if (packed <= 0)
		return 0;
	return min(packed - 1, packed_n - 1);
}

void main()
{
	float vi = float(gl_ViewIndex);
	int eye = int(gl_ViewIndex);
	vec3 buv = vec3(uv, vi);
	float l = texture(layer_tex, buv).r;
	float layer_absent = step(65000.0, l);
	// Empty-pixel fast path: every tap below (SSBO index fetch,
	// YCbCr depth sample) is wasted where the layer is FAR-clear —
	// for a small window that is ~all pixels. Exact: the reveal
	// formula below forces reveal=1 whenever layer_absent=1.
	if (layer_absent > 0.5)
	{
		out_color = vec4(1.0);
		return;
	}

	// Merge pixel -> packed texel through the compact runs (single
	// domain: same normalized uv as the layer tap, so the verdict
	// cannot misalign under a future viewport offset).
	ivec2 msize = textureSize(layer_tex, 0).xy;
	ivec2 dst_texel = ivec2(uv * vec2(msize));
	dst_texel = clamp(dst_texel, ivec2(0, 0), msize - ivec2(1, 1));
	int W = msize.x;
	int H = msize.y;
	int render_w = int(t[0]);
	int render_h = int(t[1]);
	int packed_w = int(t[2]);
	int packed_h = int(t[3]);
	int nx = int(t[4 + eye]);
	int ny = int(t[6 + eye]);
	int px = fovea_axis(dst_texel.x, render_w, W, packed_w, nx, FOVEA_HDR + eye * FOVEA_MAX_RUNS);
	int py = fovea_axis(dst_texel.y, render_h, H, packed_h, ny, FOVEA_HDR + 2 * FOVEA_MAX_RUNS + eye * FOVEA_MAX_RUNS);

	// Texel-center UV into the SBS frame (row 0 = scene top on the
	// wire, matching the tables' top-down packed rows).
	ivec2 vsize = textureSize(virt_tex, 0).xy;
	int half_w = vsize.x / 2;
	vec2 vuv = vec2((float(px + eye * half_w) + 0.5) / float(vsize.x),
	                (float(py) + 0.5) / float(vsize.y));
	float vd = texture(virt_tex, vuv).g;

	float virt_far = step(1020.5 / 1023.0, vd);
	float layer_far = step(2.041, l) * (1.0 - layer_absent);
	float virt_m = vd * 2.046;
	float vr_wins_near = (1.0 - virt_far) * (1.0 - step(l, virt_m));
	float vr_wins = max(vr_wins_near, virt_far * layer_far);
	float reveal = 1.0 - (1.0 - layer_absent) * vr_wins;
	out_color = vec4(reveal);
}
