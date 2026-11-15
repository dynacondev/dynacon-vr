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

// Soft-occlusion binary flatten, eye space: every input canvas already
// lives in the depth-stream (sync eye) frame at the working res, so the
// compare is a same-UV single tap with no reprojection and a 16B push.
// sensorW carries the warped runtime depth (off-canvas warp baked to
// far) with skinned hand depth MINed over it; volFar is the single
// depth-passthrough far wall; virt is the server SBS depth; occDil is
// the dilated known-geometry occluder (passthrough windows + hidden).
// The distance-scaled bias applies to the known-geometry term ONLY:
// mergedLin -= mergedLin*bias, then sensorLin < mergedLin reveals.
// Volume-far and VR exact-cut (no margin). Unpaired virt rebinds the
// dilated occluder (min no-op). Empty volume (0 clear, very near) never
// reveals; empty occluder (far clear) never blocks.

layout(push_constant) uniform Push
{
	// x = bias (0..0.25), y = near (m), z = far (m, may be inf), w unused.
	vec4 bias_near_far;
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 1) uniform sampler2DArray volume_tex;
layout(set = 0, binding = 2) uniform sampler2DArray virt_tex;
layout(set = 0, binding = 3) uniform sampler2DArray occ_tex;

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
	float vi = float(gl_ViewIndex);
	vec3 buv = vec3(uv, vi);
	float d = texture(sensor_tex, buv).r;
	float v = texture(volume_tex, buv).r;
	float w = texture(virt_tex, buv).r;
	float e = texture(occ_tex, buv).r;

	float bias = pc.bias_near_far.x;
	float n = pc.bias_near_far.y;
	float f = pc.bias_near_far.z;

	float sensor_present = 1.0 - step(0.9999, d);
	float sensor_lin = linearize_depth(d, n, f);
	float vol_lin = linearize_depth(v, n, f);
	float virt_lin = linearize_depth(w, n, f);
	float occ_lin = linearize_depth(e, n, f);
	occ_lin -= occ_lin * bias;
	float reveal = (1.0 - step(vol_lin, sensor_lin)) * (1.0 - step(virt_lin, sensor_lin)) *
	               (1.0 - step(occ_lin, sensor_lin)) * sensor_present;
	out_color = vec4(reveal);
}
