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

// Sensor-space known-geometry clean: the runtime sensor, the dilated
// known-geometry occluder (passthrough windows + hidden, nearest-D at
// the sensor extent/pose) and the depth-volume far wall (furthest-D,
// same sensor frame) are all sampled native — no warp resampling on
// either side. Keeps the sensor sample iff it resolves in front of
// both the biased occluder and the volume far wall; otherwise writes
// far (1.0: fail closed, never reveals).
//
// keep = sensorPresent && sensorLin < volLin && sensorLin < occBiasedLin
//   occBiasedLin = occLin - occLin*bias (distance-scaled, occluder only;
//   volume-far stays exact). Empty occluder (far clear) passes anything
//   but far; empty volume (near clear 0) blocks everything outside the
//   silhouette (correct: no sensor verdict there — window reveal comes
//   from the display-space alpha field, not the sensor).
// Reversed cuts deliberately excluded (alpha-side only, cut-before-add).

layout(push_constant) uniform Push
{
	// x = bias (0..0.25, occluder only), y = near (m), z = far (m, may
	// be inf), w unused.
	vec4 bias_near_far;
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 1) uniform sampler2DArray occ_tex;
layout(set = 0, binding = 2) uniform sampler2DArray vol_tex;

float linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		// Infinite far (Quest reports +inf): D=0 -> near, D=1 -> far.
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

void main()
{
	float vi = float(gl_ViewIndex);
	vec3 buv = vec3(uv, vi);
	float d = texture(sensor_tex, buv).r;
	float o = texture(occ_tex, buv).r;
	float v = texture(vol_tex, buv).r;

	float bias = pc.bias_near_far.x;
	float n = pc.bias_near_far.y;
	float f = pc.bias_near_far.z;

	float sensor_present = 1.0 - step(0.9999, d);
	float sensor_lin = linearize_depth(d, n, f);
	float occ_lin = linearize_depth(o, n, f);
	float vol_lin = linearize_depth(v, n, f);
	occ_lin -= occ_lin * bias;
	float keep = (1.0 - step(vol_lin, sensor_lin)) * (1.0 - step(occ_lin, sensor_lin)) * sensor_present;
	out_color = vec4(mix(1.0, d, keep));
}
