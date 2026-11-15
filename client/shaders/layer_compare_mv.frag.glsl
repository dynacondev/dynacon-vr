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

// Sensor reveal vs the min-dilated common layer, display space at W.
// sensor_tex: warped sensor D (hands MINed over it, D domain);
// layer_tex: R16F linear meters (MIN of windows + hidden + bbox, then
// the revealed sensor depths MINed back in by the add pass).
// keep = sensorPresent && sensLin < layer*(1-bias): distance-scaled
// bias, layer-only (sensor noise margin lives here now, not in a
// retired clean pass). Feeds the depth-group seed only — never the
// global path (that compare is the exact, bias-free merge verdict).
// The layer FAR clear (65504) reads as far: an absent sensor
// linearizes to near*1e6, still beyond it for near_z >= ~0.066, but
// the explicit present gate keeps fail-closed for any frustum.
// Plain scalar math only (no mat3/ternary on this driver path).

layout(push_constant) uniform Push
{
	// x = near (m), y = far (m, may be inf), z = bias (0..0.25,
	// layer only), w = pad. Mirrors the retired clean_push shape.
	vec4 near_far_bias;
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 1) uniform sampler2DArray layer_tex;

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
	float l = texture(layer_tex, buv).r;

	float n = pc.near_far_bias.x;
	float f = pc.near_far_bias.y;
	float b = pc.near_far_bias.z;

	float sensor_present = 1.0 - step(0.9999, d);
	float sens_lin = linearize_depth(d, n, f);
	float keep = (1.0 - step(l * (1.0 - b), sens_lin)) * sensor_present;
	out_color = vec4(keep);
}
