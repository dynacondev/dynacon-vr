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

// Soft-occlusion binary flatten, eye space: both inputs already live
// in the depth-stream (sync eye) frame at the working res, so the
// compare is a same-UV single tap with no reprojection. sensorW carries
// the warped *cleaned* sensor depth (known geometry + volume far wall
// already blanked to far in sensor space, off-canvas warp baked to
// far) with skinned hand depth MINed over it; virt is the server SBS
// depth (far-cleared when unpaired, i.e. neutral). The known-geometry
// bias lives in the sensor-space clean pass, not here: VR exact-cut,
// no margin. Empty volume never reveals via the clean (sensor already
// far there); empty virt (far clear) never blocks.

layout(push_constant) uniform Push
{
	// x unused (must be 0; the occluder bias lives in the sensor-space
	// clean), y = near (m), z = far (m, may be inf), w unused.
	vec4 bias_near_far;
}
pc;

layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 1) uniform sampler2DArray virt_tex;

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
	float w = texture(virt_tex, buv).r;

	float n = pc.bias_near_far.y;
	float f = pc.bias_near_far.z;

	float sensor_present = 1.0 - step(0.9999, d);
	float sensor_lin = linearize_depth(d, n, f);
	float virt_lin = linearize_depth(w, n, f);
	float reveal = (1.0 - step(virt_lin, sensor_lin)) * sensor_present;
	out_color = vec4(reveal);
}
