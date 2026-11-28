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

// Depth reveal (3d-depth-passthrough): single sensor-vs-copy test,
// MAX-blended over the binary working alpha ahead of the erode/blur
// chain so the reveal feathers like a normal window. The blend unit
// does the OR (no work tap: sampling the bound attachment would be a
// feedback loop); R8 MIN-blend precedent (reversed raster) proves the
// blend path on this format.
//
// sensor_tex: resampled sensor heightfield (R16F planar meters, sync-eye
// grid, hands in), NEAREST: interpolating depths would blend
// fore/background into false mid-depths (verdict-shader rule).
// copy_tex: merged known geometry (R16F linear meters, same
// normalized grid), NEAREST. bias: depth_bias_m over the whole copy.
//
// pass = sensor strictly nearer than copy + bias, both sides valid
// (sensor FAR = no return, copy FAR = outside every volume and window
// both read as absent — never reveal).
layout(set = 0, binding = 0) uniform sampler2DArray sensor_tex;
layout(set = 0, binding = 1) uniform sampler2DArray copy_tex;

layout(push_constant) uniform Push
{
	float bias;
}
pc;

void main()
{
	vec3 buv = vec3(uv, float(gl_ViewIndex));
	float sensor = texture(sensor_tex, buv).r;
	float known = texture(copy_tex, buv).r;
	float valid = (1.0 - step(65000.0, sensor)) * (1.0 - step(65000.0, known));
	float pass = valid * (1.0 - step(known + pc.bias, sensor));
	out_color = vec4(pass);
}
