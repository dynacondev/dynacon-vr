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

// Depth write-back (3d-depth-passthrough): sensor depths that passed
// the reveal test re-enter the shared R16F layer so the VR verdict
// occludes against them like real geometry. Fullscreen pass into the
// shared framebuffer (LOAD, MIN blend): failing pixels write FAR
// (65504), which MIN keeps as the existing window depth; passing
// pixels write the sensor depth, MIN keeps the nearer of sensor and
// window. Same single test as depth_reveal_mv (same bindings, same
// bias push) — evaluated twice per frame (once per target), keeping
// the ownership clean (feather owns the alpha pass, merge owns the
// shared framebuffer). NEAREST throughout (no depth interpolation).
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
	float d = mix(65504.0, sensor, pass);
	out_color = vec4(d);
}
