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

/*
 * Unpacked-depth range reduction (forensics): samples the decoded SBS
 * virtual-depth frame with the decoder's own sampler (the exact tap
 * the verdict/viz use: green channel, 2mm/LSB, codes 0-1023) and
 * reduces full-frame + center-patch (middle 50% each axis)
 * min/max/near/far counts into one host-visible SSBO (8 uints).
 * Dispatched throttled (~5s) from stream.cpp after the last depth
 * use; read back after the frame fence and logged. Answers whether
 * near-zero codes reach the client.
 */

#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D virt_tex;

layout(set = 0, binding = 1, std430) buffer Stats
{
	uint full_min;
	uint full_max;
	uint full_near;
	uint full_far;
	uint ctr_min;
	uint ctr_max;
	uint ctr_near;
	uint ctr_far;
} s;

void main()
{
	ivec2 sz = textureSize(virt_tex, 0);
	ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
	if (xy.x >= sz.x || xy.y >= sz.y)
		return;
	bool center = xy.x >= sz.x / 4 && xy.x < sz.x - sz.x / 4 &&
	              xy.y >= sz.y / 4 && xy.y < sz.y - sz.y / 4;
	float vd = textureLod(virt_tex, (vec2(xy) + vec2(0.5)) / vec2(sz), 0.0).g;
	uint c = uint(clamp(vd * 1023.0 + 0.5, 0.0, 1023.0));
	atomicMin(s.full_min, c);
	atomicMax(s.full_max, c);
	if (c <= 5u)
		atomicAdd(s.full_near, 1u);
	if (c >= 1021u)
		atomicAdd(s.full_far, 1u);
	if (center)
	{
		atomicMin(s.ctr_min, c);
		atomicMax(s.ctr_max, c);
		if (c <= 5u)
			atomicAdd(s.ctr_near, 1u);
		if (c >= 1021u)
			atomicAdd(s.ctr_far, 1u);
	}
}
