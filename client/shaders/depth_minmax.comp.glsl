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
 * the verdict/viz use: 2mm/LSB, codes 0-1023) and reduces
 * full-frame + center-patch (middle 50% each axis)
 * min/max/near/far counts per channel (R, G, B) into one host-visible
 * SSBO (24 uints: per channel, full min/max/near/far + center
 * min/max/near/far). The verdict/viz sample .g only; R and B expose
 * whether a shift is luma-wide (R=G=B, genuine depth offset) or
 * chroma leakage (709-full with zero chroma lifts G by +335 while
 * crushing R to Y-0.79 and B to Y-0.93). Dispatched throttled (~5s)
 * from stream.cpp after the last depth use; read back after the frame
 * fence and logged. Answers whether near-zero codes reach the client.
 */

#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D virt_tex;

layout(set = 0, binding = 1, std430) buffer Stats
{
	uint d[24];
} s;

void main()
{
	ivec2 sz = textureSize(virt_tex, 0);
	ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
	if (xy.x >= sz.x || xy.y >= sz.y)
		return;
	bool center = xy.x >= sz.x / 4 && xy.x < sz.x - sz.x / 4 &&
	              xy.y >= sz.y / 4 && xy.y < sz.y - sz.y / 4;
	vec3 rgb = textureLod(virt_tex, (vec2(xy) + vec2(0.5)) / vec2(sz), 0.0).rgb;
	for (int ch = 0; ch < 3; ++ch)
	{
		uint c = uint(clamp(rgb[ch] * 1023.0 + 0.5, 0.0, 1023.0));
		uint b = uint(ch) * 8u;
		atomicMin(s.d[b + 0u], c);
		atomicMax(s.d[b + 1u], c);
		if (c <= 5u)
			atomicAdd(s.d[b + 2u], 1u);
		if (c >= 1021u)
			atomicAdd(s.d[b + 3u], 1u);
		if (center)
		{
			atomicMin(s.d[b + 4u], c);
			atomicMax(s.d[b + 5u], c);
			if (c <= 5u)
				atomicAdd(s.d[b + 6u], 1u);
			if (c >= 1021u)
				atomicAdd(s.d[b + 7u], 1u);
		}
	}
}
