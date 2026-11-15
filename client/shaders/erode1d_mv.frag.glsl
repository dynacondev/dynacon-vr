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

// Multiview twin of erode1d.frag.glsl: identical box-minimum pass,
// reading the calling view's layer. Same push layout (24 bytes), so it
// shares the blur pipeline layout. Taps stay on exact texel centers:
// pair with the point sampler, as in the single-view path.
layout(set = 0, binding = 0) uniform sampler2DArray src;

layout(push_constant) uniform Push
{
	vec2 texel;
	vec2 dir;
	float radius;
	float pad;
}
pc;

void main()
{
	int r = int(pc.radius + 0.5);
	vec3 duv = vec3(pc.dir * pc.texel, 0);
	vec3 base_uv = vec3(uv, float(gl_ViewIndex));
	float m = texture(src, base_uv).r;
	for (int i = -8; i <= 8; ++i)
	{
		if (i == 0 || abs(i) > r)
			continue;
		m = min(m, texture(src, base_uv + duv * float(i)).r);
	}
	out_color = vec4(m);
}
