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

// Verdict apply: merged contributor alpha (full res, NEAREST — the
// feather bands are already filtered) times the half-res binary
// verdict (NEAREST — sampling a half-res texture at full-res uv is an
// exact 2x broadcast: texel floor(uv*half) = floor(k/2) per full-res
// pixel k, so no filter and no extra tap). No push. RGB stays zero
// (downstream the REPLACE blend preserves dst color and the veil bias
// touches alpha only, same as the plain merge it replaces). Always
// paired: the mask-only path (unpaired depth) takes
// vr_apply_mask_mv instead and never binds this pipeline.
layout(set = 0, binding = 0) uniform sampler2DArray merged_tex;
layout(set = 0, binding = 1) uniform sampler2DArray verdict_tex;

void main()
{
	vec3 buv = vec3(uv, float(gl_ViewIndex));
	float a = texture(merged_tex, buv).a;
	float v = texture(verdict_tex, buv).r;
	out_color = vec4(0.0, 0.0, 0.0, a * v);
}
