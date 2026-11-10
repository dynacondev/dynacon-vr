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

// World-space triplanar warning grid + fresnel rim. Face normals come from
// screen-space derivatives of the world position (flat facets, no normal
// attributes needed). Grid lines wrap the mesh faces (triplanar blend by
// dominant normal axis); the rim defines the silhouette. No fill by design.
layout(location = 0) in vec3 vWorld;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants
{
	mat4 mvp;
	vec4 obj_pos_scale;
	vec4 obj_quat;
	vec4 grid_col_scale; // rgb: grid color, w: grid spacing (m)
	vec4 cam_pos_width; // xyz: eye world position (m), w: line width (px)
}
pc;

// Hardcoded concern rim (stage 1): red, same pixel thickness as the grid.
const vec3 rim_color = vec3(1.0, 0.16, 0.10);

// Grid-line distance in pixels for one 2D coordinate pair.
float grid_px(vec2 q)
{
	vec2 fw = max(fwidth(q), vec2(1e-4));
	vec2 g = abs(fract(q - 0.5) - 0.5) / fw;
	return min(g.x, g.y);
}

void main()
{
	vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
	vec3 to_cam = pc.cam_pos_width.xyz - vWorld;
	float cam_dist = length(to_cam);
	vec3 v = to_cam / max(cam_dist, 1e-5);

	float spacing = max(pc.grid_col_scale.w, 1e-4);
	vec3 gp = vWorld / spacing;

	// Triplanar blend: each plane's lines weighted by the dominant axis
	// so grazing faces don't smear.
	vec3 w3 = pow(abs(n), vec3(4.0));
	w3 /= max(w3.x + w3.y + w3.z, 1e-4);
	float line_d = grid_px(gp.xy) * w3.z + grid_px(gp.xz) * w3.y + grid_px(gp.yz) * w3.x;

	float wpx = max(pc.cam_pos_width.w, 0.5);
	float line = 1.0 - smoothstep(wpx - 1.0, wpx, line_d);

	// Fresnel rim, same pixel thickness.
	float fres = pow(1.0 - abs(dot(n, v)), 3.0);
	float rim_d = fres / max(fwidth(fres), 1e-4);
	float rim = 1.0 - smoothstep(wpx - 1.0, wpx, rim_d);

	float a = clamp(line + rim, 0.0, 1.0);
	if (a <= 0.001)
		discard;
	out_color = vec4(pc.grid_col_scale.rgb * line + rim_color * rim, a);
}
