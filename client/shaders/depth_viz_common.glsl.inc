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

// Shared depth-viz helpers: planar-depth linearization plus the
// green->red isolation ramp used by every debug-viz colorize shader.
// Included (not compiled standalone: the .inc suffix keeps it out of
// the shader glob) by the viz fragment shaders and the splat vertex
// shader (linearize only there).
//
// Pass-2 convention (load-bearing for the shared pipeline layouts):
// every viz fragment shader declares the IDENTICAL fragment push block
// below (eye + near/far + opacity, 16 B). A pipeline bakes its set
// layout, so the shared 1-image and 2-image layouts stay valid for all
// pass-2 pipelines only while every shader's push block matches
// byte-for-byte (multiview lesson: never pair mismatched push blocks
// across a shared layout).
//
// Plain scalar math only (no mat3/ternary on this driver path).

float viz_linearize_depth(float d_norm, float n, float f)
{
	if (!(f > n) || f > 1e9)
	{
		// Infinite far plane. D=0 -> near, D=1 -> far.
		return n / max(1.0 - d_norm, 1e-6);
	}
	return (n * f) / max(f - d_norm * (f - n), 1e-6);
}

vec3 viz_ramp(float t)
{
	return mix(vec3(0.1, 0.9, 0.2), vec3(0.9, 0.15, 0.1), t);
}
