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

// Common-layer geometry fragment: R16F linear-meters output for MIN
// blending over the large-finite FAR clear (65504.0).
//
// Derivation (matches linearize_depth() EXACTLY): scene::
// projection_matrix builds a reversed-Z infinite-far projection whose
// third row is (0, 0, near, 0)-transposed, i.e. z_c = near and
// w_c = -z_view. So clip.w IS the planar eye distance in meters (d).
// The runtime D encoding is D = 1 - near/d (infinite-far branch, the
// only one the Quest runtime reports); linearize_depth inverts it as
// d = near/(1-D). Rastering clip.w writes d directly — identical to
// linearize_depth(D(d)) up to fp rounding, in both branches (finite-far
// D = f(d-n)/(d(f-n)) inverts to the same d; clip.w is still d).
// Verified against the volume_virt_mv.frag decode path, which applies
// the same D formula meters->D and inverts identically.
//
// Push block widens to the MVP pair so the shared VERT|FRAG layout reads
// nothing at the wrong offset (mask_mv precedent); the fragment stage
// reads only the eye_depth varying.
layout(push_constant) uniform PushConstants
{
	mat4 mvp[2];
}
pc;

layout(location = 0) in float eye_depth;

layout(location = 0) out vec4 out_color;

void main()
{
	float d = eye_depth;
	if (d <= 1e-6)
		discard;
	out_color = vec4(d);
}
