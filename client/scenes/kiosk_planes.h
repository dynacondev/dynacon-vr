/*
 * WiVRn VR streaming
 * Copyright (C) 2026  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2026  Patrick Nicolas <patricknicolas@laposte.net>
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

#pragma once

#include "render/scene_components.h"
#include <entt/entity/fwd.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <openxr/openxr.h>
#include <random>
#include <vector>

class scene;

namespace wivrn
{

// Small fleet of model planes for the kiosk waiting overlays, shared by the
// lobby and stream scenes. Planes spawn one by one at random positions
// within 1m of the overlay panel, fade in, then fly forever under a
// curvature integrator: maneuvers (cruise/turn/loop/roll) are just rate
// targets, banking emerges from turn rate, and a weak containment bias pulls
// the fleet back toward the live anchor moth-to-light style (positions stay
// world-frozen, nothing translates with the panel). Create once the scene
// world exists (on_focused), update every frame the overlay may show,
// destroy with the scene (planes hold entities in the scene registry).
class kiosk_planes
{
public:
	kiosk_planes(::scene & scene, uint32_t layer_mask);

	// anchor is the overlay panel pose sampled this frame: spawn placement
	// uses it once (positions freeze once spawned), and the containment bias
	// steers toward its live position every frame. Inactive planes are
	// hidden; re-showing replays the one-by-one spawn-in.
	void update(XrTime now, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient, bool active);

private:
	struct plane
	{
		entt::entity root;
		std::vector<std::shared_ptr<renderer::material>> materials;
		// per-material base colors for the CPU-side premultiplied fade,
		// index-aligned with materials
		std::vector<glm::vec4> base_colors;
		std::vector<glm::vec3> base_emissives;
		XrTime spawn_t0 = 0; // scheduled spawn moment (fade clock starts here)
		bool spawned = false; // position assigned, never un-spawned while active
		// curvature-integrator flight state (world space; the live anchor
		// only steers via the containment bias, never translates the plane)
		glm::vec3 pos{0.f};
		glm::vec3 dir{0.f, 0.f, -1.f}; // unit flight direction
		glm::vec3 right{1.f, 0.f, 0.f}; // cached lateral axis (rebuilt unless near-vertical)
		float speed = 0.35f; // constant cruise (m/s)
		float bank = 0.f; // smoothed visual bank (radians, wrapped to [-pi, pi])
		float pitch_rate = 0.f, yaw_rate = 0.f, roll_rate = 0.f; // smoothed rates (rad/s)
		float pitch_tgt = 0.f, yaw_tgt = 0.f, roll_tgt = 0.f; // maneuver targets (rad/s)
		bool roll_free = false; // roll maneuver: relax suppressed, roll owns the bank
		bool loop_maneuver = false; // loop: exempt from the climb/dive rate clamp
		double man_t = 0.; // time left in the current maneuver (seconds; backstop for figures)
		double man_dur = 1.; // expected maneuver duration (envelope runs on this)
		double man_age = 0.; // elapsed time in the current maneuver (seconds)
		double man_angle = 0.; // accumulated figure angle (loops/rolls terminate at 2pi)
		XrTime prev_now = 0; // dt clock (0 = uninitialized)
	};

	::scene & scene_;
	std::vector<plane> planes_;
	std::mt19937 rng_;
	bool was_active_ = false;
	XrTime activated_t0_ = 0; // rising-edge time: plane i spawns at + i * spawn_interval
	// per-activation livery bag: one of each paint (orange included), shuffled.
	// Guarantees variety (no all-session hidden colors) with orange always
	// present; reshuffled on every re-show.
	std::vector<size_t> livery_bag_;
	size_t livery_next_ = 0;

	void spawn_at(plane & f, const glm::vec3 & anchor_pos);
	void apply_livery(plane & f);
	void pick_maneuver(plane & f);
};

} // namespace wivrn
