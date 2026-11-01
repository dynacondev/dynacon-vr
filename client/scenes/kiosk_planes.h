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
#include <array>
#include <memory>
#include <openxr/openxr.h>
#include <optional>
#include <random>
#include <vector>

class scene;

namespace wivrn
{

// Small fleet of model planes for the kiosk waiting overlays, shared by the
// lobby and stream scenes. Each plane flies chained world-space Bezier
// segments launched from behind the overlay panel, meandering around the
// live centre for ~3x a segment time before fading out and respawning: new
// segments steer at the anchor sampled that frame, so walking away pulls
// the fleet along, while in-flight segments stay frozen. Create once the
// scene world exists (on_focused), update every frame the overlay may show,
// destroy with the scene (planes hold entities in the scene registry).
class kiosk_planes
{
public:
	kiosk_planes(::scene & scene, uint32_t layer_mask, uint32_t trail_layer_mask);

	// anchor is the overlay panel pose sampled this frame: used for fresh
	// spawns and, as it moves, to steer newly chained segments back toward
	// the centre. Inactive planes are hidden and frozen.
	// camera is the viewer position for billboarding the trails; without
	// it the ribbons fall back to facing up.
	void update(XrTime now, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient, bool active, const std::optional<glm::vec3> & camera);

private:
	struct flight
	{
		entt::entity root;
		std::vector<std::shared_ptr<renderer::material>> materials;
		// per-material base colors for the CPU-side premultiplied fade,
		// index-aligned with materials
		std::vector<glm::vec4> base_colors;
		std::vector<glm::vec3> base_emissives;
		glm::vec3 p0, p1, p2, p3;
		XrTime t0 = 0; // launch time: fade-in reference (flights never end)
		// Pacing for the CURRENT segment: cruise speed is total_len/seg_T.
		// Cruise evolves gently per segment (never redrawn), so joints
		// have no velocity step; bank_sm is the smoothed applied bank for
		// the same reason (curvature steps at joints).
		float seg_T = 11.f;
		float cruise = 0.4f;
		float bank_sm = 0.f;
		// Arc-length table for constant-speed flight (33 samples over t in
		// [0, 1]); rebuilt on every respawn and every chained segment.
		std::array<float, 33> arclen{};
		float total_len = 0.f;
		// Distance flown along the path (arc-length walk) and timestamp of
		// the last speed update, for climb-coupled airspeed.
		float dist_flown = 0.f;
		XrTime prev_now = 0;
		// Engine trail history (world space, newest last) for the ribbon,
		// with sample timestamps so the fade keys off real age.
		std::vector<glm::vec3> trail;
		std::vector<XrTime> trail_t;
		XrTime last_sample_t = 0;
		// Current plane fade, applied to the trail so it fades with the body.
		float trail_alpha = 0.f;
		// Aerobatic maneuver for this flight. Rolls/knife are orientation
		// overlays on top of the bank; the loop is a true vertical circle
		// flown off the frozen path (position and nose track the circle,
		// rejoining with zero pop). Exactly one per flight; heading always
		// returns to the path heading (heading-reversing figures would
		// fly tail-first).
		enum class maneuver
		{
			none,
			barrel,  // lazy 360 about the nose (~2.2 s)
			aileron, // snappy 360 about the nose (~1.2 s)
			knife,   // eased hold near 90 deg bank, then back level
			loop,    // true circle in the entry vertical plane
		} man = maneuver::none;
		// Rolls/knife run on distance windows (fraction of total_len);
		// the loop starts at a distance mark and exits on θ = 2π.
		double man_center = 0.;
		double man_width = 1.;
		float man_dir = 1.f; // roll/knife direction
		// True-loop state: scheduled start (fraction of flight), randomized
		// shape (radius + apex speed fraction), live integration state, and
		// the entry frame captured on the first in-loop frame. Entry bank
		// rides along, eases out over the top, and is restored at exit so
		// the handoff never pops.
		double loop_start_d = 0.; // distance fraction of total_len
		float loop_r = 0.35f, loop_top = 0.5f;
		bool loop_active = false, loop_done = false;
		double loop_theta = 0.;
		float loop_v = 0.3f;
		// Post-loop handoff: stashed closure orientation and seconds since
		// exit (< 0 while inactive). Slerps onto the live path frame.
		glm::quat loop_exit_o{1.f, 0.f, 0.f, 0.f};
		double loop_exit_t = -1.;
		// Pull-up blend: seconds since entry and its duration. Position
		// eases from the straight-line path extension onto the circle,
		// so curvature ramps 0 → 1/r instead of stepping (the nose would
		// otherwise whip upward at v/r on the entry frame).
		double loop_t = 0.;
		float loop_blend = 0.7f;
		glm::vec3 loop_p0{0.f}, loop_f{0.f, 0.f, -1.f}, loop_u{0.f, 1.f, 0.f}, loop_right{1.f, 0.f, 0.f};
		float loop_bank = 0.f;
	};

	::scene & scene_;
	std::vector<flight> flights_;
	std::mt19937 rng_;
	bool was_active_ = false;

	// Shared engine-trail ribbon: one node/primitive holding every flight's
	// camera-facing strip, rebuilt on the CPU each frame.
	entt::entity trail_entity = entt::null;
	std::shared_ptr<renderer::mesh> trail_mesh;
	std::shared_ptr<renderer::material> trail_material;
	std::shared_ptr<buffer_allocation> trail_buffer;
	// Emitter in construction-time world units (tail center of the measured
	// model OBB at origin); only the plane orientation applies per frame
	// since the node scale never changes after construction.
	glm::vec3 tail_root{0.f};

	void respawn(flight & f, XrTime now, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient);
	// Chain the next segment onto the live flight: C¹-continuous with the
	// frozen curve (same end point and tangent), steered at the live
	// anchor so the fleet meanders around a moving centre. Trail, livery
	// and lifetime are untouched — only the path, its pacing and the
	// segment's maneuver are refreshed.
	void extend(flight & f, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient);
	// Fresh segment contents: control points (chained or launched), arc
	// table, pacing and the segment's maneuver pick.
	void build_segment(flight & f, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient, bool fresh);
	// Maneuver pick for the current segment (distance-scheduled).
	void pick_maneuver(flight & f);
	static glm::vec3 bezier(const flight & f, double t);
	static glm::vec3 bezier_vel(const flight & f, double t);
	static glm::vec3 bezier_acc(const flight & f, double t);

	void build_trail(const std::optional<glm::vec3> & camera, XrTime now);
};

} // namespace wivrn
