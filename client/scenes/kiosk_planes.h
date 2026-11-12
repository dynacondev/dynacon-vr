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
// lobby and stream scenes. Clean slate: no pathing. Planes spawn one by one
// at random positions within 1m of the overlay panel, fade in, then stay
// put forever (eternal: no flight paths, no fade-outs, no respawns). Create once
// the scene world exists (on_focused), update every frame the overlay may
// show, destroy with the scene (planes hold entities in the scene registry).
class kiosk_planes
{
public:
	kiosk_planes(::scene & scene, uint32_t layer_mask);

	// anchor is the overlay panel pose sampled this frame; used for spawn
	// placement only (positions freeze once spawned). Inactive planes are
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
	};

	::scene & scene_;
	std::vector<plane> planes_;
	std::mt19937 rng_;
	bool was_active_ = false;
	XrTime activated_t0_ = 0; // rising-edge time: plane i spawns at + i * spawn_interval

	void spawn_at(plane & f, const glm::vec3 & anchor_pos);
};

} // namespace wivrn
