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

#include "kiosk_planes.h"

#include "scene.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iterator>
#include <map>
#include <numbers>
#include <spdlog/spdlog.h>
#include <string>

namespace
{
// tuning knobs, all device-tunable: model is ~13.4m wingspan in file units
constexpr int plane_count = 4;
constexpr float model_scale = 0.00675f; // ~0.13m wingspan
constexpr double fade_in = 1.2;
constexpr double spawn_interval = 1.0; // plane i spawns at activation + i * this
constexpr float spawn_radius = 1.0f;   // random position within this distance of the overlay
// curvature-integrator flight tuning: modest rates keep the fleet inside the
// leash at cruise speed; the containment bias is a tendency, not a wall
constexpr float cruise_lo = 0.30f;   // per-plane constant cruise speed range (m/s)
constexpr float cruise_hi = 0.45f;
constexpr float comfort_r = 0.6f;    // bias starts past this distance from the anchor (m)
constexpr float leash_r = 1.0f;      // bias reaches full strength here (m)
constexpr float bias_gain = 2.0f;    // steering gain toward the anchor
constexpr float bias_max_rate = 0.8f; // bias steering cap (rad/s): loops (1.2+) overrule it locally
constexpr float rate_smooth = 3.0f;  // maneuver target blending rate (1/s)
constexpr float bank_coord_gain = 0.6f; // coordinated bank per yaw rate
constexpr float bank_max = 1.0f;     // bank clamp (radians)
constexpr float bank_relax = 2.0f;    // bank return rate toward coordinated value (1/s)
// Fixed det=+1 basis flips the apparent bank sense vs the old mirrored
// frame: positive bank about +dir lifts the inside wing (leans OUT), so
// the coordinated default is -1. Flip only if planes lean out of turns.
constexpr float bank_sign = -1.f;

double smoothstep(double e0, double e1, double x)
{
	double t = std::clamp((x - e0) / (e1 - e0), 0., 1.);
	return t * t * (3. - 2. * t);
}

// body paint jobs for the "Material" part, with per-channel variance;
// glass, emissive and dark/metal parts stay as authored
struct paint
{
	const char * name;
	glm::vec3 base;
	float variance;
};
const paint paints[]{
        {"white", {1.f, 1.f, 1.f}, 0.f},
        {"grey", {0.62f, 0.62f, 0.62f}, 0.12f},
        {"orange", {0.95f, 0.27f, 0.02f}, 0.06f}, // vivid dynacon orange: high red, low green so it never reads yellow
        {"navy", {0.03f, 0.06f, 0.36f}, 0.10f},   // dark #0E1B8D indigo
};
// fallback draw weights when the per-activation livery bag is exhausted
// (shouldn't happen): mostly orange, some grey and navy, white rare
const std::array<double, std::size(paints)> paint_weights{1.0, 3.0, 6.0, 2.0};

glm::vec3 random_in_ball(std::mt19937 & rng, float radius)
{
	// rejection sampling: uniform in the ball, scaled to radius
	std::uniform_real_distribution<float> dist(-1.f, 1.f);
	for (;;)
	{
		glm::vec3 v(dist(rng), dist(rng), dist(rng));
		float l2 = glm::dot(v, v);
		if (l2 <= 1.f && l2 >= 1e-6f)
			return v * radius;
	}
}
} // namespace

namespace wivrn
{

kiosk_planes::kiosk_planes(::scene & scene, uint32_t layer_mask) :
        scene_(scene), rng_(std::random_device{}())
{
	for (int i = 0; i < plane_count; ++i)
	{
		auto && [entity, node] = scene_.add_gltf("assets://plane.glb", layer_mask);
		node.name = "kiosk-plane-" + std::to_string(i);
		node.scale = {model_scale, model_scale, model_scale};
		node.visible = false;

		// one material clone per source material so each plane fades
		// independently; textures stay shared, only flags and the uniform
		// staging are per-plane. The white body part gets a per-plane
		// paint job from the palette above.
		plane f;
		f.root = entity;
		const paint & paint = paints[std::uniform_int_distribution<size_t>(0, std::size(paints) - 1)(rng_)];
		std::uniform_real_distribution<float> jitter(-paint.variance, paint.variance);
		const glm::vec3 body_tint{
		        std::clamp(paint.base.x + jitter(rng_), 0.f, 1.f),
		        std::clamp(paint.base.y + jitter(rng_), 0.f, 1.f),
		        std::clamp(paint.base.z + jitter(rng_), 0.f, 1.f),
		};
		// add_gltf shares mesh objects (and their materials) across
		// instantiations: deep-copy each mesh so per-plane material clones
		// don't leak into the other planes. GPU buffers stay shared.
		std::map<std::shared_ptr<renderer::mesh>, std::shared_ptr<renderer::mesh>> mesh_copies;
		std::map<std::shared_ptr<renderer::material>, std::shared_ptr<renderer::material>> clones;
		std::function<void(entt::entity)> visit = [&](entt::entity e) {
			if (auto * n = scene_.world.try_get<components::node>(e))
			{
				if (n->mesh)
				{
					auto [mesh_it, mesh_new] = mesh_copies.try_emplace(n->mesh, nullptr);
					if (mesh_new)
					{
						mesh_it->second = std::make_shared<renderer::mesh>();
						mesh_it->second->buffer = n->mesh->buffer;
						mesh_it->second->primitives = n->mesh->primitives;
						for (auto & primitive: mesh_it->second->primitives)
						{
							auto [it, inserted] = clones.try_emplace(primitive.material_, nullptr);
							if (inserted)
							{
								auto src = primitive.material_;
								it->second = scene_.create_material([src, body_tint](renderer::material & m) {
									m.staging = src->staging;
									m.base_color_texture = src->base_color_texture;
									m.metallic_roughness_texture = src->metallic_roughness_texture;
									m.occlusion_texture = src->occlusion_texture;
									m.emissive_texture = src->emissive_texture;
									m.normal_texture = src->normal_texture;
									m.double_sided = src->double_sided;
									m.fragment_shader = src->fragment_shader;
									m.name = src->name + "-kiosk";
									if (src->name == "Material")
										m.staging.base_color_factor = glm::vec4(body_tint, 1.f);
									m.blend_enable = true;
									m.depth_write_enable = false;
								});
								f.materials.push_back(it->second);
								f.base_colors.push_back(it->second->staging.base_color_factor);
								f.base_emissives.push_back(glm::vec3(it->second->staging.base_emissive_factor));
							}
							primitive.material_ = it->second;
						}
					}
					n->mesh = mesh_it->second;
				}
				for (auto && [child, child_node]: scene_.world.view<components::node>().each())
				{
					if (child_node.parent == e)
						visit(child);
				}
			}
		};
		visit(entity);
		planes_.push_back(std::move(f));
	}
}

void kiosk_planes::spawn_at(plane & f, const glm::vec3 & anchor_pos)
{
	// random position in a ball around the overlay; frozen here so later
	// panel motion never perturbs a spawned plane
	glm::vec3 pos = anchor_pos + random_in_ball(rng_, spawn_radius);
	pos.y = std::clamp(pos.y, anchor_pos.y - 0.6f, anchor_pos.y + 0.9f);

	auto * node = scene_.world.try_get<components::node>(f.root);
	if (not node)
		return;
	node->position = pos;
	// level random heading; model nose points along +Z
	std::uniform_real_distribution<float> yaw_dist(0.f, 2.f * std::numbers::pi_v<float>);
	float yaw0 = yaw_dist(rng_);
	node->orientation = glm::angleAxis(yaw0, glm::vec3(0.f, 1.f, 0.f));
	// flight state: level cruise on the spawn heading, maneuver picked on
	// the first integrated frame (man_t = 0)
	f.pos = pos;
	// match the spawn orientation above: identity nose +Z rotated by yaw0
	f.dir = glm::vec3(std::sin(yaw0), 0.f, std::cos(yaw0));
	// matching lateral axis (up x dir); rebuilt on the first frame anyway
	f.right = glm::vec3(std::cos(yaw0), 0.f, -std::sin(yaw0));
	f.speed = std::uniform_real_distribution<float>(cruise_lo, cruise_hi)(rng_);
	f.bank = 0.f;
	f.pitch_rate = f.yaw_rate = f.roll_rate = 0.f;
	f.pitch_tgt = f.yaw_tgt = f.roll_tgt = 0.f;
	f.man_t = 0.;
	f.prev_now = 0;
	// fresh livery every spawn: construction-time draws could hide a color
	// all session. The spawn is invisible (pre-fade), so the swap never pops.
	// Logcat records the pick for on-device color forensics.
	apply_livery(f);
	f.spawned = true;
}

void kiosk_planes::apply_livery(plane & f)
{
	// per-activation bag first (guaranteed variety), weighted draw fallback
	// (shouldn't happen: bag covers the fleet)
	const paint & p = livery_next_ < livery_bag_.size()
	        ? paints[livery_bag_[livery_next_++]]
	        : paints[std::discrete_distribution<size_t>(paint_weights.begin(), paint_weights.end())(rng_)];
	std::uniform_real_distribution<float> jitter(-p.variance, p.variance);
	const glm::vec3 body_tint{
	        std::clamp(p.base.x + jitter(rng_), 0.f, 1.f),
	        std::clamp(p.base.y + jitter(rng_), 0.f, 1.f),
	        std::clamp(p.base.z + jitter(rng_), 0.f, 1.f),
	};
	for (size_t i = 0; i < f.materials.size(); ++i)
	{
		if (f.materials[i]->name == "Material-kiosk")
		{
			f.materials[i]->staging.base_color_factor = glm::vec4(body_tint, 1.f);
			f.base_colors[i] = glm::vec4(body_tint, 1.f);
		}
	}
	spdlog::info("kiosk_planes: spawn livery {}", p.name);
}

void kiosk_planes::pick_maneuver(plane & f)
{
	// maneuvers are just rate targets for the shared integrator: cruise is
	// near-zero curvature, a turn is sustained yaw (bank follows), a loop is
	// a full 2pi of pitch, a roll is a full 2pi of roll with a breath of yaw.
	// Durations complete the figure exactly, then a new target is drawn.
	static const std::array<double, 4> weights{5.0, 3.0, 1.0, 1.0};
	auto coin = [&] { return std::uniform_real_distribution<float>(0.f, 1.f)(rng_) < 0.5f ? -1.f : 1.f; };
	switch (std::discrete_distribution<int>(weights.begin(), weights.end())(rng_))
	{
		case 0: // cruise
		default:
			f.yaw_tgt = coin() * std::uniform_real_distribution<float>(0.1f, 0.3f)(rng_);
			f.pitch_tgt = coin() * std::uniform_real_distribution<float>(0.f, 0.15f)(rng_);
			f.roll_tgt = 0.f;
			f.man_t = std::uniform_real_distribution<double>(2., 4.)(rng_);
			break;
		case 1: // banked turn (bank emerges from yaw via the coordination below)
			f.yaw_tgt = coin() * std::uniform_real_distribution<float>(0.6f, 1.2f)(rng_);
			f.pitch_tgt = coin() * std::uniform_real_distribution<float>(0.f, 0.2f)(rng_);
			f.roll_tgt = 0.f;
			f.man_t = std::uniform_real_distribution<double>(1.5, 3.)(rng_);
			break;
		case 2: // loop: sustained pitch for exactly one revolution
			f.yaw_tgt = 0.f;
			f.pitch_tgt = coin() * std::uniform_real_distribution<float>(1.2f, 1.8f)(rng_);
			f.roll_tgt = 0.f;
			f.man_t = 2. * std::numbers::pi / std::abs(f.pitch_tgt);
			break;
		case 3: // barrel roll: one full revolution with a hint of yaw
			f.yaw_tgt = coin() * std::uniform_real_distribution<float>(0.1f, 0.3f)(rng_);
			f.pitch_tgt = 0.f;
			f.roll_tgt = coin() * float(2. * std::numbers::pi / std::uniform_real_distribution<double>(1.2, 2.)(rng_));
			f.man_t = 2. * std::numbers::pi / std::abs(f.roll_tgt);
			break;
	}
}

void kiosk_planes::update(XrTime now, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient, bool active)
{
	(void)anchor_orient; // placement is position-only; heading is random
	if (active && !was_active_)
	{
		// (re)opening: replay the one-by-one spawn-in from scratch, with a
		// fresh shuffled livery bag (one of each paint, orange guaranteed)
		activated_t0_ = now;
		for (auto & f: planes_)
		{
			f.spawned = false;
			f.spawn_t0 = 0;
		}
		livery_bag_ = {2, 1, 3, 0};
		std::shuffle(livery_bag_.begin(), livery_bag_.end(), rng_);
		livery_next_ = 0;
	}
	was_active_ = active;

	for (size_t slot = 0; slot < planes_.size(); ++slot)
	{
		plane & f = planes_[slot];
		auto * node = scene_.world.try_get<components::node>(f.root);
		if (not node)
			continue;
		if (not active)
		{
			node->visible = false;
			continue;
		}

		XrTime due = activated_t0_ + XrDuration(slot * spawn_interval * 1e9);
		if (now < due)
		{
			node->visible = false;
			continue;
		}
		if (not f.spawned)
		{
			spawn_at(f, anchor_pos);
			f.spawn_t0 = due;
		}
		node->visible = true;

		// eternal: fade in once, then stay at full opacity forever
		double age = double(now - f.spawn_t0) * 1e-9;
		if (age < 0.)
			age = 0.;
		float alpha = float(smoothstep(0., fade_in, age));
		for (size_t i = 0; i < f.materials.size(); ++i)
		{
			auto & material = f.materials[i];
			// the transparent pipeline blends premultiplied (ONE,
			// ONE_MINUS_SRC_ALPHA) while lit.frag outputs straight alpha,
			// which made alpha fades no-ops: premultiply on the CPU
			material->staging.base_color_factor = glm::vec4(glm::vec3(f.base_colors[i]) * alpha, alpha);
			material->staging.base_emissive_factor = glm::vec4(f.base_emissives[i] * alpha, 0.f);
			std::memcpy(material->buffer->map(), &material->staging, sizeof(material->staging));
			material->buffer->unmap();
		}

		// curvature-integrator flight: advance the frozen spawn state along
		// the current maneuver targets, biased weakly home on overshoot
		double dt = 0.;
		if (f.prev_now != 0)
			dt = std::clamp(double(now - f.prev_now) * 1e-9, 0., 0.05);
		f.prev_now = now;
		if (dt <= 0.)
			continue;

		f.man_t -= dt;
		if (f.man_t <= 0.)
			pick_maneuver(f);

		const float blend = std::min(1.f, float(dt * rate_smooth));
		f.pitch_rate += (f.pitch_tgt - f.pitch_rate) * blend;
		f.yaw_rate += (f.yaw_tgt - f.yaw_rate) * blend;
		f.roll_rate += (f.roll_tgt - f.roll_rate) * blend;

		// moth-to-light: steer toward the live anchor, stronger the further
		// past comfort. Capped well below loop pitch rates so figures can
		// range out; a tendency, never a wall.
		float yaw_bias = 0.f, pitch_bias = 0.f;
		glm::vec3 to_anchor = anchor_pos - f.pos;
		float dist = glm::length(to_anchor);
		if (dist > comfort_r && dist > 1e-4f)
		{
			glm::vec3 want = to_anchor / dist;
			glm::vec3 r = f.right;
			if (std::abs(f.dir.y) < 0.98f)
				r = glm::normalize(glm::cross(glm::vec3{0.f, 1.f, 0.f}, f.dir));
			glm::vec3 u = glm::cross(f.dir, r);
			glm::vec3 corr = want - f.dir * glm::dot(want, f.dir);
			float w = std::min(1.f, (dist - comfort_r) / (leash_r - comfort_r));
			yaw_bias = std::clamp(bias_gain * glm::dot(corr, r), -bias_max_rate, bias_max_rate) * w;
			pitch_bias = std::clamp(bias_gain * glm::dot(corr, u), -bias_max_rate, bias_max_rate) * w;
		}
		float yaw = f.yaw_rate + yaw_bias;
		float pitch = f.pitch_rate + pitch_bias;

		// Flight frame uses right = up×dir throughout. The intuitive dir×up
		// order mirrors the orientation basis below to det=-1, and quat_cast
		// of that is garbage (squish/shrink, nose off velocity) — so this
		// order is load-bearing, not aesthetic. Re-orthogonalized every
		// valid frame (loops go vertical; the cache only covers those).
		if (std::abs(f.dir.y) < 0.98f)
		{
			f.right = glm::normalize(glm::cross(glm::vec3{0.f, 1.f, 0.f}, f.dir));
			f.right = glm::normalize(f.right - f.dir * glm::dot(f.right, f.dir));
		}
		glm::vec3 up0 = glm::cross(f.dir, f.right);
		f.dir = glm::normalize(f.dir + (f.right * yaw + up0 * pitch) * float(dt));
		// re-orthogonalize against the FINAL dir: the cache above is
		// perpendicular to the pre-integration dir, off by yaw*dt. Without
		// this the orientation basis below carries shear into quat_cast.
		f.right = glm::normalize(f.right - f.dir * glm::dot(f.right, f.dir));
		f.pos += f.dir * f.speed * float(dt);

		// rolls integrate; everything relaxes toward the coordinated bank
		// for the current yaw rate. Wrap first: a finished roll is 2pi,
		// which is 0 — relaxing unwrapped would spin all the way back.
		float bank_tgt = std::clamp(yaw * bank_coord_gain, -bank_max, bank_max);
		f.bank += f.roll_rate * float(dt);
		f.bank = std::atan2(std::sin(f.bank), std::cos(f.bank));
		f.bank += (bank_tgt - f.bank) * std::min(1.f, float(dt * bank_relax));

		node->position = f.pos;
		// Orientation basis, same convention as the flight frame: columns
		// (up×dir, dir×right, dir) have det=+1, a valid rotation mapping
		// local +Z (nose) onto dir. Built from the final dir so the frame
		// is exactly orthonormal — a stale axis here shears the model.
		glm::vec3 basis_up = glm::cross(f.dir, f.right);
		glm::quat orient = glm::quat_cast(glm::mat3(f.right, basis_up, f.dir));
		node->orientation = glm::angleAxis(bank_sign * f.bank, f.dir) * orient;
	}
}

} // namespace wivrn
