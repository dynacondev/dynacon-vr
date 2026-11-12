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
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iterator>
#include <limits>
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
constexpr float comfort_r = 0.3f;    // bias starts past this distance from the anchor (m)
constexpr float leash_r = 0.5f;      // bias reaches full strength here (m)
constexpr float bias_gain = 2.0f;    // steering gain toward the anchor
constexpr float bias_max_rate = 0.8f; // bias steering cap (rad/s): loops (1.2+) overrule it locally
constexpr float pitch_max_rate = 0.35f; // climb/dive cap (rad/s); loops are exempt
constexpr float max_climb_sin = 0.55f; // flight-path angle cap outside loops (~33deg): rate caps alone still walk the nose vertical over seconds
constexpr float climb_factor = 0.5f; // speed follows climb angle: 0.5x uphill, 1.5x downhill
constexpr float speed_min = 0.18f;   // effective speed clamp (m/s)
constexpr float speed_max = 0.65f;
constexpr float rate_smooth = 3.0f;  // pitch/yaw target blending rate (1/s); roll snaps (visual only)
constexpr float bank_coord_gain = 0.6f; // coordinated bank per yaw rate
constexpr float bank_max = 1.0f;     // bank clamp (radians)
constexpr float bank_relax = 5.0f;    // bank return rate toward coordinated value (1/s): tracks, not chases
// formation flying: leader flies solo maneuvers, the rest pursue slots
constexpr double form_gap_lo = 12.0; // next formation starts 12-20s after activation...
constexpr double form_gap_hi = 20.0;
constexpr double form_cool_lo = 15.0; // ...and 15-35s after each breakup
constexpr double form_cool_hi = 35.0;
constexpr double form_dur_lo = 3.0; // formation holds 3-10s
constexpr double form_dur_hi = 10.0;
constexpr float form_gain = 3.0f;    // slot-pursuit steering gain (hotter than the leash)
constexpr float form_max_rate = 1.5f; // slot-pursuit rate cap (rad/s)
constexpr float form_catchup = 0.8f; // speed boost per meter of along-track error
constexpr float form_speed_max = 0.85f; // catch-up cap (m/s)
// Fixed det=+1 basis flips the apparent bank sense vs the old mirrored
// frame: positive bank about +dir lifts the inside wing (leans OUT), so
// the coordinated default is -1. Flip only if planes lean out of turns.
constexpr float bank_sign = -1.f;

// Engine trail ribbon: resampled while moving, fades out by age.
constexpr size_t trail_points = 60;
constexpr double trail_sample_dt = 1. / 30.;
constexpr double trail_max_age = trail_points * trail_sample_dt;
constexpr float trail_width_head = 0.02f;  // ribbon half-width at the engine (m)
constexpr float trail_width_tail = 0.055f; // ... at the faded end
constexpr float trail_alpha_max = 0.1425f; // 5% under the original 0.15
constexpr float trail_gap = 0.05f; // emitter hangs behind the tail along the flight direction

struct trail_vertex
{
	glm::vec4 position;
	glm::vec4 color;
};

constexpr size_t trail_verts_per_plane = trail_points * 5;
constexpr size_t trail_indices_per_plane = (trail_points - 1) * 4 * 6;
constexpr size_t trail_max_verts = size_t(plane_count) * trail_verts_per_plane;
constexpr size_t trail_max_indices = size_t(plane_count) * trail_indices_per_plane;
constexpr size_t trail_vert_bytes = trail_max_verts * sizeof(trail_vertex);

// formation slot offsets in the leader frame (right, up, fwd), meters;
// slot 0 is the leader (origin). Echelon sides are coined at pick time.
static const std::vector<std::vector<glm::vec3>> formation_shapes = {
        {{0, 0, 0}, {-0.35f, 0, -0.30f}}, // pair echelon left
        {{0, 0, 0}, {0.35f, 0, -0.30f}}, // pair echelon right
        {{0, 0, 0}, {0, 0.05f, -0.45f}}, // pair line astern
        {{0, 0, 0}, {-0.35f, 0, -0.30f}, {0.35f, 0, -0.30f}}, // vic
        {{0, 0, 0}, {-0.35f, 0, -0.30f}, {-0.70f, 0, -0.60f}}, // echelon3 left
        {{0, 0, 0}, {0.35f, 0, -0.30f}, {0.70f, 0, -0.60f}}, // echelon3 right
        {{0, 0, 0}, {0, 0.05f, -0.45f}, {0, 0.10f, -0.90f}}, // astern3
        {{0, 0, 0}, {-0.35f, 0, -0.30f}, {0.35f, 0, -0.30f}, {0, 0.10f, -0.65f}}, // diamond
        {{0, 0, 0}, {-0.35f, 0, -0.30f}, {0.40f, 0, -0.30f}, {0.75f, 0, -0.60f}}, // finger-four
        {{0, 0, 0}, {-0.35f, 0, -0.30f}, {-0.70f, 0, -0.60f}, {-1.05f, 0, -0.90f}}, // echelon4 left
        {{0, 0, 0}, {0.35f, 0, -0.30f}, {0.70f, 0, -0.60f}, {1.05f, 0, -0.90f}}, // echelon4 right
        {{0, 0, 0}, {0, 0.05f, -0.45f}, {0, 0.10f, -0.90f}, {0, 0.15f, -1.35f}}, // astern4
};

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

kiosk_planes::kiosk_planes(::scene & scene, uint32_t layer_mask, uint32_t trail_layer_mask) :
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
		f.trail_tint = body_tint; // overwritten by apply_livery at spawn
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
									// NOTE: depth_write stays enabled (opaque-style depth):
									// disabling it here made the models render X-ray
									// once the flag started being honored. Fades still
									// blend correctly via premultiplied alpha.
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

	// Measure the model OBB in plane-root space for the trail emitter: walk
	// the first plane's subtree accumulating node transforms (the file's
	// own rotations live on child nodes, so the raw primitive OBBs alone
	// point the wrong way).
	glm::vec3 obb_min{std::numeric_limits<float>::max()};
	glm::vec3 obb_max{std::numeric_limits<float>::lowest()};
	std::function<void(entt::entity, const glm::mat4 &)> measure = [&](entt::entity e, const glm::mat4 & parent) {
		const auto & n = scene_.world.get<components::node>(e);
		glm::mat4 m = parent * (glm::translate(glm::mat4(1), n.position) * glm::mat4(n.orientation) * glm::scale(glm::mat4(1), n.scale));
		if (n.mesh)
		{
			for (auto & prim: n.mesh->primitives)
			{
				for (int i = 0; i < 8; ++i)
				{
					glm::vec3 corner{
					        (i & 1) ? prim.obb_max.x : prim.obb_min.x,
					        (i & 2) ? prim.obb_max.y : prim.obb_min.y,
					        (i & 4) ? prim.obb_max.z : prim.obb_min.z,
					};
					glm::vec3 w = m * glm::vec4(corner, 1);
					obb_min.x = std::min(obb_min.x, w.x);
					obb_min.y = std::min(obb_min.y, w.y);
					obb_min.z = std::min(obb_min.z, w.z);
					obb_max.x = std::max(obb_max.x, w.x);
					obb_max.y = std::max(obb_max.y, w.y);
					obb_max.z = std::max(obb_max.z, w.z);
				}
			}
		}
		for (auto && [child, child_node]: scene_.world.view<components::node>().each())
		{
			if (child_node.parent == e)
				measure(child, m);
		}
	};
	measure(planes_.front().root, glm::mat4(1));
	// Nose points along +Z: emit at the tail center.
	tail_root = {(obb_min.x + obb_max.x) * 0.5f, (obb_min.y + obb_max.y) * 0.5f, obb_min.z};

	// Shared trail ribbon: one node/primitive holding every plane's
	// camera-facing strip (debug_draw shaders: unlit position+color).
	trail_entity = scene_.world.create();
	auto & trail_node = scene_.world.emplace<components::node>(trail_entity);
	trail_node.name = "kiosk-trails";
	trail_node.layer_mask = trail_layer_mask;
	trail_node.visible = false;

	trail_mesh = std::make_shared<renderer::mesh>();
	auto & prim = trail_mesh->primitives.emplace_back();
	prim.indexed = true;
	prim.index_type = vk::IndexType::eUint16;
	prim.index_offset = trail_vert_bytes;
	prim.topology = vk::PrimitiveTopology::eTriangleList;
	prim.cull_mode = vk::CullModeFlagBits::eNone;
	prim.vertex_shader = "debug_draw.vert";
	prim.layout.add_vertex_attribute("Position", vk::Format::eR32G32B32A32Sfloat, 0, 0);
	prim.layout.add_vertex_attribute("Color", vk::Format::eR32G32B32A32Sfloat, 0, 1);
	prim.obb_min = {-10.f, -10.f, -10.f};
	prim.obb_max = {10.f, 10.f, 10.f};
	prim.material_ = scene_.create_material([](renderer::material & m) {
		m.name = "kiosk-trail";
		m.fragment_shader = "debug_draw.frag";
		m.blend_enable = true;
		// Additive: overlapping trail segments sum order-independently
		// instead of slicing each other out with sorted over-blends.
		m.blend_additive = true;
		m.depth_write_enable = false;
	});

	trail_buffer = std::make_shared<buffer_allocation>(
	        scene_.device,
	        vk::BufferCreateInfo{
	                .size = trail_vert_bytes + trail_max_indices * sizeof(uint16_t),
	                .usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer,
	        },
	        VmaAllocationCreateInfo{
	                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
	        },
	        "kiosk_planes::trails");

	// Fixed strip topology: one independent 5-wide quad strip per plane.
	// The across-width alpha profile (0 at the edges) makes the ribbon
	// soft on all sides; length fade is written per-vertex each frame.
	auto * index = reinterpret_cast<uint16_t *>(trail_buffer->map()) + trail_vert_bytes / sizeof(uint16_t);
	for (int p = 0; p < plane_count; ++p)
	{
		for (size_t j = 0; j < trail_points - 1; ++j)
		{
			for (size_t k = 0; k < 4; ++k)
			{
				uint16_t a = uint16_t(p * trail_verts_per_plane + 5 * j + k);
				*index++ = a;
				*index++ = a + 1;
				*index++ = a + 5;
				*index++ = a + 1;
				*index++ = a + 6;
				*index++ = a + 5;
			}
		}
	}

	prim.vertex_offset.push_back(0);
	prim.vertex_count = uint32_t(trail_max_verts);
	prim.index_count = uint32_t(trail_max_indices);
	trail_mesh->buffer = trail_buffer;
	trail_node.mesh = trail_mesh;
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
	f.eff_speed = f.speed;
	f.bank = 0.f;
	f.pitch_rate = f.yaw_rate = f.roll_rate = 0.f;
	f.pitch_tgt = f.yaw_tgt = f.roll_tgt = 0.f;
	f.man_t = 0.;
	f.prev_now = 0;
	// fresh flight state below; clear any stale trail so a re-show never
	// streaks across the scene from the old position
	f.trail.clear();
	f.trail_t.clear();
	f.last_sample_t = 0;
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
	// trails follow the livery body color
	f.trail_tint = body_tint;
	spdlog::info("kiosk_planes: spawn livery {}", p.name);
}

void kiosk_planes::start_formation(XrTime now)
{
	// gather the spawned fleet; need at least a pair
	std::vector<size_t> candidates;
	for (size_t i = 0; i < planes_.size(); ++i)
	{
		if (planes_[i].spawned)
			candidates.push_back(i);
	}
	if (candidates.size() < 2)
	{
		next_formation_t_ = now + XrDuration(5. * 1e9); // retry soon
		return;
	}
	std::shuffle(candidates.begin(), candidates.end(), rng_);
	size_t count = std::uniform_int_distribution<size_t>(2, candidates.size())(rng_);
	// shapes valid for this headcount only
	std::vector<int> shapes;
	for (size_t s = 0; s < formation_shapes.size(); ++s)
	{
		if (formation_shapes[s].size() == count)
			shapes.push_back(int(s));
	}
	if (shapes.empty())
	{
		next_formation_t_ = now + XrDuration(5. * 1e9);
		return;
	}
	formation_shape_ = shapes[std::uniform_int_distribution<size_t>(0, shapes.size() - 1)(rng_)];
	formation_leader_ = int(candidates[0]);
	for (size_t i = 0; i < planes_.size(); ++i)
		planes_[i].formation_slot = -1;
	planes_[formation_leader_].formation_slot = 0;
	for (size_t k = 1; k < count; ++k)
	{
		plane & f = planes_[candidates[k]];
		f.formation_slot = int(k);
		// freeze solo state: no maneuver timers, no relax fight, no leash.
		// Rates decay out below; the slot blend slews steering in.
		f.loop_maneuver = false;
		f.roll_free = false;
		f.roll_rate = 0.f;
		f.form_w = 0.f;
	}
	formation_active_ = true;
	formation_end_t_ = now + XrDuration(std::uniform_real_distribution<double>(form_dur_lo, form_dur_hi)(rng_) * 1e9);
	spdlog::info("kiosk_planes: formation shape {} with {} planes (leader {})", formation_shape_, count, formation_leader_);
}

void kiosk_planes::release_formation()
{
	for (auto & f: planes_)
	{
		if (f.formation_slot < 0)
			continue;
		f.formation_slot = -1;
		if (f.spawned)
			pick_maneuver(f); // fresh enveloped maneuver: smooth by construction
	}
	formation_active_ = false;
	formation_leader_ = -1;
	formation_shape_ = -1;
	next_formation_t_ = 0; // rearmed by the caller with now
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
			f.roll_free = false;
			f.loop_maneuver = false;
			f.man_t = f.man_dur = std::uniform_real_distribution<double>(2., 4.)(rng_);
			break;
		case 1: // banked turn (bank emerges from yaw via the coordination below)
			f.yaw_tgt = coin() * std::uniform_real_distribution<float>(0.6f, 1.2f)(rng_);
			f.pitch_tgt = coin() * std::uniform_real_distribution<float>(0.f, 0.2f)(rng_);
			f.roll_tgt = 0.f;
			f.roll_free = false;
			f.loop_maneuver = false;
			f.man_t = f.man_dur = std::uniform_real_distribution<double>(1.5, 3.)(rng_);
			break;
		case 2: // loop: sustained pitch for exactly one revolution, always
		        // nose-up (inside loop only — never bunt/dive into it)
			f.yaw_tgt = 0.f;
			f.pitch_tgt = std::uniform_real_distribution<float>(1.2f, 1.8f)(rng_);
			f.roll_tgt = 0.f;
			f.roll_free = false;
			f.loop_maneuver = true;
			f.man_dur = 2. * std::numbers::pi / f.pitch_tgt;
			f.man_t = 2.0 * f.man_dur; // backstop only; the angle test below terminates first
			// planar entry: kill any yaw carryover (and the leash pull) so
			// the figure starts in-plane. A decaying yaw would tilt the
			// loop, and the exit frame would then snap to the tilted plane.
			// The stop-then-pull-up reads as a deliberate entry, and the
			// envelope eases the pitch in from here.
			f.yaw_rate = 0.f;
			f.bias_w = 0.f;
			break;
		case 3: // barrel roll: one full revolution with a hint of yaw
			f.yaw_tgt = coin() * std::uniform_real_distribution<float>(0.1f, 0.3f)(rng_);
			f.pitch_tgt = 0.f;
			f.roll_tgt = coin() * float(2. * std::numbers::pi / std::uniform_real_distribution<double>(1.2, 2.)(rng_));
			f.roll_free = true;
			f.loop_maneuver = false;
			f.man_dur = 2. * std::numbers::pi / std::abs(f.roll_tgt);
			f.man_t = 2.0 * f.man_dur; // backstop only; the angle test below terminates first
			break;
	}
	f.man_age = 0.;
	f.man_angle = 0.;
}

void kiosk_planes::update(XrTime now, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient, bool active, const std::optional<glm::vec3> & camera)
{
	(void)anchor_orient; // placement is position-only; heading is random
	// the below-GUI passes always include the trails layer: hide the ribbon
	// here, the per-plane loop below only runs its visible branch on active
	if (auto * trail_node = scene_.world.try_get<components::node>(trail_entity))
		trail_node->visible = active;
	if (active && !was_active_)
	{
		// (re)opening: replay the one-by-one spawn-in from scratch, with a
		// fresh shuffled livery bag (one of each paint, orange guaranteed)
		activated_t0_ = now;
		for (auto & f: planes_)
		{
			f.spawned = false;
			f.spawn_t0 = 0;
			f.bias_w = 1.f;
			f.formation_slot = -1;
		}
		livery_bag_ = {2, 1, 3, 0};
		std::shuffle(livery_bag_.begin(), livery_bag_.end(), rng_);
		livery_next_ = 0;
		formation_active_ = false;
		formation_leader_ = -1;
		formation_shape_ = -1;
		next_formation_t_ = now + XrDuration(std::uniform_real_distribution<double>(form_gap_lo, form_gap_hi)(rng_) * 1e9);
	}
	was_active_ = active;

	// formation scheduler: breakup on schedule, start when due
	if (active && formation_active_ && now >= formation_end_t_)
	{
		release_formation();
		next_formation_t_ = now + XrDuration(std::uniform_real_distribution<double>(form_cool_lo, form_cool_hi)(rng_) * 1e9);
	}
	if (active && !formation_active_ && now >= next_formation_t_)
		start_formation(now);

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

		const bool slaved = formation_active_ && f.formation_slot > 0;
		if (not slaved)
		{
			f.man_t -= dt;
			f.man_age += dt;
			// figures terminate on completed angle (2pi minus a breath — the
			// residual at threshold is single-digit degrees, then the next
			// figure ramps in) or once the exit ramp has decayed the rate to
			// ~zero; the backstop only catches a figure that somehow stalls.
			// Time-based maneuvers terminate on the clock.
			bool figure = f.loop_maneuver || f.roll_free;
			bool done = f.man_t <= 0.;
			if (not done && figure)
			{
				double rate_now = f.loop_maneuver ? std::abs(f.pitch_rate) : std::abs(f.roll_rate);
				double rate_tgt = f.loop_maneuver ? std::abs(f.pitch_tgt) : std::abs(f.roll_tgt);
				done = f.man_angle >= 2. * std::numbers::pi - 0.2 ||
				       (f.man_age > f.man_dur && rate_now < 0.08 * rate_tgt);
			}
			if (done)
				pick_maneuver(f);

			// maneuver envelope: smoothstep in and out of the targets so figures
			// join through near-zero curvature with continuous derivatives — no
			// straight-to-curve jump at either end. Ramps clamp to a fraction of
			// short figures. Figures ramps out by angle (a timed exit would
			// stall the rate before the 2pi closes); time-based maneuvers ramp
			// out by the clock. The bias stays outside the envelope (smooth).
			double ramp = std::min(0.7, 0.35 * f.man_dur);
			float env_in = float(smoothstep(0., ramp, f.man_age));
			float env_out;
			double nominal_rate = 0.;
			if (f.loop_maneuver)
				nominal_rate = std::abs(f.pitch_tgt);
			else if (f.roll_free)
				nominal_rate = std::abs(f.roll_tgt);
			if (nominal_rate > 0.)
				env_out = 1.f - float(smoothstep(2. * std::numbers::pi - nominal_rate * ramp, 2. * std::numbers::pi, f.man_angle));
			else
				env_out = 1.f - float(smoothstep(f.man_dur - ramp, f.man_dur, f.man_age));
			float env = env_in * env_out;
			const float blend = std::min(1.f, float(dt * rate_smooth));
			f.pitch_rate += (f.pitch_tgt * env - f.pitch_rate) * blend;
			f.yaw_rate += (f.yaw_tgt * env - f.yaw_rate) * blend;
			f.roll_rate = f.roll_tgt * env; // visual only: envelope spins rolls up/down, no snap
			if (f.loop_maneuver)
				f.man_angle += f.pitch_rate * float(dt);
			else if (f.roll_free)
				f.man_angle += std::abs(f.roll_rate) * float(dt);
		}
		else
		{
			// slaved follower: maneuver timers frozen, slot steering slewed
			// in instead. Rates are set directly below (capped), eased by
			// form_w — the join transient reads as a deliberate rejoin.
			f.form_w += (1.f - f.form_w) * std::min(1.f, float(dt * 2.5f));
		}

		// Flight frame, rebuilt every frame from the current dir: right =
		// up×dir throughout (the intuitive dir×up order mirrors the
		// orientation basis below to det=-1, and quat_cast of that is
		// garbage — so this order is load-bearing, not aesthetic).
		// Loops freeze the frame: the axis stays fixed for the whole
		// figure, so the loop is exactly planar about one axis and can
		// pass through vertical with no pole snap on entry or exit.
		glm::vec3 r = f.right;
		if (not f.loop_maneuver)
		{
			// NaN guard more than anything now: the attitude clamp keeps
			// normal flight far from the pole, so this branch is routine.
			if (std::abs(f.dir.y) < 0.98f)
			{
				r = glm::normalize(glm::cross(glm::vec3{0.f, 1.f, 0.f}, f.dir));
				r = glm::normalize(r - f.dir * glm::dot(r, f.dir));
			}
			f.right = r;
		}
		glm::vec3 u = glm::cross(f.dir, r);

		float yaw = 0.f, pitch = 0.f;
		glm::vec3 to_slot{0.f};
		if (not slaved)
		{
			// moth-to-light: steer toward the live anchor, stronger the further
			// past comfort; a tendency, never a wall. Fully stood down mid-loop
			// (planar figures close exactly), and slewed everywhere else: the
			// gate ramps over ~0.25s so loop boundaries never step the rate.
			float bias_gate = f.loop_maneuver ? 0.f : 1.f;
			f.bias_w += (bias_gate - f.bias_w) * std::min(1.f, float(dt * 4.0));
			float yaw_bias = 0.f, pitch_bias = 0.f;
			glm::vec3 to_anchor = anchor_pos - f.pos;
			float dist = glm::length(to_anchor);
			if (dist > comfort_r && dist > 1e-4f)
			{
				glm::vec3 want = to_anchor / dist;
				glm::vec3 corr = want - f.dir * glm::dot(want, f.dir);
				float w = std::min(1.f, (dist - comfort_r) / (leash_r - comfort_r));
				yaw_bias = std::clamp(bias_gain * glm::dot(corr, r), -bias_max_rate, bias_max_rate) * w * f.bias_w;
				pitch_bias = std::clamp(bias_gain * glm::dot(corr, u), -bias_max_rate, bias_max_rate) * w * f.bias_w;
			}
			yaw = f.yaw_rate + yaw_bias;
			pitch = f.pitch_rate + pitch_bias;
			// model planes don't rocket: cap climb/dive authority everywhere
			// except loops, which need the full pitch rate for the figure.
			if (not f.loop_maneuver)
				pitch = std::clamp(pitch, -pitch_max_rate, pitch_max_rate);
		}
		else
		{
			// slot pursuit in the leader frame: no leash (the slot may sit
			// outside it) and no maneuver targets. Steering is direct and
			// capped, eased in by form_w, so the join reads as a rejoin.
			const plane & lead = planes_[size_t(formation_leader_)];
			glm::vec3 lup = glm::cross(lead.dir, lead.right);
			glm::vec3 off = formation_shapes[size_t(formation_shape_)][size_t(f.formation_slot)];
			glm::vec3 want = lead.pos + lead.right * off.x + lup * off.y + lead.dir * off.z;
			to_slot = want - f.pos;
			if (glm::length(to_slot) > 1e-4f)
			{
				glm::vec3 corr = to_slot - f.dir * glm::dot(to_slot, f.dir);
				yaw = std::clamp(form_gain * glm::dot(corr, r), -form_max_rate, form_max_rate) * f.form_w;
				pitch = std::clamp(form_gain * glm::dot(corr, u), -form_max_rate, form_max_rate) * f.form_w;
			}
		}

		f.dir = glm::normalize(f.dir + (r * yaw + u * pitch) * float(dt));
		if (not f.loop_maneuver)
		{
			// attitude clamp, not just rate: sustained bias could still walk
			// the nose to vertical over seconds. Loops are exempt (the
			// figure needs the full circle); everything else stays under
			// max_climb_sin so the pole is never approached outside a loop.
			if (std::abs(f.dir.y) > max_climb_sin)
			{
				f.dir.y = std::copysign(max_climb_sin, f.dir.y);
				f.dir = glm::normalize(f.dir);
			}
			// re-orthogonalize against the FINAL dir: the frame above is
			// perpendicular to the pre-integration dir, off by yaw*dt.
			// Without this the orientation basis below carries shear.
			f.right = glm::normalize(r - f.dir * glm::dot(r, f.dir));
		}
		// speed follows climb angle: slower uphill, faster downhill.
		// Followers instead match the leader plus a closing term so the
		// join reads as a rejoin (hurry when behind, check when ahead).
		float speed;
		if (not slaved)
			speed = std::clamp(f.speed * (1.f - climb_factor * f.dir.y), speed_min, speed_max);
		else
		{
			const plane & lead = planes_[size_t(formation_leader_)];
			float along = glm::dot(to_slot, f.dir);
			speed = std::clamp(lead.eff_speed + form_catchup * along, speed_min, form_speed_max);
		}
		f.pos += f.dir * speed * float(dt);
		f.eff_speed = speed;

		// rolls integrate; everything relaxes toward the coordinated bank
		// for the current yaw rate. Wrap first: a finished roll is 2pi,
		// which is 0 — relaxing unwrapped would spin all the way back.
		// bank tracks the applied (post-clamp, post-bias) yaw — that is 1/R at
		// constant speed, i.e. corner-radius-driven — fast enough to read as
		// linked, not chasing. Rolls own the bank outright: no pull mid-roll.
		float bank_tgt = std::clamp(yaw * bank_coord_gain, -bank_max, bank_max);
		f.bank += f.roll_rate * float(dt);
		f.bank = std::atan2(std::sin(f.bank), std::cos(f.bank));
		if (not f.roll_free)
			f.bank += (bank_tgt - f.bank) * std::min(1.f, float(dt * bank_relax));

		node->position = f.pos;
		// Orientation basis, same convention as the flight frame: columns
		// (up×dir, dir×right, dir) have det=+1, a valid rotation mapping
		// local +Z (nose) onto dir. Built from the final dir so the frame
		// is exactly orthonormal — a stale axis here shears the model.
		glm::vec3 basis_up = glm::cross(f.dir, f.right);
		glm::quat orient = glm::quat_cast(glm::mat3(f.right, basis_up, f.dir));
		node->orientation = glm::angleAxis(bank_sign * f.bank, f.dir) * orient;

		// record the engine position for the trail ribbon: the measured
		// tail offset is already in world units at the construction
		// scale, so only the orientation applies (uniform scale). The
		// sample hangs behind the tail so the ribbon never touches the
		// body. A long stall (hidden overlay, hitches) clears the trail
		// instead of streaking across the scene on resume.
		glm::vec3 tail = f.pos + node->orientation * tail_root - f.dir * trail_gap;
		if (f.last_sample_t != 0 and double(now - f.last_sample_t) * 1e-9 > 0.5)
		{
			f.trail.clear();
			f.trail_t.clear();
		}
		if (f.last_sample_t == 0 or double(now - f.last_sample_t) * 1e-9 >= trail_sample_dt)
		{
			f.trail.push_back(tail);
			f.trail_t.push_back(now);
			if (f.trail.size() > trail_points)
			{
				f.trail.erase(f.trail.begin());
				f.trail_t.erase(f.trail_t.begin());
			}
			f.last_sample_t = now;
		}
		if (auto * trail_node = scene_.world.try_get<components::node>(trail_entity))
			trail_node->visible = active;
		f.trail_alpha = alpha;
	}

	if (active)
		build_trail(camera, now);
}

void kiosk_planes::build_trail(const std::optional<glm::vec3> & camera, XrTime now)
{
	auto * verts = reinterpret_cast<trail_vertex *>(trail_buffer->map());
	const glm::vec3 up{0.f, 1.f, 0.f};

	for (int p = 0; p < plane_count; ++p)
	{
		const plane & f = planes_[size_t(p)];
		trail_vertex * out = verts + size_t(p) * trail_verts_per_plane;

		// slot j=0 is the newest sample; pad short trails with the oldest
		// point (written at zero alpha below) so the fixed index buffer
		// always has valid data. Fade keys off real sample age, not slot
		// index, so a young trail never shows a hard end.
		glm::vec3 ring[trail_points];
		float ring_age[trail_points]{};
		const size_t n = f.trail.size();
		const bool live = n >= 2;
		for (size_t j = 0; j < trail_points; ++j)
		{
			const size_t src = live ? n - 1 - std::min(j, n - 1) : 0;
			ring[j] = live ? f.trail[src] : glm::vec3(0.f);
			ring_age[j] = live ? float(double(now - f.trail_t[src]) * 1e-9) : 0.f;
		}

		glm::vec3 last_side{1.f, 0.f, 0.f};
		for (size_t j = 0; j < trail_points; ++j)
		{
			const float t = float(j) / float(trail_points - 1); // 0 = engine, 1 = faded end
			const glm::vec3 & pt = ring[j];

			glm::vec3 dir = ring[std::min(j + 1, trail_points - 1)] - ring[j > 0 ? j - 1 : j];
			glm::vec3 side = last_side;
			if (glm::length(dir) > 1e-6f)
			{
				glm::vec3 dir_n = dir / glm::length(dir);
				glm::vec3 axis = camera ? (*camera - pt) : up;
				if (glm::length(axis) > 1e-6f)
				{
					glm::vec3 s = glm::cross(dir_n, axis / glm::length(axis));
					if (glm::length(s) > 1e-4f)
						side = s / glm::length(s);
					else
					{
						s = glm::cross(dir_n, up);
						if (glm::length(s) > 1e-4f)
							side = s / glm::length(s);
					}
				}
				last_side = side;
			}

			const float width = trail_width_head + (trail_width_tail - trail_width_head) * t;
			// gradual age falloff (long tail, not a hard end) times a head
			// ramp over the first segments, so a young trail can never show
			// a bright leading edge. The tail ramp keys off the live point
			// count (not the slot index): padding duplicates the oldest
			// sample, so without it a young trail would end in a bright
			// streak cut off at the last live point.
			const float age_frac = std::clamp(ring_age[j] / float(trail_max_age), 0.f, 1.f);
			const float envelope = std::pow(1.f - age_frac, 1.5f);
			const float head_ramp = float(smoothstep(0., 8., double(j)));
			const float tail_ramp = j < n ? 1.f - float(smoothstep(double(n > 4 ? n - 4 : 0), double(n > 1 ? n - 1 : 0), double(j))) : 0.f;
			// tint follows the plane's livery body color
			const float a = live ? trail_alpha_max * f.trail_alpha * envelope * head_ramp * tail_ramp : 0.f;
			const glm::vec3 tint = f.trail_tint * a;

			// soft across-width profile: transparent edges, bright core
			constexpr float across[5] = {-1.f, -0.5f, 0.f, 0.5f, 1.f};
			constexpr float across_alpha[5] = {0.f, 0.6f, 1.f, 0.6f, 0.f};
			for (size_t k = 0; k < 5; ++k)
			{
				const float ak = a * across_alpha[k];
				out[5 * j + k] = {
				        glm::vec4(pt + side * (across[k] * width * 0.5f), 1.f),
				        glm::vec4(tint * across_alpha[k], ak), // premultiplied
				};
			}
		}
	}
}

} // namespace wivrn
