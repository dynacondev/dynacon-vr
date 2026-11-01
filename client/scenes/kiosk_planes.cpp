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
#include <glm/ext/matrix_transform.hpp>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <spdlog/spdlog.h>
#include <string>

namespace
{
// tuning knobs, all device-tunable: model is ~13.4m wingspan in file units
constexpr int plane_count = 5;
constexpr float model_scale = 0.00675f; // ~0.13m wingspan
// Cruise evolution: each segment nudges the flight's cruise speed a little
// instead of redrawing it, so there is no velocity step at joints. Fresh
// flights draw 0.32–0.48 m/s; segments drift ±0.06 within the floor/cap.
constexpr float cruise_fresh_lo = 0.32f;
constexpr float cruise_fresh_hi = 0.48f;
constexpr float cruise_drift = 0.06f;
constexpr float cruise_max = 0.35f;
// Speed floor: paths shorter than dur would otherwise crawl (and a loop
// extension makes it worse). Over-long flights are shortened so cruise
// never drops below this; loop scheduling must respect it too.
constexpr float min_cruise_speed = 0.15f;
// Loiter behaviour: inside this radius legs meander freely; beyond it
// sampling homes hard back toward the live centre.
constexpr float loiter_radius = 1.0f;
constexpr double fade_in = 1.2;
// End-of-segment margin: figures are scheduled clear of the joint so a
// loop never overruns into the next chained segment.
constexpr double fade_out = 2.0;

// model nose points along +Z once the file's own root rotation (-90deg
// about X on "Air Plane") is applied; tweak below only if a re-export
// changes that convention
constexpr float model_yaw_adjust = 0.f;
// mirrored model X flips the apparent bank direction; flip once on device
// if planes bank out of their turns instead of into them
constexpr float bank_sign = 1.f;
// radians of bank per m/s^2 of lateral acceleration (coordinated-turn feel)
constexpr float bank_gain = 1.6f;
// hard bank limit, ~66 deg: steep wing-over turns stay readable at this scale
constexpr float bank_max = 1.15f;

// half-angle of the exclusion cone around the anchor-to-user axis (+Z in
// anchor space): flights inside it cross right in front of the face
constexpr float exclusion_tan = 0.2679491924f; // tan(15 deg)

// Legality limits for control-point legs: no clustered points (which stall
// the plane into a pivot) and no ultra-steep turns between legs. Yaw is
// allowed more freedom than pitch, like a real aircraft.
constexpr float min_leg_len = 0.45f;
constexpr float max_yaw_turn = 1.7453f;    // 100 deg direction change per leg: steep wing-over turns
constexpr float min_turn_radius = 0.32f;   // legs must span this much arc per radian of turn: big turns go wide
constexpr float max_flight_range = 2.0f;   // control points stay within this distance of the anchor
constexpr float launch_yaw_dev = 3.14159265f; // full circle: spawn headings cover all directions, cone-trimmed into flybys
constexpr float launch_pitch_dev = 0.5f;
// Homing: with this probability per leg, steer back toward the anchor so
// flights swing by the user instead of diffusing away forever. The
// exclusion cone still trims anything inside 15 deg into a near miss.
constexpr float homing_prob = 0.5f;
constexpr float homing_blend = 0.55f;
// Yaw sampling bias: legs always turn at least this fraction of the limit
// so flights wiggle constantly instead of sneaking in near-straight legs.
constexpr float min_turn_fraction = 0.25f;
constexpr float max_pitch_change = 0.5236f; // 30 deg climb/dive change per leg
constexpr float max_abs_pitch = 0.7854f;   // 45 deg absolute climb/dive

// Aerobatic maneuvers: at most one per flight, centered mid-flight so the
// fade never hides it. Rolls/knife are single-axis about the nose (same
// pattern as the coordinated bank, so provably safe); the loop is a true
// energy-shaped circle integrated live (entry speed in, slow apex, fast
// exit), so there are no velocity steps at either end.
constexpr double man_none_prob = 0.15;
constexpr double man_barrel_prob = 0.25;
constexpr double man_aileron_prob = 0.18;
constexpr double man_knife_prob = 0.12;
constexpr double man_loop_prob = 0.30; // remainder
constexpr double barrel_secs = 2.2;
constexpr double aileron_secs = 1.2;
constexpr double knife_hold_secs = 2.0;
constexpr double knife_trans_secs = 0.8;
constexpr float knife_angle = 1.45f; // ~83 deg: reads edge-on, never fully vanishes
// True-loop shape: radius and apex-speed fraction are randomized per flight
// (visibly different loops); duration emerges from the energy profile,
// and the figure is shrunk until it fits its distance slot.
constexpr float loop_r_lo = 0.20f;
constexpr float loop_r_hi = 0.45f;
constexpr double loop_top_lo = 0.45;
constexpr double loop_top_hi = 0.65;

// Engine trail ribbon: resampled while moving, fades out by age.
constexpr size_t trail_points = 60;
constexpr double trail_sample_dt = 1. / 30.;
constexpr double trail_max_age = trail_points * trail_sample_dt;
constexpr float trail_width_head = 0.02f;  // ribbon half-width at the engine (m)
constexpr float trail_width_tail = 0.055f; // ... at the faded end
constexpr float trail_alpha_max = 0.15f;
constexpr glm::vec3 trail_tint{0.9f, 0.92f, 0.98f};
// Gap between the tail and the start of the ribbon (m).
constexpr float trail_gap = 0.05f;

double smoothstep(double e0, double e1, double x)
{
	double t = std::clamp((x - e0) / (e1 - e0), 0., 1.);
	return t * t * (3. - 2. * t);
}

// Maneuver angle profile: trapezoidal angular velocity with smoothstep
// corners — spins up over the first `ramp` fraction of the window, holds a
// constant rate, spins back down. Total sweep is exactly `turns` revolutions
// and both rate and acceleration hit zero at the ends, so entries/exits
// never snap (plain smoothstep whips through the middle at 1.5x average
// rate, which reads as a violent flick on small fast figures).
double maneuver_angle(double phase, double turns, double ramp)
{
	ramp = std::clamp(ramp, 0.05, 0.45);
	phase = std::clamp(phase, 0., 1.);
	auto ss = [](double s) {
		s = std::clamp(s, 0., 1.);
		return s * s * (3. - 2. * s);
	};
	auto ss_int = [](double s) { // ∫₀ˢ smoothstep, for the ramp corners
		s = std::clamp(s, 0., 1.);
		return s * s * s - 0.5 * s * s * s * s;
	};
	double integral;
	if (phase < ramp)
		integral = ramp * ss_int(phase / ramp);
	else if (phase <= 1. - ramp)
		integral = ramp * 0.5 + (phase - ramp);
	else
		integral = (1. - ramp) - ramp * ss_int((1. - phase) / ramp);
	return turns * 2. * std::numbers::pi * integral / (1. - ramp);
}

// Energy-shaped loop speed: full entry speed at the bottom, `top` fraction
// at the apex, full again at exit — like trading airspeed for altitude and
// back. Rate and along-track acceleration both vanish at bottom and apex,
// so handoffs are smooth in velocity as well as position.
double loop_shape(double theta, double top)
{
	double a = 1. - top * top;
	return std::sqrt(std::max(1. - a * (1. - std::cos(theta)) * 0.5, 1e-6));
}

// ∫₀^{2π} dθ/shape: loop duration = r·J/entry_speed. Simpson, 64 panels —
double loop_J(double top)
{
	const int n = 64;
	double h = 2. * std::numbers::pi / n, sum = 0.;
	for (int i = 0; i <= n; ++i)
	{
		double w = (i == 0 || i == n) ? 0.5 : (i % 2 ? 4. : 2.);
		sum += w / loop_shape(i * h, top);
	}
	return sum * h / 3.;
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
// Fleet livery mix: mostly orange, some grey and navy, white rare.
const std::array<double, std::size(paints)> paint_weights{1.0, 3.0, 6.0, 2.0};

} // namespace

namespace
{
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
} // namespace

namespace wivrn
{

kiosk_planes::kiosk_planes(::scene & scene, uint32_t layer_mask, uint32_t trail_layer_mask) :
        scene_(scene), rng_(std::random_device{}())
{
	std::vector<size_t> livery_bag{2, 2, 1, 3, 0}; // orange-heavy, one of each
	std::shuffle(livery_bag.begin(), livery_bag.end(), rng_);
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
		flight f;
		f.root = entity;
		const paint & paint = paints[livery_bag[size_t(i) % livery_bag.size()]];
		spdlog::info("kiosk_planes: construction livery {} for plane {}", paint.name, i);
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
		flights_.push_back(std::move(f));
	}

	// Measure the model OBB in plane-root space for the trail emitter: walk
	// the first flight's subtree accumulating node transforms (the file's
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
	measure(flights_.front().root, glm::mat4(1));
	// Nose points along +Z: emit at the tail center.
	tail_root = {(obb_min.x + obb_max.x) * 0.5f, (obb_min.y + obb_max.y) * 0.5f, obb_min.z};

	// Shared trail ribbon: one node/primitive holding every flight's
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

	// Fixed strip topology: one independent 5-wide quad strip per flight.
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

glm::vec3 kiosk_planes::bezier(const flight & f, double t)
{
	double u = 1. - t;
	return f.p0 * float(u * u * u) + f.p1 * float(3. * u * u * t) + f.p2 * float(3. * u * t * t) + f.p3 * float(t * t * t);
}

glm::vec3 kiosk_planes::bezier_vel(const flight & f, double t)
{
	double u = 1. - t;
	return (f.p1 - f.p0) * float(3. * u * u) + (f.p2 - f.p1) * float(6. * u * t) + (f.p3 - f.p2) * float(3. * t * t);
}

glm::vec3 kiosk_planes::bezier_acc(const flight & f, double t)
{
	double u = 1. - t;
	return 6.f * float(u) * (f.p2 - 2.f * f.p1 + f.p0) + 6.f * float(t) * (f.p3 - 2.f * f.p2 + f.p1);
}

void kiosk_planes::build_segment(flight & f, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient, bool fresh)
{
	glm::mat3 R = glm::mat3_cast(anchor_orient);
	std::uniform_real_distribution<float> lateral(-0.3f, 0.3f);
	std::uniform_real_distribution<float> vertical(-0.2f, 0.2f);
	std::uniform_real_distribution<float> near_dist(0.6f, 1.0f);
	std::uniform_real_distribution<float> mid_dist(0.8f, 1.3f);
	std::uniform_real_distribution<float> far_dist(1.5f, 2.2f);
	std::uniform_real_distribution<float> cruise_fresh_dist(cruise_fresh_lo, cruise_fresh_hi);
	std::uniform_real_distribution<float> cruise_drift_dist(-cruise_drift, cruise_drift);

	auto clamp_height = [&](glm::vec3 v) {
		v.y = std::clamp(v.y, anchor_pos.y - 0.6f, anchor_pos.y + 0.9f);
		return v;
	};

	if (fresh)
	{
		// launch from just behind the overlay panel; the anchor is frozen
		// here so later panel motion never perturbs the flight
		f.p0 = anchor_pos + R * glm::vec3(lateral(rng_), vertical(rng_), -0.2f);
	}
	else
	{
		// Chain onto the live curve end: same point, same tangent
		// direction — but capped at 1 m. The full old leg (up to 5 m when
		// stretched) would lunge far outward before the home-biased legs
		// yank back: that out-and-back slalom read as flying away. The cap
		// only reshapes the curve (arc-walk speed is distance-based), so
		// continuity is unaffected.
		f.p0 = f.p3;
		glm::vec3 old_leg = f.p3 - f.p2;
		float old_len = glm::length(old_leg);
		f.p1 = f.p0 + (old_len > 1e-6f ? old_leg / old_len * std::min(old_len, 1.0f) : glm::vec3{0.f, 0.f, -1.f});
	}

	// Keep flights out of the exclusion cone around the anchor-to-user
	// axis: paths crossing right in front of the face are startling (and
	// alias badly). Offending control points move to the nearest cone edge
	// with a small margin; the fade-in covers the spawn behind the panel.
	// p0 stays exempt: it sits behind the panel by construction.
	// (Inverse-rotate by the anchor orientation: R is orthonormal, so this
	// is the world-to-anchor transform without needing a transpose.)
	const glm::quat Rinv = glm::inverse(anchor_orient);
	auto push_outside_cone = [&](const glm::vec3 & world) {
		glm::vec3 local = Rinv * (world - anchor_pos);
		if (local.z > 0.f and std::abs(local.x) < local.z * exclusion_tan and
		    std::abs(local.y) < local.z * exclusion_tan)
		{
			const float bx = local.z * exclusion_tan * 1.2f;
			const float by = local.z * exclusion_tan * 1.2f;
			if (std::abs(local.x) / bx < std::abs(local.y) / by)
				local.x = local.x >= 0.f ? bx : -bx;
			else
				local.y = local.y >= 0.f ? by : -by;
		}
		return anchor_pos + R * local;
	};
	auto leg_pitch = [](const glm::vec3 & d) {
		float len = glm::length(d);
		return len > 1e-6f ? std::asin(std::clamp(d.y / len, -1.f, 1.f)) : 0.f;
	};
	auto yaw_change = [](const glm::vec3 & d0, const glm::vec3 & d1) {
		glm::vec2 a(d0.x, d0.z), b(d1.x, d1.z);
		float la = glm::length(a), lb = glm::length(b);
		if (la < 1e-4f or lb < 1e-4f)
			return 999.f; // near-vertical leg: force a resample
		return std::acos(std::clamp(glm::dot(a / la, b / lb), -1.f, 1.f));
	};

	// Sample each control point chained tip-to-tail off the previous point,
	// steering around the previous leg direction: the achieved turn then
	// matches the sampled one by construction (the old code sampled around
	// the anchor, so the turn check measured a different vector than the
	// sampler produced and the fallback kept invalid legs). Cone push and
	// height clamp are verified after the fact with a few retries; the
	// fallback flies straight, which always satisfies the turn limits.
	// Loiter steering: near the centre legs meander freely; straying past
	// the loiter radius homes every leg back toward the live anchor, so the
	// fleet hangs around a moving centre instead of diffusing away. The pull
	// stays gentle on purpose: a hard blend would demand a >limit U-turn,
	// fail leg_ok on every attempt, and drop to the straight fallback (the
	// escape bug). Feasible intermediate headings turn home progressively.
	float homing_p = homing_prob, homing_b = homing_blend;
	float yaw_scale = 1.f;
	if (not fresh)
	{
		float d = glm::distance(f.p1, anchor_pos);
		if (d > loiter_radius)
		{
			homing_p = 1.f;
			homing_b = 0.6f;
			yaw_scale = 1.4f;
		}
		else if (d > loiter_radius * 0.6f)
		{
			homing_p = 0.8f;
			homing_b = 0.7f;
		}
	}
	auto sample_point = [&](const glm::vec3 & base, const glm::vec3 & prev2, const glm::vec3 & prev1, bool check_turn, auto & dist_rng, float yaw_lim, float pitch_lim, float hp, float hb, const glm::vec3 * launch_dir = nullptr) {
		auto gen = [&]() {
			glm::vec3 pd = launch_dir ? *launch_dir : prev1 - prev2;
			float pp = launch_dir ? 0.f : leg_pitch(pd);
			float p = std::clamp(pp + std::uniform_real_distribution<float>(-pitch_lim, pitch_lim)(rng_), -max_abs_pitch, max_abs_pitch);
			float u = std::uniform_real_distribution<float>(-1.f, 1.f)(rng_);
			float y = std::atan2(pd.x, pd.z) + (u >= 0.f ? 1.f : -1.f) * (min_turn_fraction + (1.f - min_turn_fraction) * std::abs(u)) * yaw_lim;
			glm::vec3 d{std::cos(p) * std::sin(y), std::sin(p), std::cos(p) * std::cos(y)};
			if (not launch_dir and std::uniform_real_distribution<float>(0.f, 1.f)(rng_) < hp)
			{
				glm::vec3 to_anchor = anchor_pos - base;
				if (glm::length(to_anchor) > 1e-3f)
					d = glm::normalize(glm::mix(d, to_anchor / glm::length(to_anchor), hb));
			}
			return base + d * dist_rng(rng_);
		};
		auto leg_ok = [&](const glm::vec3 & cand) {
			glm::vec3 d1 = cand - prev1;
			if (glm::length(d1) < min_leg_len or std::abs(leg_pitch(d1)) > max_abs_pitch)
				return false;
			if (glm::length(cand - anchor_pos) > max_flight_range)
				return false;
			if (check_turn)
			{
				glm::vec3 d0 = prev1 - prev2;
				if (glm::length(d0) < min_leg_len)
					return false;
				float yaw = yaw_change(d0, d1);
				if (yaw > max_yaw_turn)
					return false;
				// Wide banks only: a large direction change needs long legs,
				// otherwise it is a lightning-fast pivot, not a turn.
				if (std::min(glm::length(d0), glm::length(d1)) < min_turn_radius * yaw)
					return false;
				if (std::abs(leg_pitch(d1) - leg_pitch(d0)) > max_pitch_change)
					return false;
			}
			return true;
		};
		glm::vec3 p = clamp_height(push_outside_cone(gen()));
		for (int attempt = 0; attempt < 12 and not leg_ok(p); ++attempt)
			p = clamp_height(push_outside_cone(gen()));
		if (not leg_ok(p))
		{
			// Last resort: fly straight — but clamped into the range ball
			// around the live anchor, so a sampler failure can never
			// become an escape. (An unclamped straight fallback here was
			// the fly-away bug: far-out U-turns fail leg_ok, and straight
			// carried the plane ever farther.)
			glm::vec3 pd = launch_dir ? *launch_dir : prev1 - prev2;
			float pl = glm::length(pd);
			glm::vec3 dn = pl > 1e-6f ? pd / pl : glm::vec3{0.f, 0.f, -1.f};
			glm::vec3 fb = base + dn * dist_rng.max();
			glm::vec3 off = fb - anchor_pos;
			if (glm::length(off) > max_flight_range)
				fb = anchor_pos + off / glm::length(off) * (max_flight_range * 0.95f);
			p = clamp_height(push_outside_cone(fb));
		}
		return p;
	};
	const glm::vec3 launch_dir = R * glm::vec3{0.f, 0.f, -1.f};
	if (fresh)
		f.p1 = sample_point(f.p0, f.p0, f.p0, false, near_dist, launch_yaw_dev, launch_pitch_dev, homing_p, homing_b, &launch_dir);
	f.p2 = sample_point(f.p1, f.p0, f.p1, true, mid_dist, max_yaw_turn * yaw_scale, max_pitch_change, homing_p, homing_b);
	f.p3 = sample_point(f.p2, f.p1, f.p2, true, far_dist, max_yaw_turn * yaw_scale, max_pitch_change, homing_p, homing_b);

	// Arc-length table for constant-speed flight: with clustered control
	// points gone the curve is well-behaved, and walking it by distance
	// (not time) removes the slow-down into what is left of the turns.
	f.total_len = 0.f;
	glm::vec3 prev = bezier(f, 0.);
	f.arclen[0] = 0.f;
	for (size_t i = 1; i < f.arclen.size(); ++i)
	{
		glm::vec3 p = bezier(f, double(i) / double(f.arclen.size() - 1));
		f.total_len += glm::distance(p, prev);
		f.arclen[i] = f.total_len;
		prev = p;
	}

	// Segment pacing from an evolving cruise: no velocity step at joints.
	// seg_T follows (bounded, so degenerate paths can't stall or sprint).
	if (fresh)
		f.cruise = cruise_fresh_dist(rng_);
	else
		f.cruise = std::clamp(f.cruise + cruise_drift_dist(rng_), min_cruise_speed, cruise_max);
	f.seg_T = std::clamp(f.total_len / std::max(f.cruise, 1e-3f), 4.f, 16.f);
	f.dist_flown = 0.f;
	pick_maneuver(f);
}

void kiosk_planes::respawn(flight & f, XrTime now, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient)
{
	build_segment(f, anchor_pos, anchor_orient, true);
	f.t0 = now;
	f.trail.clear();
	f.trail_t.clear();
	f.last_sample_t = 0;
	f.prev_now = 0;
	f.bank_sm = 0.f;
}

void kiosk_planes::extend(flight & f, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient)
{
	// New segment, same flight: trail keeps flowing, livery stays, the
	// lifetime clock keeps running. Only path, pacing and figure refresh.
	build_segment(f, anchor_pos, anchor_orient, false);
}

void kiosk_planes::pick_maneuver(flight & f)
{
	// Aerobatic maneuver for this segment. Everything is scheduled in DISTANCE
	// fraction (dist_flown/total_len), never time: climb-coupled airspeed
	// means fast downhill flights exhaust their path early, and a
	// time-scheduled figure would never trigger. Rolls/knife run on
	// distance windows; the loop starts at a distance mark and exits on
	// θ = 2π (duration emerges, so the figure is shrunk — never the flight
	// stretched — until it fits).
	f.man = flight::maneuver::none;
	f.loop_active = false;
	f.loop_done = false;
	f.loop_theta = 0.;
	f.loop_t = 0.;
	f.loop_exit_t = -1.;
	{
		double u = std::uniform_real_distribution<double>(0., 1.)(rng_);
		double v_c = f.total_len / std::max(double(f.seg_T), 1e-3);
		double len = std::max(f.total_len, 1e-3f);
		const char * man_name = "none";
		if (u < man_none_prob + man_barrel_prob + man_aileron_prob + man_knife_prob)
		{
			double secs = 0.;
			if (u < man_none_prob)
				f.man = flight::maneuver::none;
			else if (u < man_none_prob + man_barrel_prob)
			{
				f.man = flight::maneuver::barrel;
				secs = barrel_secs;
				man_name = "barrel";
			}
			else if (u < man_none_prob + man_barrel_prob + man_aileron_prob)
			{
				f.man = flight::maneuver::aileron;
				secs = aileron_secs;
				man_name = "aileron";
			}
			else
			{
				f.man = flight::maneuver::knife;
				secs = knife_hold_secs + 2. * knife_trans_secs;
				man_name = "knife";
			}
			if (f.man != flight::maneuver::none)
			{
				f.man_dir = std::uniform_real_distribution<float>(0., 1.)(rng_) < 0.5f ? -1.f : 1.f;
				// window in distance fraction: seconds of cruise mapped
				// onto path length, clear of both fades
				double width = std::min(secs * v_c / len, 0.5);
				double lo = width * 0.5 + (fade_in + 0.5) * v_c / len;
				double hi = 1. - width * 0.5 - (fade_out + 0.5) * v_c / len;
				if (hi > lo)
				{
					f.man_width = width;
					f.man_center = std::uniform_real_distribution<double>(lo, hi)(rng_);
				}
				else
				{
					f.man = flight::maneuver::none;
					man_name = "none";
				}
			}
		}
		else
		{
			// True loop: randomized diameter and apex energy. Shrinks the
			// figure (tighter radius, gentler dip) until the estimated
			// duration fits the slot — the flight is never stretched, so
			// cruise (and its floor) stays intact.
			f.man = flight::maneuver::loop;
			man_name = "loop";
			f.loop_r = std::uniform_real_distribution<float>(loop_r_lo, loop_r_hi)(rng_);
			f.loop_top = std::uniform_real_distribution<double>(loop_top_lo, loop_top_hi)(rng_);
			bool placed = false;
			for (int attempt = 0; attempt < 6 && !placed; ++attempt)
			{
				double J = loop_J(f.loop_top);
				double T_est = f.loop_r * J / std::max(v_c, 0.05);
				double lo = std::max(0.22, (fade_in + 0.5) * v_c / len);
				double hi = std::min(0.72, 1. - (fade_out + 0.5 + T_est * 1.2) * v_c / len);
				if (hi > lo)
				{
					f.loop_start_d = std::uniform_real_distribution<double>(lo, hi)(rng_);
					placed = true;
				}
				else
				{
					f.loop_r = std::max(f.loop_r * 0.8f, 0.15f);
					f.loop_top = std::min(f.loop_top + 0.08, 0.7);
				}
			}
			if (!placed)
			{
				f.man = flight::maneuver::none;
				man_name = "none";
			}
		}
		spdlog::info("kiosk_planes: maneuver {} r={:.2f} top={:.2f} start={:.2f}", man_name, f.loop_r, f.loop_top, f.loop_start_d);
	}
}

void kiosk_planes::update(XrTime now, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient, bool active, const std::optional<glm::vec3> & camera)
{
	if (active && !was_active_)
	{
		// (re)opening: respawn planes that never flew or expired, staggered
		// in *time* so they launch one at a time instead of bunching at the
		// spawn point. (Spreading their clocks alone does not work: the
		// arc-length walk starts every fresh flight at dist_flown = 0, so
		// staggered clocks without staggered start times still pile up at
		// p0.) Untouched planes keep their own clocks, so a flickering
		// overlay can't regroup the fleet into synchronized bundles.
		std::uniform_real_distribution<double> jitter(0., 0.5);
		size_t slot = 0;
		for (auto & f: flights_)
		{
			if (f.total_len > 0.f)
				continue;
			respawn(f, now, anchor_pos, anchor_orient);
			f.t0 = now + XrTime((double(slot++) + jitter(rng_)) * 1.5e9);
		}
	}
	was_active_ = active;

	for (auto & f: flights_)
	{
		auto * node = scene_.world.try_get<components::node>(f.root);
		if (not node)
			continue;
		node->visible = active;
		if (not active)
			continue;

		double age = double(now - f.t0) * 1e-9;
		if (age < 0.)
			age = 0.;
		if (f.dist_flown >= f.total_len && !f.loop_active)
		{
			// Segment flown: chain the next one onto the live flight
			// (same point/tangent, steered at the live centre) and keep
			// flying this frame on the fresh arc table.
			extend(f, anchor_pos, anchor_orient);
		}
		// Arc-length walk with climb-coupled airspeed: the plane works
		// uphill and swoops downhill instead of holding exact constant
		// speed, which reads as natural velocity variation without any
		// pivoting. Degenerate paths fall back to raw time.
		auto arc_param = [&](float dist) {
			float s = std::clamp(dist, 0.f, f.total_len);
			size_t seg = 1;
			while (seg < f.arclen.size() - 1 and f.arclen[seg] < s)
				++seg;
			float s0 = f.arclen[seg - 1], s1 = f.arclen[seg];
			return (float(seg - 1) + (s1 > s0 ? (s - s0) / (s1 - s0) : 0.f)) / float(f.arclen.size() - 1);
		};
		float tt = f.total_len > 1e-6f ? arc_param(f.dist_flown) : 0.f;
		float v_cruise = f.total_len / std::max(f.seg_T, 1e-3f);

		glm::vec3 vel_probe = bezier_vel(f, tt);
		float probe_len = glm::length(vel_probe);
		float slope = probe_len > 1e-6f ? vel_probe.y / probe_len : 0.f;
		double dt = f.prev_now == 0 ? 1. / 60. : std::clamp(double(now - f.prev_now) * 1e-9, 0.001, 0.1);
		f.prev_now = now;
		// Catch-up: up to 2.5x pace when left behind, easing to 1x inside
		// the loiter radius. Distance-driven so perfectly smooth — the
		// fleet follows a moving centre instead of receding segment by
		// segment. (node->position here is last frame's: 1-frame lag.)
		float behind = glm::distance(node->position, anchor_pos);
		float catchup = 1.f + std::min(behind / std::max(loiter_radius, 1e-3f), 1.5f);
		float v = v_cruise * std::clamp(1.f - 0.35f * slope, 0.7f, 1.35f) * catchup;

		// True loop-the-loop, event-driven: starts at a distance mark, exits
		// on θ = 2π. Entry frame is frozen from live state; θ integrates
		// the energy profile at the *entry* speed, so position and velocity
		// are exactly continuous at both ends — fast in, slow over the
		// top, fast out. Path progress pauses mid-loop; expiry waits for
		// the exit so a long loop can never pop mid-figure.
		bool in_loop = false;
		if (f.man == flight::maneuver::loop && !f.loop_done && f.seg_T > 1e-3f)
		{
			double dist_frac = f.total_len > 1e-3f ? f.dist_flown / f.total_len : 0.;
			if (!f.loop_active && dist_frac >= f.loop_start_d && age >= fade_in + 0.3)
			{
				glm::vec3 entry_vel = bezier_vel(f, tt);
				glm::vec3 f0 = glm::length(entry_vel) > 1e-6f ? glm::normalize(entry_vel) : glm::vec3{0.f, 0.f, -1.f};
				glm::vec3 world_up{0.f, 1.f, 0.f};
				glm::vec3 u = world_up - f0 * glm::dot(world_up, f0);
				u = glm::length(u) > 1e-4f ? glm::normalize(u) : world_up;
				glm::vec3 r0 = glm::cross(world_up, f0);
				r0 = glm::length(r0) > 1e-4f ? glm::normalize(r0) : glm::vec3{1.f, 0.f, 0.f};
				f.loop_p0 = bezier(f, tt);
				f.loop_f = f0;
				f.loop_u = u;
				f.loop_right = r0;
				f.loop_v = v;
				f.loop_theta = 0.;
				f.loop_t = 0.;
				// Blend duration scales with the entry turn rate (r/v):
				// tighter or faster loops get a longer ease-in, capped to
				// stay a small fraction of the figure.
				f.loop_blend = std::clamp(0.8f * f.loop_r / std::max(v, 0.05f), 0.4f, 1.2f);
				glm::vec3 entry_acc = bezier_acc(f, tt);
				float v2 = glm::dot(entry_vel, entry_vel);
				float kappa = glm::dot(glm::cross(entry_vel, entry_acc), world_up) / std::max(v2 * std::sqrt(v2), 1e-6f);
				f.loop_bank = std::clamp(bank_sign * -kappa * v * v * bank_gain, -bank_max, bank_max);
				f.loop_active = true;
			}
			in_loop = f.loop_active;
		}
		if (in_loop)
		{
			f.loop_t += dt;
			f.loop_theta += f.loop_v * loop_shape(f.loop_theta, f.loop_top) / f.loop_r * dt;
			if (f.loop_theta >= 2. * std::numbers::pi)
			{
				f.loop_theta = 2. * std::numbers::pi;
				f.loop_active = false;
				f.loop_done = true;
			}
		}
		if (!in_loop)
		{
			f.dist_flown = std::min(f.dist_flown + v * float(dt), f.total_len);
			tt = f.total_len > 1e-6f ? arc_param(f.dist_flown) : tt;
		}

		glm::vec3 pos = bezier(f, tt);
		glm::vec3 fwd{0.f, 0.f, 1.f};
		bool have_fwd = false;
		if (in_loop)
		{
			// Circle frame: tangent carries the nose, fixed entry right
			// keeps the wings stable through vertical. Entry bank eases
			// out over the top (cos²: zero rate at ends and apex) and is
			// fully restored at exit, where frame meets frozen path frame.
			// Position eases from the straight-line path extension onto
			// the circle over loop_blend: curvature ramps 0 → 1/r instead
			// of stepping, so the pull-up has no entry yank (velocity is
			// continuous throughout; only higher derivatives ease).
			float c = float(std::cos(f.loop_theta)), s = float(std::sin(f.loop_theta));
			glm::vec3 circle = f.loop_p0 + f.loop_r * ((1.f - c) * f.loop_u + s * f.loop_f);
			double ease = smoothstep(0., 1., std::min(f.loop_t / f.loop_blend, 1.));
			if (f.loop_t < f.loop_blend)
			{
				glm::vec3 straight = f.loop_p0 + f.loop_f * (f.loop_v * float(f.loop_t));
				pos = glm::mix(straight, circle, float(ease));
			}
			else
				pos = circle;
			// Nose angle eases with the same weight (φ = θ·ease): zero
			// pitch rate on the entry frame, converging exactly onto the
			// tangent by blend end. Briefly lags the path like angle of
			// attack, then flies the circle nose-on.
			double phi = f.loop_theta * ease;
			float cn = float(std::cos(phi)), sn = float(std::sin(phi));
			glm::vec3 tangent = sn * f.loop_u + cn * f.loop_f;
			fwd = s * f.loop_u + c * f.loop_f; // trail follows true motion
			have_fwd = true;
			glm::vec3 upv = glm::cross(tangent, f.loop_right);
			float bank_blend = f.loop_bank * (cn * 0.5f + 0.5f);
			bank_blend *= bank_blend; // cos²(φ/2): 1 at ends, 0 inverted
			node->orientation = glm::angleAxis(bank_blend, tangent) *
			                    glm::quat_cast(glm::mat3(f.loop_right, upv, tangent));
			// Closure frame exits through here: stash it so the post-loop
			// handoff below can slerp home instead of stepping.
			if (f.loop_theta >= 2. * std::numbers::pi - 1e-9)
			{
				f.loop_exit_o = node->orientation;
				f.loop_exit_t = 0.;
			}
		}
		else
		{
			glm::vec3 vel = bezier_vel(f, tt);
			if (glm::length(vel) > 1e-6f)
			{
				fwd = glm::normalize(vel);
				have_fwd = true;
				// model nose (+Z after the file's root rotation) onto the
				// flight direction; mirrored X is absorbed by bank_sign
				glm::vec3 up0{0.f, 1.f, 0.f};
				glm::vec3 right = glm::cross(up0, fwd);
				if (glm::length(right) < 1e-4f)
					right = {1.f, 0.f, 0.f};
				else
					right = glm::normalize(right);
				glm::vec3 up = glm::cross(fwd, right);
				glm::quat orient = glm::quat_cast(glm::mat3(right, up, fwd));

				// Coordinated-turn bank from path curvature instead of the old
				// sine wobble: lateral acceleration over speed squared, scaled to
				// an arcade feel and clamped. Positive bank about the nose rolls
				// into the turn; flip bank_sign once on device if they bank out.
				glm::vec3 acc = bezier_acc(f, tt);
				float v2 = glm::dot(vel, vel);
				float kappa = glm::dot(glm::cross(vel, acc), up0) / std::max(v2 * std::sqrt(v2), 1e-6f);
				float bank = std::clamp(bank_sign * -kappa * v * v * bank_gain, -bank_max, bank_max);
				// Smooth the applied bank (τ ≈ 0.35 s): curvature steps at
				// segment joints would otherwise snap the roll angle. Rides
				// along (extend never resets it); maneuver rolls stay raw.
				f.bank_sm += (bank - f.bank_sm) * float(std::min(1., dt / 0.35));
				// Aerobatic maneuver overlay, all eased so nothing snaps:
				// rolls/knife rotate about the nose. Heading-preserving
				// by construction.
				float extra_roll = 0.f;
				if (f.man == flight::maneuver::barrel || f.man == flight::maneuver::aileron ||
				    f.man == flight::maneuver::knife)
				{
					// distance-driven phase: fast flights exhaust path
					// early, so a time window could never open
					double dfrac = f.total_len > 1e-3f ? f.dist_flown / f.total_len : 0.;
					double phase = std::clamp((dfrac - (f.man_center - f.man_width * 0.5)) / f.man_width, 0., 1.);
					switch (f.man)
					{
					case flight::maneuver::barrel:
						extra_roll = f.man_dir * float(maneuver_angle(phase, 1., 0.30));
						break;
					case flight::maneuver::aileron:
						extra_roll = f.man_dir * float(maneuver_angle(phase, 1., 0.25));
						break;
					case flight::maneuver::knife:
					{
						// hold near 90 deg through the middle of the window,
						// eased in/out over knife_trans_secs each side
						double window_s = f.man_width * f.total_len / std::max(v_cruise, 0.05f);
						double t_in = knife_trans_secs / window_s;
						double pulse = smoothstep(0., std::min(t_in, 0.5), phase) *
						               (1. - smoothstep(std::max(1. - t_in, 0.5), 1., phase));
						extra_roll = f.man_dir * knife_angle * float(pulse);
						break;
					}
					default:
						break;
					}
				}
				orient = glm::angleAxis(f.bank_sm + extra_roll, fwd) * orient;

				if (model_yaw_adjust != 0.f)
					orient = orient * glm::angleAxis(model_yaw_adjust, glm::vec3(0.f, 1.f, 0.f));
				// Post-loop handoff: ease from the stashed closure frame
				// onto the live path frame over the same blend duration.
				// Zero rotation rate at both ends, so the pull-out never
				// steps the nose.
				if (f.loop_exit_t >= 0.)
				{
					f.loop_exit_t += dt;
					float b = float(smoothstep(0., 1., std::min(f.loop_exit_t / f.loop_blend, 1.)));
					orient = glm::slerp(f.loop_exit_o, orient, b);
					if (f.loop_exit_t >= f.loop_blend)
						f.loop_exit_t = -1.;
				}
				node->orientation = orient;
			}
		}
		node->position = pos;

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

		// Record the engine position for the trail ribbon: the measured
		// tail offset is already in world units at the construction
		// scale, so only the orientation applies (uniform scale). The
		// sample hangs a little behind the tail so the ribbon never
		// touches the body. A long stall (hidden overlay, hitches) clears
		// the trail instead of streaking across the scene on resume.
		// Pre-launch (staggered start still in the future) records nothing:
		// the trail must grow out of the flight, not pre-exist it.
		glm::vec3 tail = pos + node->orientation * tail_root - (have_fwd ? fwd * trail_gap : glm::vec3{0.f});
		if (age <= 0.)
		{
			f.trail.clear();
			f.trail_t.clear();
			f.last_sample_t = 0;
		}
		else if (f.last_sample_t != 0 and double(now - f.last_sample_t) * 1e-9 > 0.5)
		{
			f.trail.clear();
			f.trail_t.clear();
		}
		else if (f.last_sample_t == 0 or double(now - f.last_sample_t) * 1e-9 >= trail_sample_dt)
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
	constexpr double trail_max_age = trail_points * trail_sample_dt;

	for (int p = 0; p < plane_count; ++p)
	{
		const flight & f = flights_[size_t(p)];
		trail_vertex * out = verts + size_t(p) * trail_verts_per_plane;

		// Slot j=0 is the newest sample; pad short trails with the oldest
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
			// Gradual age falloff (long tail, not a hard end) times a head
			// ramp over the first segments, so a young trail can never show
			// a bright leading edge no matter the sample age. The tail ramp
			// keys off the live point count (not the slot index): padding
			// duplicates the oldest sample, so without it a young trail
			// would end in a bright streak cut off at the last live point.
			const float age_frac = std::clamp(ring_age[j] / float(trail_max_age), 0.f, 1.f);
			const float envelope = std::pow(1.f - age_frac, 1.5f);
			const float head_ramp = float(smoothstep(0., 8., double(j)));
			const float tail_ramp = j < n ? 1.f - float(smoothstep(double(n > 4 ? n - 4 : 0), double(n > 1 ? n - 1 : 0), double(j))) : 0.f;
			const float a = live ? trail_alpha_max * f.trail_alpha * envelope * head_ramp * tail_ramp : 0.f;

			// Soft across-width profile: transparent edges, bright core.
			constexpr float across[5] = {-1.f, -0.5f, 0.f, 0.5f, 1.f};
			constexpr float across_alpha[5] = {0.f, 0.6f, 1.f, 0.6f, 0.f};
			for (size_t k = 0; k < 5; ++k)
			{
				const float ak = a * across_alpha[k];
				out[5 * j + k] = {
				        glm::vec4(pt + side * (across[k] * width * 0.5f), 1.f),
				        glm::vec4(trail_tint * ak, ak), // premultiplied
				};
			}
		}
	}
}

} // namespace wivrn
