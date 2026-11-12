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
#include <cstring>
#include <functional>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iterator>
#include <map>
#include <numbers>
#include <string>

namespace
{
// tuning knobs, all device-tunable: model is ~13.4m wingspan in file units
constexpr int plane_count = 4;
constexpr float model_scale = 0.00675f; // ~0.13m wingspan
constexpr double fade_in = 1.2;
constexpr double spawn_interval = 1.0; // plane i spawns at activation + i * this
constexpr float spawn_radius = 1.0f;   // random position within this distance of the overlay

double smoothstep(double e0, double e1, double x)
{
	double t = std::clamp((x - e0) / (e1 - e0), 0., 1.);
	return t * t * (3. - 2. * t);
}

// body paint jobs for the "Material" part, with per-channel variance;
// glass, emissive and dark/metal parts stay as authored
struct paint
{
	glm::vec3 base;
	float variance;
};
const paint paints[]{
        {{1.f, 1.f, 1.f}, 0.f},                        // white
        {{0.62f, 0.62f, 0.62f}, 0.12f},                // grey
        {{0.75f, 0.38f, 0.04f}, 0.14f},                // dark dynacon orange
        {{0.03f, 0.06f, 0.36f}, 0.10f},                // dark #0E1B8D navy
};

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
	node->orientation = glm::angleAxis(yaw_dist(rng_), glm::vec3(0.f, 1.f, 0.f));
	f.spawned = true;
}

void kiosk_planes::update(XrTime now, const glm::vec3 & anchor_pos, const glm::quat & anchor_orient, bool active)
{
	(void)anchor_orient; // placement is position-only; heading is random
	if (active && !was_active_)
	{
		// (re)opening: replay the one-by-one spawn-in from scratch
		activated_t0_ = now;
		for (auto & f: planes_)
		{
			f.spawned = false;
			f.spawn_t0 = 0;
		}
	}
	was_active_ = active;

	size_t slot = 0;
	for (auto & f: planes_)
	{
		auto * node = scene_.world.try_get<components::node>(f.root);
		if (not node)
		{
			++slot;
			continue;
		}
		if (not active)
		{
			node->visible = false;
			++slot;
			continue;
		}

		XrTime due = activated_t0_ + XrDuration(slot * spawn_interval * 1e9);
		++slot;
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
	}
}

} // namespace wivrn
