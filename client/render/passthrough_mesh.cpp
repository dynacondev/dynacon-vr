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

#include "passthrough_mesh.h"

#include "utils/mapped_file.h"
#include <array>
#include <bit>
#include <cmath>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/quaternion.hpp>
#include <numeric>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

// File-scope bake helpers (needed both inside append_primitive's anonymous
// namespace and by the global conforming pass in flatten_gltf below).

// Bitwise position key: collapse bitwise-identical baked positions so
// duplicated verts from overlapping primitives share one index.
struct weld_key
{
	uint32_t x, y, z;
	bool operator==(const weld_key & o) const
	{
		return x == o.x && y == o.y && z == o.z;
	}
};
struct weld_hash
{
	size_t operator()(weld_key const & k) const noexcept
	{
		size_t h = std::hash<uint32_t>{}(k.x);
		h ^= std::hash<uint32_t>{}(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
		h ^= std::hash<uint32_t>{}(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
		return h;
	}
};

// Sanity cap: refuse absurd meshes rather than handing them to the runtime
constexpr size_t max_triangles = 4'000'000;

// Conforming closure over [range_vert, ...)/[range_start, ...): force-split
// tris at hanging vertices. Defined after the anonymous namespace (it is
// called both per-primitive and globally from flatten_gltf).
void conforming_closure_range(passthrough_mesh::triangle_soup & out, uint32_t range_vert, size_t range_start); // fwd (defined below)

// Explicit silhouette loop extraction (closed-volume dynamic AA): one
// record per unique position-keyed edge with both opposite vertices.
// Sort-based (3T directed edges, 64-bit canonical keys), linear sweep.
// Boundary/non-manifold/inconsistent groups encode c1==c0 (forced emit,
// fail-safe over-reveal). Runs at map load; off the frame path.
std::pair<std::vector<passthrough_mesh::silhouette_edge>, passthrough_mesh::silhouette_edge_stats>
passthrough_mesh::build_silhouette_edges(const passthrough_mesh::triangle_soup & soup)
{
	std::vector<silhouette_edge> edges;
	silhouette_edge_stats stats{};
	if (soup.indices.size() < 3 or soup.vertices.empty())
		return {edges, stats};
	// Canonical position ids (cross-primitive duplicates share one key).
	std::unordered_map<weld_key, uint32_t, weld_hash> canon;
	canon.reserve(soup.vertices.size() * 2);
	uint32_t next_canon = 0;
	std::vector<uint32_t> vcanon(soup.vertices.size(), UINT32_MAX);
	for (size_t v = 0; v < soup.vertices.size(); ++v)
	{
		const XrVector3f & p = soup.vertices[v];
		weld_key k{std::bit_cast<uint32_t>(p.x), std::bit_cast<uint32_t>(p.y), std::bit_cast<uint32_t>(p.z)};
		auto it = canon.find(k);
		if (it == canon.end())
		{
			canon.emplace(k, next_canon);
			vcanon[v] = next_canon++;
		}
		else
			vcanon[v] = it->second;
	}
	struct directed
	{
		uint64_t key;
		uint32_t a, b, c;
	};
	size_t tri_count = soup.indices.size() / 3;
	std::vector<directed> dirs;
	dirs.reserve(tri_count * 3);
	for (size_t t = 0; t < tri_count; ++t)
	{
		uint32_t i0 = soup.indices[t * 3], i1 = soup.indices[t * 3 + 1], i2 = soup.indices[t * 3 + 2];
		if (i0 >= soup.vertices.size() or i1 >= soup.vertices.size() or i2 >= soup.vertices.size())
			continue;
		if (i0 == i1 or i1 == i2 or i2 == i0)
			continue;
		uint32_t c0 = vcanon[i0], c1 = vcanon[i1], c2 = vcanon[i2];
		if (c0 == UINT32_MAX or c1 == UINT32_MAX or c2 == UINT32_MAX)
			continue;
		auto push = [&](uint32_t u, uint32_t v, uint32_t w) {
			uint32_t cu = vcanon[u], cv = vcanon[v];
			uint64_t key = cu < cv ? ((uint64_t)cu << 32 | cv) : ((uint64_t)cv << 32 | cu);
			dirs.push_back({key, u, v, w});
		};
		push(i0, i1, i2);
		push(i1, i2, i0);
		push(i2, i0, i1);
	}
	std::sort(dirs.begin(), dirs.end(), [](const directed & l, const directed & r) { return l.key < r.key; });
	edges.reserve(tri_count * 3 / 2);
	for (size_t i = 0; i < dirs.size();)
	{
		size_t j = i + 1;
		while (j < dirs.size() and dirs[j].key == dirs[i].key)
			++j;
		size_t n = j - i;
		auto pos_of = [&](uint32_t v) -> XrVector3f { return soup.vertices[v]; };
		if (n == 2)
		{
			const auto & e0 = dirs[i], & e1 = dirs[i + 1];
			bool opposite = (e0.a == e1.b and e0.b == e1.a);
			if (opposite)
			{
				edges.push_back({pos_of(e0.a), pos_of(e0.b), pos_of(e0.c), pos_of(e1.c)});
			}
			else
			{
				// Inconsistent winding: force emit (safe direction).
				edges.push_back({pos_of(e0.a), pos_of(e0.b), pos_of(e0.c), pos_of(e0.c)});
				++stats.inconsistent;
			}
		}
		else if (n == 1)
		{
			const auto & e0 = dirs[i];
			edges.push_back({pos_of(e0.a), pos_of(e0.b), pos_of(e0.c), pos_of(e0.c)});
			++stats.boundary;
		}
		else
		{
			// Non-manifold (3+): keep first twin pair, force emit.
			const auto & e0 = dirs[i];
			edges.push_back({pos_of(e0.a), pos_of(e0.b), pos_of(e0.c), pos_of(e0.c)});
			++stats.non_manifold;
		}
		i = j;
	}
	stats.edges = edges.size();
	return {std::move(edges), stats};
}

namespace
{
struct error_category : std::error_category
{
	const char * name() const noexcept override
	{
		return "fastgltf";
	}

	std::string message(int condition) const override
	{
		return std::string(fastgltf::getErrorMessage(static_cast<fastgltf::Error>(condition)));
	}
};
static error_category fastgltf_error_category;

std::vector<uint32_t> read_indices(fastgltf::Asset & asset, const fastgltf::Accessor & accessor)
{
	std::vector<uint32_t> out(accessor.count);
	switch (accessor.componentType)
	{
		case fastgltf::ComponentType::UnsignedByte: {
			std::vector<uint8_t> tmp(accessor.count);
			fastgltf::copyFromAccessor<uint8_t>(asset, accessor, tmp.data());
			std::ranges::copy(tmp, out.begin());
			break;
		}
		case fastgltf::ComponentType::UnsignedShort: {
			std::vector<uint16_t> tmp(accessor.count);
			fastgltf::copyFromAccessor<uint16_t>(asset, accessor, tmp.data());
			std::ranges::copy(tmp, out.begin());
			break;
		}
		case fastgltf::ComponentType::UnsignedInt:
			fastgltf::copyFromAccessor<uint32_t>(asset, accessor, out.data());
			break;
		default:
			throw std::runtime_error("Invalid index component type");
	}
	return out;
}

// Expand strip/fan index sequences into triangle lists, preserving winding
std::vector<uint32_t> expand_topology(fastgltf::PrimitiveType type, std::vector<uint32_t> indices)
{
	if (type == fastgltf::PrimitiveType::Triangles)
		return indices;

	std::vector<uint32_t> out;
	if (type == fastgltf::PrimitiveType::TriangleStrip)
	{
		out.reserve((indices.size() > 2 ? indices.size() - 2 : 0) * 3);
		for (size_t i = 0; i + 2 < indices.size(); ++i)
		{
			if (i % 2 == 0)
				out.insert(out.end(), {indices[i], indices[i + 1], indices[i + 2]});
			else
				out.insert(out.end(), {indices[i + 1], indices[i], indices[i + 2]});
		}
	}
	else if (type == fastgltf::PrimitiveType::TriangleFan)
	{
		out.reserve((indices.size() > 2 ? indices.size() - 2 : 0) * 3);
		for (size_t i = 1; i + 1 < indices.size(); ++i)
			out.insert(out.end(), {indices[0], indices[i], indices[i + 1]});
	}
	return out;
}

void append_primitive(
        fastgltf::Asset & asset,
        const fastgltf::Primitive & primitive,
        const glm::mat4 & world_transform,
        bool flip_winding,
        passthrough_mesh::triangle_soup & out)
{
	using enum fastgltf::PrimitiveType;
	if (primitive.type != Triangles and primitive.type != TriangleStrip and primitive.type != TriangleFan)
	{
		spdlog::debug("passthrough_mesh: skipping non-triangle primitive");
		return;
	}

	auto position_attr = primitive.findAttribute("POSITION");
	if (position_attr == primitive.attributes.cend())
	{
		spdlog::warn("passthrough_mesh: primitive without POSITION, skipping");
		return;
	}

	const fastgltf::Accessor & position_accessor = asset.accessors.at(position_attr->accessorIndex);

	// Bake node transform into positions, epsilon-welded: near-coincident
	// positions share one vertex (remap records the welded index per
	// position in accessor order).
	uint32_t base_vertex = out.vertices.size();
	size_t index_start = out.indices.size();
	std::vector<uint32_t> weld_remap;
	weld_remap.reserve(position_accessor.count);
	std::unordered_map<weld_key, uint32_t, weld_hash> welded;
	fastgltf::iterateAccessor<glm::vec3>(asset, position_accessor, [&](glm::vec3 position) {
		glm::vec3 world = glm::vec3(world_transform * glm::vec4(position, 1));
		weld_key k{std::bit_cast<uint32_t>(world.x), std::bit_cast<uint32_t>(world.y), std::bit_cast<uint32_t>(world.z)};
		auto it = welded.find(k);
		if (it != welded.end())
		{
			weld_remap.push_back(it->second - base_vertex);
			return;
		}
		uint32_t idx = (uint32_t)out.vertices.size();
		out.vertices.push_back({world.x, world.y, world.z});
		welded.emplace(k, idx);
		weld_remap.push_back(idx - base_vertex);
	});

	std::vector<uint32_t> indices;
	if (primitive.indicesAccessor)
		indices = read_indices(asset, asset.accessors.at(*primitive.indicesAccessor));
	else
	{
		indices.resize(position_accessor.count);
		std::iota(indices.begin(), indices.end(), 0);
	}

	indices = expand_topology(primitive.type, std::move(indices));

	if (indices.size() % 3 != 0)
	{
		spdlog::warn("passthrough_mesh: index count not a multiple of 3, skipping primitive");
		out.vertices.resize(base_vertex);
		return;
	}

	// Indices address positions in accessor order: remap through the weld.
	// Also rejects out-of-range values before any vertex dereference.
	if (!indices.empty() && *std::max_element(indices.begin(), indices.end()) >= weld_remap.size())
	{
		spdlog::warn("passthrough_mesh: index out of range, skipping primitive");
		out.vertices.resize(base_vertex);
		return;
	}
	for (uint32_t & idx : indices)
		idx = base_vertex + weld_remap[idx];

	if (out.indices.size() + indices.size() > max_triangles * 3)
		throw std::runtime_error("passthrough_mesh: model exceeds triangle budget");

	// Reject non-finite baked positions with the same skip discipline
	// as malformed indices below: NaN/inf would otherwise pin the
	// worklist (all comparisons false) until the budget throw, after
	// burning ~100MB transiently.
	for (size_t v = base_vertex; v < out.vertices.size(); ++v)
	{
		const XrVector3f & p = out.vertices[v];
		if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
		{
			spdlog::warn("passthrough_mesh: non-finite baked position, skipping primitive");
			out.vertices.resize(base_vertex);
			return;
		}
	}

	// Bound baked edge length (world meters): with cull-back, a huge
	// triangle straddling the near plane or projecting far outside the
	// guardband mis-culls wholesale (pre-clip facing on unclipped
	// coords is spec-legal; Adreno guardband is 32k-class, overflow
	// measured to ~19M px). Small triangles bound both the straddle
	// damage and the fully-in-front coordinate span. Threshold is
	// pre-placement-scale: placements are ~uniform-1 (config default),
	// larger scales re-open the hole proportionally. Midpoints are
	// bitwise-identical across shared edges, so the output has no
	// positional gaps (up to float rounding); the conforming closure
	// below additionally removes topological T-junctions (hanging
	// vertices whose long-vs-halved edge specs disagree in the
	// rasterizer's fixed-point equations: deficit pinholes). Each
	// half keeps the parent orientation, so flip_winding composes
	// unchanged.
	constexpr float kMaxBakeEdge = 0.10f;
	constexpr float kMaxBakeEdge2 = kMaxBakeEdge * kMaxBakeEdge;
	auto edge_len2 = [&](uint32_t u, uint32_t v) {
		float dx = out.vertices[u].x - out.vertices[v].x;
		float dy = out.vertices[u].y - out.vertices[v].y;
		float dz = out.vertices[u].z - out.vertices[v].z;
		return dx * dx + dy * dy + dz * dz;
	};
	auto v3 = [&](uint32_t v) {
		const XrVector3f & p = out.vertices[v];
		return glm::vec3(p.x, p.y, p.z);
	};
	// Longest^2 / 2area (1.155 equilateral, huge for needles).
	auto aspect_of = [](const glm::vec3 & a, const glm::vec3 & b, const glm::vec3 & c) {
		glm::vec3 e1 = b - a, e2 = c - a, e3 = c - b;
		float area2 = glm::length(glm::cross(e1, e2));
		if (!(area2 > 0))
			return std::numeric_limits<float>::max();
		float m = std::max({glm::dot(e1, e1), glm::dot(e2, e2), glm::dot(e3, e3)});
		return m / area2;
	};
	std::vector<std::array<uint32_t, 3>> pending;
	pending.reserve(indices.size() / 3 + 1);
	for (size_t i = 0; i < indices.size(); i += 3)
	{
		uint32_t a = indices[i];
		uint32_t b = indices[i + 1];
		uint32_t c = indices[i + 2];
		if (a == b || b == c || c == a)
			continue; // welded-degenerate: zero area, rasterizer-culls anyway
		if (flip_winding)
			pending.push_back({a, c, b});
		else
			pending.push_back({a, b, c});
	}
	while (!pending.empty())
	{
		auto tri = pending.back();
		pending.pop_back();
		uint32_t a = tri[0], b = tri[1], c = tri[2];
		float e_ab = edge_len2(a, b);
		float e_bc = edge_len2(b, c);
		float e_ca = edge_len2(c, a);
		if (e_ab <= kMaxBakeEdge2 && e_bc <= kMaxBakeEdge2 && e_ca <= kMaxBakeEdge2)
		{
			if (out.indices.size() + 3 > max_triangles * 3)
				throw std::runtime_error("passthrough_mesh: model exceeds triangle budget");
			out.indices.insert(out.indices.end(), {a, b, c});
			continue;
		}
		if (out.indices.size() + (pending.size() + 2) * 3 > max_triangles * 3)
			throw std::runtime_error("passthrough_mesh: model exceeds triangle budget");
		// Aspect-capped split: among splits that strictly shrink the max
		// edge, pick the one minimizing the worst child aspect; tie-break
		// longest edge. Blind longest-edge splitting preserves needles
		// (children stay needles); the cap prefers splits that actually
		// improve shape. The longest edge usually qualifies; tied-max/NaN
		// fall back to the longest-edge halve below, so progress is
		// guaranteed. The halves tile the parent (bitwise-identical
		// scalar midpoint everywhere: watertight up to float rounding)
		// with the parent orientation (cyclic order preserved).
		float parent_max = std::max({e_ab, e_bc, e_ca});
		uint32_t e0 = a, e1 = b, w = c;
		{
			float best_worst = std::numeric_limits<float>::max();
			float best_len = -1.0f;
			bool have = false;
			uint32_t eu[3] = {a, b, c};
			uint32_t ev[3] = {b, c, a};
			uint32_t ew[3] = {c, a, b};
			for (int k = 0; k < 3; ++k)
			{
				uint32_t u = eu[k], v = ev[k], ow = ew[k];
				float len = edge_len2(u, v);
				glm::vec3 pu = v3(u), pv = v3(v), pw = v3(ow);
				glm::vec3 m = (pu + pv) * 0.5f;
				auto d2 = [](const glm::vec3 & x, const glm::vec3 & y) {
					glm::vec3 d = x - y;
					return glm::dot(d, d);
				};
				float cmax = std::max({d2(pu, m), d2(m, pw), d2(pu, pw), d2(m, pv), d2(pv, pw)});
				if (!(cmax < parent_max))
					continue; // no progress: longest edge survives
				float worst = std::max(aspect_of(pu, m, pw), aspect_of(m, pv, pw));
				if (worst < best_worst || (worst == best_worst && len > best_len))
				{
					best_worst = worst;
					best_len = len;
					e0 = u;
					e1 = v;
					w = ow;
					have = true;
				}
			}
			if (!have)
			{
				// Degenerate-adjacent fallback (NaN-safe): halve the longest.
				if (e_ab >= e_bc && e_ab >= e_ca)
				{
					e0 = a;
					e1 = b;
					w = c;
				}
				else if (e_bc >= e_ca)
				{
					e0 = b;
					e1 = c;
					w = a;
				}
				else
				{
					e0 = c;
					e1 = a;
					w = b;
				}
			}
		}
		// Midpoint from the selected edge endpoints (identical float ops
		// everywhere: shared edges stay bitwise-identical). The finite
		// check doubles as the NaN guard: NaN comparisons select nothing
		// above, the fallback halves the longest, and a NaN midpoint
		// skips the primitive here instead of looping to the budget.
		XrVector3f p0 = out.vertices[e0];
		XrVector3f p1 = out.vertices[e1];
		float mx = (p0.x + p1.x) * 0.5f;
		float my = (p0.y + p1.y) * 0.5f;
		float mz = (p0.z + p1.z) * 0.5f;
		if (!std::isfinite(mx) || !std::isfinite(my) || !std::isfinite(mz))
		{
			spdlog::warn("passthrough_mesh: midpoint overflow, skipping primitive");
			out.vertices.resize(base_vertex);
			out.indices.resize(index_start);
			return;
		}
		out.vertices.push_back({mx, my, mz});
		uint32_t m = (uint32_t)out.vertices.size() - 1;
		// Halves (e0,m,w)+(m,e1,w): cyclic reorderings of the verified
		// per-edge cases, so orientation is preserved by construction.
		pending.push_back({e0, m, w});
		pending.push_back({m, e1, w});
	}
	conforming_closure_range(out, base_vertex, index_start);
}

void append_node(
        fastgltf::Asset & asset,
        size_t node_index,
        const glm::mat4 & parent_transform,
        passthrough_mesh::triangle_soup & out)
{
	const fastgltf::Node & node = asset.nodes[node_index];

	auto TRS = std::get<fastgltf::TRS>(node.transform);
	glm::mat4 local =
	        glm::translate(glm::mat4(1), glm::make_vec3(TRS.translation.data())) *
	        glm::mat4(glm::make_quat(TRS.rotation.data())) *
	        glm::scale(glm::mat4(1), glm::make_vec3(TRS.scale.data()));
	glm::mat4 world = parent_transform * local;

	// A negative determinant mirrors the geometry: compensate by flipping
	// triangle winding so the runtime sees consistent CCW front faces
	bool flip = glm::determinant(world) < 0;

	if (node.meshIndex)
	{
		for (const fastgltf::Primitive & primitive: asset.meshes[*node.meshIndex].primitives)
			append_primitive(asset, primitive, world, flip, out);
	}

	for (size_t child: node.children)
		append_node(asset, child, world, out);
}

} // namespace

// Conforming closure over [range_vert, ...)/[range_start, ...): force-split
// tris at hanging vertices so both sides of every shared edge reference
// identical endpoints. The rasterizer evaluates each side's edge equation
// in fixed point; a long edge on one side vs two halves on the other can
// disagree by snapping and drop the pixel on both sides (deficit pinhole,
// which tier-1 erode widens into a hairline). Midpoints computed from
// identical endpoints are bitwise-identical, so the lookup below never
// creates vertices — it only reuses existing ones. Boundary/unconnected
// matches are split too (exact tilings either way; only costs tris).
// Runs per primitive inside append_primitive and once globally over the
// whole soup in flatten_gltf (cross-primitive duplicates need the global
// pass: weld/posmap are per-primitive).
void conforming_closure_range(passthrough_mesh::triangle_soup & out, uint32_t range_vert, size_t range_start)
{
	auto same_pos = [&](uint32_t u, uint32_t v) {
		const XrVector3f & a = out.vertices[u];
		const XrVector3f & b = out.vertices[v];
		return std::bit_cast<uint32_t>(a.x) == std::bit_cast<uint32_t>(b.x) &&
		       std::bit_cast<uint32_t>(a.y) == std::bit_cast<uint32_t>(b.y) &&
		       std::bit_cast<uint32_t>(a.z) == std::bit_cast<uint32_t>(b.z);
	};
	// Vertex set is fixed by closure (no pushes), so build once.
	std::unordered_map<weld_key, uint32_t, weld_hash> posmap;
	for (size_t v = range_vert; v < out.vertices.size(); ++v)
	{
		const XrVector3f & p = out.vertices[v];
		posmap[{std::bit_cast<uint32_t>(p.x), std::bit_cast<uint32_t>(p.y), std::bit_cast<uint32_t>(p.z)}] =
		        (uint32_t)v;
	}
	constexpr size_t kMaxClosureRounds = 64;
	for (size_t round = 0; round < kMaxClosureRounds; ++round)
	{
		struct hanging
		{
			size_t pos;
			uint32_t e0, e1, w, m;
		};
		std::vector<hanging> splits;
		for (size_t p = range_start; p + 3 <= out.indices.size(); p += 3)
		{
			uint32_t t[3] = {out.indices[p], out.indices[p + 1], out.indices[p + 2]};
			if (t[0] == t[1] || t[1] == t[2] || t[2] == t[0] || same_pos(t[0], t[1]) ||
			    same_pos(t[1], t[2]) || same_pos(t[2], t[0]))
				continue;
			for (int k = 0; k < 3; ++k)
			{
				uint32_t a = t[k], b = t[(k + 1) % 3], c = t[(k + 2) % 3];
				const XrVector3f & pa = out.vertices[a];
				const XrVector3f & pb = out.vertices[b];
				weld_key mk{std::bit_cast<uint32_t>((pa.x + pb.x) * 0.5f),
				            std::bit_cast<uint32_t>((pa.y + pb.y) * 0.5f),
				            std::bit_cast<uint32_t>((pa.z + pb.z) * 0.5f)};
				auto it = posmap.find(mk);
				if (it == posmap.end())
					continue;
				uint32_t m = it->second;
				if (same_pos(m, a) || same_pos(m, b) || same_pos(m, c))
					continue;
				splits.push_back({p, a, b, c, m});
				break; // one split per tri per round (two would rewrite the same slots)
			}
		}
		if (splits.empty())
			break;
		if (round + 1 == kMaxClosureRounds)
			spdlog::warn("passthrough_mesh: conforming closure cap hit, residual T-junctions remain");
		for (auto & s : splits)
		{
			if (out.indices.size() + 3 > max_triangles * 3)
				throw std::runtime_error("passthrough_mesh: model exceeds triangle budget");
			out.indices[s.pos] = s.e0;
			out.indices[s.pos + 1] = s.m;
			out.indices[s.pos + 2] = s.w;
			out.indices.insert(out.indices.end(), {s.m, s.e1, s.w});
		}
	}
}

passthrough_mesh::triangle_soup passthrough_mesh::flatten_gltf(const std::filesystem::path & path)
{
	utils::mapped_file mapping(path);
	auto data_buffer = fastgltf::GltfDataBuffer::FromBytes(mapping.data(), mapping.size());
	if (auto error = data_buffer.error(); error != fastgltf::Error::None)
		throw std::system_error((int)error, fastgltf_error_category);

	fastgltf::Parser parser;
	auto options =
	        fastgltf::Options::DontRequireValidAssetMember |
	        fastgltf::Options::AllowDouble |
	        fastgltf::Options::DecomposeNodeMatrices;

	auto expected_asset = parser.loadGltf(data_buffer.get(), path.parent_path(), options);
	if (auto error = expected_asset.error(); error != fastgltf::Error::None)
		throw std::system_error((int)error, fastgltf_error_category);

	fastgltf::Asset asset = std::move(expected_asset.get());

	// Resolve external buffer URIs relative to the model file.
	// .glb buffers and data URIs need no handling; keep file mappings alive.
	std::vector<utils::mapped_file> external_buffers;
	for (fastgltf::Buffer & buffer: asset.buffers)
	{
		if (std::holds_alternative<fastgltf::sources::URI>(buffer.data))
		{
			fastgltf::sources::URI uri = std::get<fastgltf::sources::URI>(buffer.data);
			if (!uri.uri.isLocalPath())
				throw std::runtime_error("passthrough_mesh: non-local buffer URI not supported");
			auto buffer_path = path.parent_path() / std::filesystem::path(uri.uri.path());
			auto & mapping = external_buffers.emplace_back(buffer_path);
			std::span<const std::byte> bytes{mapping.data(), mapping.size()};
			buffer.data = fastgltf::sources::ByteView{
			        fastgltf::span<const std::byte>(bytes.data() + uri.fileByteOffset, bytes.size() - uri.fileByteOffset),
			        fastgltf::MimeType::None};
		}
	}

	triangle_soup out;

	// Root nodes: default scene when set, else every node without a parent
	std::vector<size_t> roots;
	if (asset.defaultScene and *asset.defaultScene < asset.scenes.size())
	{
		const auto & scene_nodes = asset.scenes[*asset.defaultScene].nodeIndices;
		roots.assign(scene_nodes.begin(), scene_nodes.end());
	}
	else
	{
		std::unordered_set<size_t> children;
		for (const auto & node: asset.nodes)
			children.insert(node.children.begin(), node.children.end());
		for (size_t i = 0; i < asset.nodes.size(); ++i)
		{
			if (not children.contains(i))
				roots.push_back(i);
		}
	}

	for (size_t root: roots)
		append_node(asset, root, glm::mat4(1), out);

	if (out.vertices.empty() or out.indices.empty())
		throw std::runtime_error("passthrough_mesh: no triangles found in model");

	// Global conforming pass: heals cross-primitive T-junctions (the
	// per-primitive pass inside append_primitive only sees its own
	// range). No new vertices; idempotent on conformed ranges.
	conforming_closure_range(out, 0, 0);

	spdlog::info("passthrough_mesh: {} -> {} vertices, {} triangles", path.string(), out.vertices.size(), out.indices.size() / 3);
	return out;
}

namespace passthrough_mesh
{
// Latitude-longitude grid, 16x8 (256 tris incl. degenerate pole rows:
// harmless, rasterizer culls zero-area). Quad (i,j): a=(i,j), b=(i,j+1),
// d=(i+1,j), e=(i+1,j+1) with latitude increasing upward; tris (a,e,b) +
// (a,d,e) wind outward-CCW (verified: equator quad gives +X normals).
std::vector<XrVector3f> unit_sphere_positions()
{
	constexpr int rows = 8, cols = 16;
	constexpr float pi = 3.14159265358979323846f;
	std::vector<XrVector3f> out;
	out.reserve((size_t)rows * cols * 6);
	auto at = [&](int i, int j) {
		float lat = -pi / 2.f + pi * (float)i / (float)rows;
		float lon = 2.f * pi * (float)(j % cols) / (float)cols;
		float cl = std::cos(lat);
		return XrVector3f{cl * std::cos(lon), std::sin(lat), cl * std::sin(lon)};
	};
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			XrVector3f a = at(i, j), b = at(i, j + 1), d = at(i + 1, j), e = at(i + 1, j + 1);
			out.push_back(a);
			out.push_back(e);
			out.push_back(b);
			out.push_back(a);
			out.push_back(d);
			out.push_back(e);
		}
	}
	return out;
}

// Closed unit cylinder, radius 1, y in [-1,1]: 16-sided wall plus top
// and bottom cap fans. Side quad (j,j+1): tris (b0,t0,t1) + (b0,t1,b1)
// give +radial normals (spot-checked at lon 0); caps (C,t1,t0) top /
// (C,b0,b1) bottom give +/-Y. All outward-CCW, matching the sphere
// above so the same cull-front exit-wall pipe serves both.
std::vector<XrVector3f> unit_cylinder_positions()
{
	constexpr int segs = 16;
	constexpr float pi = 3.14159265358979323846f;
	std::vector<XrVector3f> out;
	out.reserve((size_t)segs * 12);
	auto ring = [&](float y, int j) {
		float lon = 2.f * pi * (float)(j % segs) / (float)segs;
		return XrVector3f{std::cos(lon), y, std::sin(lon)};
	};
	for (int j = 0; j < segs; ++j)
	{
		XrVector3f b0 = ring(-1.f, j), b1 = ring(-1.f, j + 1);
		XrVector3f t0 = ring(1.f, j), t1 = ring(1.f, j + 1);
		// Wall (outward).
		out.push_back(b0);
		out.push_back(t0);
		out.push_back(t1);
		out.push_back(b0);
		out.push_back(t1);
		out.push_back(b1);
		// Top cap (+Y): fan (C,t1,t0).
		out.push_back(XrVector3f{0, 1, 0});
		out.push_back(t1);
		out.push_back(t0);
		// Bottom cap (-Y): fan (C,b0,b1).
		out.push_back(XrVector3f{0, -1, 0});
		out.push_back(b0);
		out.push_back(b1);
	}
	return out;
}
} // namespace passthrough_mesh
