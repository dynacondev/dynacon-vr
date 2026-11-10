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
#include <cmath>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/quaternion.hpp>
#include <numeric>
#include <spdlog/spdlog.h>
#include <unordered_set>

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

// Sanity cap: refuse absurd meshes rather than handing them to the runtime
constexpr size_t max_triangles = 4'000'000;

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

	// Bake node transform into positions
	uint32_t base_vertex = out.vertices.size();
	fastgltf::iterateAccessor<glm::vec3>(asset, position_accessor, [&](glm::vec3 position) {
		glm::vec3 world = glm::vec3(world_transform * glm::vec4(position, 1));
		out.vertices.push_back({world.x, world.y, world.z});
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

	if (out.indices.size() + indices.size() > max_triangles * 3)
		throw std::runtime_error("passthrough_mesh: model exceeds triangle budget");

	for (size_t i = 0; i < indices.size(); i += 3)
	{
		uint32_t a = indices[i] + base_vertex;
		uint32_t b = indices[i + 1] + base_vertex;
		uint32_t c = indices[i + 2] + base_vertex;
		if (flip_winding)
			out.indices.insert(out.indices.end(), {a, c, b});
		else
			out.indices.insert(out.indices.end(), {a, b, c});
	}
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
} // namespace passthrough_mesh
