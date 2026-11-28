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

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>
#include <openxr/openxr.h>

namespace passthrough_mesh
{
// Baked, hierarchy-space (model-space) triangle list for the mask
// raster. World placement arrives later via per-eye MVPs, never
// baked here.
// Only positions matter to the runtime: no normals, UVs or materials.
struct triangle_soup
{
	std::vector<XrVector3f> vertices;
	std::vector<uint32_t> indices; // 3 per triangle
};

// Explicit silhouette band edge (48B, inline positions): a,b are the
// edge endpoints, c0/c1 the opposite vertices of the two incident
// triangles. c1==c0 marks a forced emit (boundary, non-manifold or
// inconsistent winding): the band shader always covers it (fail-safe
// over-reveal). Built once at map load from the post-closure soup;
// positions are model-space like the soup above.
struct silhouette_edge
{
	XrVector3f a, b, c0, c1;
};
struct silhouette_edge_stats
{
	size_t edges = 0;
	size_t boundary = 0;
	size_t non_manifold = 0;
	size_t inconsistent = 0;
};
std::pair<std::vector<silhouette_edge>, silhouette_edge_stats> build_silhouette_edges(const triangle_soup & soup);

// Parse a .glb/.gltf file and bake all node transforms into a single
// indexed triangle list. Skinned meshes use their rest pose, morph targets,
// points/lines are skipped. Throws std::exception on parse errors.
triangle_soup flatten_gltf(const std::filesystem::path & path);

// Unit-sphere positions for the proximity depth field (non-indexed
// triangles, outward-CCW winding so cull-front keeps the inside/exit
// wall from any viewpoint). One shared generator, expanded with
// per-frame MVPs by the hidden mask record.
std::vector<XrVector3f> unit_sphere_positions();

// Unit-cylinder positions for the proximity depth field (non-indexed
// triangles, outward-CCW winding like the sphere above). Radius 1,
// y in [-1,1] (inscribed in the unit cube, so the same cube-corner
// scissor stays a conservative superset): the record scales it to
// (r, depth/2, r) around the body column center.
std::vector<XrVector3f> unit_cylinder_positions();
} // namespace passthrough_mesh
