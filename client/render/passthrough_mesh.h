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

// Parse a .glb/.gltf file and bake all node transforms into a single
// indexed triangle list. Skinned meshes use their rest pose, morph targets,
// points/lines are skipped. Throws std::exception on parse errors.
triangle_soup flatten_gltf(const std::filesystem::path & path);

// Unit-sphere positions for the proximity depth field (non-indexed
// triangles, outward-CCW winding so cull-front keeps the inside/exit
// wall from any viewpoint). One shared generator, expanded with
// per-frame MVPs by the hidden mask record.
std::vector<XrVector3f> unit_sphere_positions();
} // namespace passthrough_mesh
