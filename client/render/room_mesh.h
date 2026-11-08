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

#include "xr/instance.h"
#include "xr/session.h"
#include <glm/vec3.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace wivrn
{

// Room triangle mesh for kiosk obstacle avoidance (Quest only). Queries the
// Space Setup room mesh once via XR_META_spatial_entity_room_mesh, caches it
// in world space behind a uniform grid, and answers closest-surface tests.
// Everything else degrades gracefully: no permission, no extensions, no mesh
// data, non-Quest runtimes — all inert, exactly one warn each, kiosk flies
// exactly as before. All calls render-thread only (same as kiosk_planes).
class room_mesh
{
public:
	room_mesh(xr::instance & instance, xr::session & session);
	~room_mesh();

	room_mesh(const room_mesh &) = delete;
	room_mesh & operator=(const room_mesh &) = delete;

	// Route scene lifetime events here (query results, capture complete).
	void on_xr_event(const xr::event & event);

	// Drive the state machine; call every frame the kiosk may show. Fires
	// the async query once permission + extensions are ready, (re)locates
	// the cached mesh when the world space changes.
	void ensure(XrSpace world, XrTime now);

	// True once mesh triangles are cached in the current world space.
	bool ready() const
	{
		return state_ == state::ready;
	}

	// Closest mesh surface to p within margin: push_out moves p away along
	// (p - closest) and returns true. clear_of is the pure test for the
	// sampler (reject when false). Mesh triangles and baked boxes constrain
	// together: either one can reject or push, so partial meshes and manual
	// walls cover each other's holes. Both no-op when not ready, and both
	// fail closed while committed room data is still unbaked (locate
	// pending/failed) so sampling never flies blind through walls that are
	// merely unplaced.
	bool clear_of(const glm::vec3 & p, float margin) const;
	bool push_out(glm::vec3 & p, float margin) const;

	// Distance from p to the nearest baked surface (triangles and boxes),
	// capped at max_dist. Returns 0 when sampling must fail closed
	// (committed data awaiting bake) and max_dist when nothing is within
	// range or no room data is baked. Used to fit segment legs to the
	// free space around the plane instead of sampling blind.
	float clearance(const glm::vec3 & p, float max_dist) const;

	// System room-setup flow (XR_FB_scene_capture). Only call when the
	// settings row offers it (can_capture): the system takes over the
	// headset, and the completion event re-triggers the query by itself.
	bool can_capture();
	// True once the first probe settles (room found or definitively
	// absent): kiosk hides pre-decision flights so nothing visibly
	// clips through a wall that was merely not loaded yet.
	bool decided() const
	{
		return decided_;
	}
	// Committed room data exists but nothing is baked into world space
	// yet (locate pending/failed): sampling must fail closed, not fly
	// blind through walls that are merely unplaced.
	bool awaiting_bake() const
	{
		return state_ == state::ready && (not obstacles_.empty() || not local_indices_.empty()) &&
		       world_boxes_.empty() && world_tris_.empty();
	}
	void request_capture();

	// One-line status for the lobby settings row and logcat.
	std::string status() const;

private:
	enum class state
	{
		unsupported,    // extension missing (non-Quest etc.): inert forever
		need_permission,// waiting on the USE_SCENE runtime grant
		idle,           // ready to fire the async query
		querying,       // waiting on the results-available event
		ready,          // mesh cached, located into located_world_
		no_data,        // query ok but no room mesh (no Space Setup here?)
		failed,         // query/retrieve/fetch failed (logged, retryable)
	};

	xr::instance * inst_ = nullptr;
	xr::session * sess_ = nullptr;

	bool have_query_ = false;
	bool have_room_mesh_ = false;
	bool have_scene_ = false;
	bool have_capture_ = false;
	bool permission_logged_ = false;
	bool extensions_logged_ = false;
	XrTime last_now_ = 0; // last ensure() time
	XrTime next_poll_ = 0; // next scheduled re-query (no_data/ready/failed backoff)
	XrTime last_perm_check_ = 0;
	bool perm_cached_ = false;

	PFN_xrQuerySpacesFB xrQuerySpacesFB{};
	PFN_xrRetrieveSpaceQueryResultsFB xrRetrieveSpaceQueryResultsFB{};
	PFN_xrSetSpaceComponentStatusFB xrSetSpaceComponentStatusFB{};
	PFN_xrGetSpaceComponentStatusFB xrGetSpaceComponentStatusFB{};
	PFN_xrGetSpaceRoomMeshMETA xrGetSpaceRoomMeshMETA{};
	PFN_xrGetSpaceSemanticLabelsFB xrGetSpaceSemanticLabelsFB{};
	PFN_xrGetSpaceBoundary2DFB xrGetSpaceBoundary2DFB{};
	PFN_xrGetSpaceBoundingBox2DFB xrGetSpaceBoundingBox2DFB{};
	PFN_xrGetSpaceBoundingBox3DFB xrGetSpaceBoundingBox3DFB{};
	PFN_xrGetSpaceRoomMeshFaceIndicesMETA xrGetSpaceRoomMeshFaceIndicesMETA{};
	PFN_xrRequestSceneCaptureFB xrRequestSceneCaptureFB{};

	state state_ = state::idle;
	XrAsyncRequestIdFB pending_request_{0};
	XrTime next_locate_ = 0; // backoff for total locate failure
	XrTime query_t0_ = 0; // last fire_query, for the querying watchdog below
	bool decided_ = false;
	bool saw_spaces_ = false;
	bool pending_ = false;
	bool results_pending_ = false;
	// Mesh-local triangles, baked to world space on locate.
	struct tri
	{
		glm::vec3 a, b, c;
	};
	// Room planes/volumes fallback (manual setups have no triangle mesh):
	// per-anchor local points (2D boundary at z=0, or 3D box corners),
	// baked to world AABBs alongside the mesh. Same keep-out semantics.
	struct obstacle
	{
		XrSpace space = XR_NULL_HANDLE;
		std::vector<glm::vec3> local;
		std::string label;
	};
	struct box
	{
		glm::vec3 mn{0.f}, mx{0.f};
	};
	// Probe scratch (async enable flow spans frames): candidate spaces,
	// per-space vert ranges, fetched verts/indices, and in-flight
	// enable requests (request id -> probe index).
	std::vector<XrSpace> probe_spaces_;
	std::vector<obstacle> probe_obstacles_;
	std::vector<std::pair<uint32_t, uint32_t>> probe_ranges_;
	std::vector<glm::vec3> probe_verts_;
	std::vector<uint32_t> probe_indices_;
	std::map<XrAsyncRequestIdFB, size_t> pending_enables_;
	// Filtered query chain (ROOM_MESH, BOUNDED_2D, BOUNDED_3D), then an
	// unfiltered census sweep; drained one at a time by fire_next_query.
	std::vector<std::optional<XrSpaceComponentTypeFB>> query_queue_;
	std::string fail_reason_;

	// Mesh anchor spaces from the query (caller-owned, destroyed here),
	// each with its vertex range into local_verts_ (poses differ per room).
	std::vector<XrSpace> mesh_spaces_;
	std::vector<std::pair<uint32_t, uint32_t>> space_vert_ranges_;
	std::vector<glm::vec3> local_verts_;
	std::vector<uint32_t> local_indices_; // 3 per triangle
	std::vector<obstacle> obstacles_;     // local polygons/boxes
	std::vector<box> world_boxes_;        // baked, same frame as mesh
	std::vector<tri> world_tris_;
	XrSpace located_world_ = XR_NULL_HANDLE;

	// Uniform grid over the world AABB, cell content = tri indices.
	glm::vec3 grid_min_{0.f};
	float grid_cell_ = 0.5f;
	glm::ivec3 grid_dim_{0};
	std::vector<std::vector<uint32_t>> grid_;

	void fire_query(std::optional<XrSpaceComponentTypeFB> component);
	void begin_cycle();
	void fire_next_query();
	void fetch_all_probe();
	// Fetch one probed space's mesh into the probe buffers.
	bool fetch_space_mesh(size_t probe_idx);
	bool fetch_planes(size_t probe_idx);
	void bake_obstacles(XrSpace world, XrTime now);
	// Commit probed mesh (or settle empty); destroys non-contributing spaces.
	void finish_probe();
	void on_query_results(XrAsyncRequestIdFB request_id);
	void on_capture_complete(XrResult result);
	void clear_mesh();
	bool locate(XrSpace world, XrTime now);
	void build_grid();
	bool closest(const glm::vec3 & p, float max_dist, glm::vec3 & closest_out) const;
	bool permission_granted();
};

} // namespace wivrn
