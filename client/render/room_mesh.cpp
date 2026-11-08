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

#include "room_mesh.h"

#include "android/permissions.h"
#include "xr/to_string.h"
#include <glm/geometric.hpp>
#include <cstring>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace wivrn
{

room_mesh::room_mesh(xr::instance & instance, xr::session & session) :
        inst_(&instance), sess_(&session)
{
	have_query_ = inst_->has_extension(XR_FB_SPATIAL_ENTITY_EXTENSION_NAME) &&
	              inst_->has_extension(XR_FB_SPATIAL_ENTITY_QUERY_EXTENSION_NAME);
	have_scene_ = inst_->has_extension(XR_FB_SCENE_EXTENSION_NAME);
#ifdef XR_META_SPATIAL_ENTITY_ROOM_MESH_EXTENSION_NAME
	have_room_mesh_ = inst_->has_extension(XR_META_SPATIAL_ENTITY_ROOM_MESH_EXTENSION_NAME);
#endif
#ifdef XR_FB_SCENE_CAPTURE_EXTENSION_NAME
	have_capture_ = inst_->has_extension(XR_FB_SCENE_CAPTURE_EXTENSION_NAME);
#endif
	if (have_query_ && (have_room_mesh_ || have_scene_))
	{
		try
		{
			xrSetSpaceComponentStatusFB = inst_->get_proc<PFN_xrSetSpaceComponentStatusFB>("xrSetSpaceComponentStatusFB");
			xrGetSpaceComponentStatusFB = inst_->get_proc<PFN_xrGetSpaceComponentStatusFB>("xrGetSpaceComponentStatusFB");
			xrQuerySpacesFB = inst_->get_proc<PFN_xrQuerySpacesFB>("xrQuerySpacesFB");
			xrRetrieveSpaceQueryResultsFB = inst_->get_proc<PFN_xrRetrieveSpaceQueryResultsFB>("xrRetrieveSpaceQueryResultsFB");
			if (have_room_mesh_)
			{
				xrGetSpaceRoomMeshMETA = inst_->get_proc<PFN_xrGetSpaceRoomMeshMETA>("xrGetSpaceRoomMeshMETA");
				xrGetSpaceRoomMeshFaceIndicesMETA = inst_->get_proc<PFN_xrGetSpaceRoomMeshFaceIndicesMETA>("xrGetSpaceRoomMeshFaceIndicesMETA");
			}
			if (have_scene_)
			{
				xrGetSpaceSemanticLabelsFB = inst_->get_proc<PFN_xrGetSpaceSemanticLabelsFB>("xrGetSpaceSemanticLabelsFB");
				xrGetSpaceBoundary2DFB = inst_->get_proc<PFN_xrGetSpaceBoundary2DFB>("xrGetSpaceBoundary2DFB");
				xrGetSpaceBoundingBox2DFB = inst_->get_proc<PFN_xrGetSpaceBoundingBox2DFB>("xrGetSpaceBoundingBox2DFB");
				xrGetSpaceBoundingBox3DFB = inst_->get_proc<PFN_xrGetSpaceBoundingBox3DFB>("xrGetSpaceBoundingBox3DFB");
			}
			if (have_capture_)
				xrRequestSceneCaptureFB = inst_->get_proc<PFN_xrRequestSceneCaptureFB>("xrRequestSceneCaptureFB");
		}
		catch (const std::system_error & e)
		{
			spdlog::warn("room_mesh: proc lookup failed ({}), kiosk flies unconstrained", e.what());
			have_query_ = false;
		}
	}
}

room_mesh::~room_mesh()
{
	clear_mesh();
}

bool room_mesh::permission_granted()
{
#ifdef __ANDROID__
	// check_permission is a JNI round-trip until granted (then cached
	// internally): throttle our own calls to one per 5 s.
	if (not perm_cached_ || last_now_ - last_perm_check_ > XrTime(5e9))
	{
		last_perm_check_ = last_now_;
		if (check_permission("com.oculus.permission.USE_SCENE"))
			perm_cached_ = true;
	}
	return perm_cached_;
#else
	return true;
#endif
}

void room_mesh::clear_mesh()
{
	for (XrSpace space: mesh_spaces_)
		xrDestroySpace(space);
	mesh_spaces_.clear();
	space_vert_ranges_.clear();
	obstacles_.clear();
	world_boxes_.clear();
	local_verts_.clear();
	local_indices_.clear();
	world_tris_.clear();
	grid_.clear();
	located_world_ = XR_NULL_HANDLE;
}

void room_mesh::ensure(XrSpace world, XrTime now)
{
	if (not have_query_ or (not have_room_mesh_ and not have_scene_))
	{
		if (not extensions_logged_)
		{
			extensions_logged_ = true;
			spdlog::warn("room_mesh: room query extensions missing, kiosk flies unconstrained");
		}
		state_ = state::unsupported;
		decided_ = true; // non-Quest: never hide the fleet waiting on data that can't come
		return;
	}
	if (not permission_granted())
	{
		if (not permission_logged_)
		{
			permission_logged_ = true;
			spdlog::info("room_mesh: waiting for USE_SCENE grant, kiosk flies unconstrained");
		}
		state_ = state::need_permission;
		return;
	}
	last_now_ = now;
	switch (state_)
	{
	case state::need_permission:
	case state::idle:
		begin_cycle();
		break;
	case state::failed:
		// Back off: a synchronous failure would otherwise re-fire (and
		// log) every frame.
		if (now >= next_poll_)
			begin_cycle();
		break;
	case state::querying:
		// Watchdog: without this a lost completion event strands the
		// provider here with avoidance neutered. Back to idle to retry.
		if (now - query_t0_ > XrTime(15e9))
		{
			pending_ = false;
			pending_enables_.clear();
			state_ = state::idle;
			spdlog::info("room_mesh: query timed out, retrying");
		}
		break;
	case state::no_data:
		// A room set up outside the app (system Settings) only shows up
		// on re-query: poll again every 30 s (armed by fire_query).
		if (now >= next_poll_)
			begin_cycle();
		break;
	case state::ready:
		// Rooms can be edited/deleted out from under us: re-query every
		// 5 min. The old data keeps serving until the new one commits.
		if (not pending_ && now >= next_poll_)
			begin_cycle();
		else if (world != located_world_)
			locate(world, now);
		break;
	case state::unsupported:
		break;
	}
}

void room_mesh::begin_cycle()
{
	// One filtered query per geometric component type, then an unfiltered
	// sweep (anchor census for rooms whose anchors match nothing). Fired
	// one at a time; results accumulate into the probe set.
	if (pending_)
		return;
	for (XrSpace space: probe_spaces_)
		xrDestroySpace(space);
	probe_spaces_.clear();
	probe_ranges_.clear();
	probe_verts_.clear();
	probe_indices_.clear();
	probe_obstacles_.clear();
	pending_enables_.clear();
	saw_spaces_ = false;
	query_queue_.clear();
	if (have_room_mesh_)
		query_queue_.push_back(XR_SPACE_COMPONENT_TYPE_ROOM_MESH_META);
	if (have_scene_)
	{
		query_queue_.push_back(XR_SPACE_COMPONENT_TYPE_BOUNDED_2D_FB);
		query_queue_.push_back(XR_SPACE_COMPONENT_TYPE_BOUNDED_3D_FB);
	}
	query_queue_.push_back(std::nullopt);
	fire_next_query();
}

void room_mesh::fire_next_query()
{
	if (not query_queue_.empty())
	{
		auto filter = query_queue_.back();
		query_queue_.pop_back();
		fire_query(filter);
		return;
	}
	// Chain exhausted with spaces collected (or none at all): enable
	// LOCATABLE explicitly (LOAD does not reliably do it), then fetch.
	for (size_t i = 0; i < probe_spaces_.size(); ++i)
	{
		XrSpaceComponentStatusFB status{
		        .type = XR_TYPE_SPACE_COMPONENT_STATUS_FB,
		        .next = nullptr,
		};
		bool enabled = false;
		if (XrResult res = xrGetSpaceComponentStatusFB(probe_spaces_[i], XR_SPACE_COMPONENT_TYPE_LOCATABLE_FB, &status); res == XR_SUCCESS)
			enabled = status.enabled && not status.changePending;
		if (enabled)
			continue;
		XrSpaceComponentStatusSetInfoFB set_info{
		        .type = XR_TYPE_SPACE_COMPONENT_STATUS_SET_INFO_FB,
		        .next = nullptr,
		        .componentType = XR_SPACE_COMPONENT_TYPE_LOCATABLE_FB,
		        .enabled = XR_TRUE,
		        .timeout = 0,
		};
		XrAsyncRequestIdFB request = 0;
		if (XrResult res = xrSetSpaceComponentStatusFB(probe_spaces_[i], &set_info, &request); res == XR_SUCCESS)
			pending_enables_[request] = i;
		else
			spdlog::info("room_mesh: cannot enable locatable on an anchor ({})", xr::to_string(res));
	}
	if (pending_enables_.empty())
	{
		fetch_all_probe();
		finish_probe();
	}
	// else: completions drain through on_xr_event into finish_probe().
}

void room_mesh::fetch_all_probe()
{
	for (size_t i = 0; i < probe_spaces_.size(); ++i)
	{
		if (have_room_mesh_ and fetch_space_mesh(i))
			continue;
		if (have_scene_)
			fetch_planes(i);
	}
}

void room_mesh::fire_query(std::optional<XrSpaceComponentTypeFB> component)
{
	// Mirrors Meta's canonical query shape (XrSceneManager sample): the
	// component filter chains a storage-location filter, and timeout is 0
	// (async completion still arrives via RESULTS_AVAILABLE/COMPLETE).
	// nullopt = unfiltered census sweep.
	XrSpaceStorageLocationFilterInfoFB storage_filter{
	        .type = XR_TYPE_SPACE_STORAGE_LOCATION_FILTER_INFO_FB,
	        .next = nullptr,
	        .location = XR_SPACE_STORAGE_LOCATION_LOCAL_FB,
	};
	XrSpaceComponentFilterInfoFB component_filter{
	        .type = XR_TYPE_SPACE_COMPONENT_FILTER_INFO_FB,
	        .next = &storage_filter,
	        .componentType = component ? *component : XR_SPACE_COMPONENT_TYPE_ROOM_MESH_META,
	};
	XrSpaceQueryInfoFB query{
	        .type = XR_TYPE_SPACE_QUERY_INFO_FB,
	        .next = nullptr,
	        .queryAction = XR_SPACE_QUERY_ACTION_LOAD_FB,
	        .maxResultCount = 128, // rooms routinely exceed a handful of anchors; truncated queries silently drop whole walls
	        .timeout = 0,
	        .filter = component ? reinterpret_cast<const XrSpaceFilterInfoBaseHeaderFB *>(&component_filter) : nullptr,
	        .excludeFilter = nullptr,
	};
	if (pending_)
		return;
	results_pending_ = false;
	XrAsyncRequestIdFB request = 0;
	XrResult res = xrQuerySpacesFB(*sess_, reinterpret_cast<const XrSpaceQueryInfoBaseHeaderFB *>(&query), &request);
	next_poll_ = last_now_ + XrTime(30e9);
	if (res != XR_SUCCESS)
	{
		state_ = state::failed;
		next_poll_ = last_now_ + XrTime(5e9);
		decided_ = true;
		fail_reason_ = std::string("xrQuerySpacesFB: ") + xr::to_string(res);
		spdlog::warn("room_mesh: query failed ({}), kiosk flies unconstrained", fail_reason_);
		return;
	}
	pending_request_ = request;
	pending_ = true;
	query_t0_ = last_now_;
	if (state_ != state::ready)
		state_ = state::querying; // re-polls keep serving the old bake
	spdlog::info("room_mesh: room query fired ({})", component ? "component filter" : "unfiltered census");
}

void room_mesh::on_xr_event(const xr::event & event)
{
	// RESULTS_AVAILABLE can fire per space; the list is only complete at
	// COMPLETE, so merely note arrival here and retrieve exactly once on
	// completion. (Retrieving on the first available event risks a partial
	// list; ignoring a success-COMPLETE with no availability strands the
	// state machine in `querying` forever.)
	if (event.header.type == XR_TYPE_EVENT_DATA_SPACE_QUERY_RESULTS_AVAILABLE_FB)
	{
		if (pending_ && event.space_query_results_available.requestId == pending_request_)
			results_pending_ = true;
	}
	else if (event.header.type == XR_TYPE_EVENT_DATA_SPACE_QUERY_COMPLETE_FB)
	{
		// Async query failure path: without this, a failed query (which
		// never produces RESULTS_AVAILABLE) would sit in `querying`
		// forever with no retry.
		if (pending_ && event.space_query_complete.requestId == pending_request_)
		{
			pending_ = false;
			if (event.space_query_complete.result != XR_SUCCESS)
			{
				state_ = state::failed;
				next_poll_ = last_now_ + XrTime(5e9);
				fail_reason_ = std::string("async query: ") + xr::to_string(event.space_query_complete.result);
				spdlog::warn("room_mesh: query failed ({}), kiosk flies unconstrained", fail_reason_);
			}
			else if (results_pending_)
			{
				on_query_results(pending_request_);
			}
			else
			{
				// Finished cleanly with zero spaces: continue the chain;
				// the unfiltered census at its end tells empty room from
				// unmatchable anchors.
				fire_next_query();
			}
		}
	}
	else if (event.header.type == XR_TYPE_EVENT_DATA_SPACE_SET_STATUS_COMPLETE_FB)
	{
		auto it = pending_enables_.find(event.space_set_status_complete.requestId);
		if (it != pending_enables_.end())
		{
			if (event.space_set_status_complete.result != XR_SUCCESS)
				spdlog::info("room_mesh: enable locatable refused ({})", xr::to_string(event.space_set_status_complete.result));
			pending_enables_.erase(it);
			if (pending_enables_.empty())
			{
				fetch_all_probe();
				finish_probe();
			}
		}
	}
	else if (event.header.type == XR_TYPE_EVENT_DATA_SCENE_CAPTURE_COMPLETE_FB)
	{
		on_capture_complete(event.scene_capture_complete.result);
	}
	else if (event.header.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING)
	{
		// Map shift / recenter: the baked world transform may be stale.
		// Force a re-locate on next ensure() (runs once, then clean).
		located_world_ = XR_NULL_HANDLE;
	}
}

void room_mesh::on_capture_complete(XrResult result)
{
	// Abandon any in-flight enables: the new cycle re-discovers from
	// scratch (stale completions find no entry and are ignored).
	pending_ = false;
	for (XrSpace space: probe_spaces_)
		xrDestroySpace(space);
	probe_spaces_.clear();
	probe_ranges_.clear();
	probe_verts_.clear();
	probe_indices_.clear();
	probe_obstacles_.clear();
	pending_enables_.clear();
	if (result == XR_SUCCESS)
	{
		spdlog::info("room_mesh: scene capture complete, re-querying room mesh");
		state_ = state::idle;
	}
	else
	{
		state_ = state::failed;
		next_poll_ = last_now_ + XrTime(5e9);
		fail_reason_ = std::string("scene capture: ") + xr::to_string(result);
		spdlog::warn("room_mesh: scene capture failed ({}), kiosk flies unconstrained", xr::to_string(result));
	}
}

void room_mesh::on_query_results(XrAsyncRequestIdFB request_id)
{
	pending_ = false;
	XrSpaceQueryResultsFB results{
	        .type = XR_TYPE_SPACE_QUERY_RESULTS_FB,
	        .next = nullptr,
	        .resultCapacityInput = 0,
	        .resultCountOutput = 0,
	        .results = nullptr,
	};
	auto fail = [&](const std::string & why) {
		// A failed *re-query* must not wipe a live mesh: only first-time
		// failures change state; live meshes keep serving on backoff.
		if (state_ != state::ready)
		{
			state_ = state::failed;
			next_poll_ = last_now_ + XrTime(5e9);
		}
		else
		{
			next_poll_ = last_now_ + XrTime(30e9);
		}
		fail_reason_ = why;
		decided_ = true;
		spdlog::warn("room_mesh: query retrieve failed ({}), kiosk flies unconstrained", fail_reason_);
	};
	if (XrResult res = xrRetrieveSpaceQueryResultsFB(*sess_, request_id, &results); res != XR_SUCCESS)
	{
		fail(std::string("retrieve count: ") + xr::to_string(res));
		return;
	}
	if (results.resultCountOutput == 0)
	{
		fire_next_query();
		return;
	}
	std::vector<XrSpaceQueryResultFB> items(results.resultCountOutput);
	results.resultCapacityInput = uint32_t(items.size());
	results.results = items.data();
	if (results.resultCountOutput >= 128)
		spdlog::warn("room_mesh: query hit the result cap, some anchors may be missing");
	if (XrResult res = xrRetrieveSpaceQueryResultsFB(*sess_, request_id, &results); res != XR_SUCCESS)
	{
		fail(std::string("retrieve: ") + xr::to_string(res));
		return;
	}
	for (uint32_t i = 0; i < results.resultCountOutput; ++i)
	{
		bool known = false;
		for (XrSpace s : probe_spaces_)
			known = known or s == items[i].space;
		if (not known)
		{
			probe_spaces_.push_back(items[i].space);
			probe_ranges_.push_back({0, 0});
		}
	}
	if (results.resultCountOutput > 0)
		saw_spaces_ = true;
	fire_next_query();
	return;
}

bool room_mesh::fetch_space_mesh(size_t probe_idx)
{
	XrSpace space = probe_spaces_[probe_idx];
	XrSpaceRoomMeshGetInfoMETA get_info{
	        .type = XR_TYPE_SPACE_ROOM_MESH_GET_INFO_META,
	        .next = nullptr,
	        .recognizedSemanticLabelCount = 0, // all faces, labels ignored
	        .recognizedSemanticLabels = nullptr,
	};
	XrRoomMeshMETA mesh{
	        .type = XR_TYPE_ROOM_MESH_META,
	        .next = nullptr,
	};
	if (XrResult res = xrGetSpaceRoomMeshMETA(space, &get_info, &mesh); res != XR_SUCCESS)
	{
		spdlog::info("room_mesh: space has no mesh to fetch ({})", xr::to_string(res));
		return false;
	}
	if (mesh.vertexCountOutput == 0 || mesh.faceCountOutput == 0)
		return false;
	std::vector<XrVector3f> verts(mesh.vertexCountOutput);
	std::vector<XrRoomMeshFaceMETA> faces(mesh.faceCountOutput);
	mesh.vertexCapacityInput = uint32_t(verts.size());
	mesh.vertices = verts.data();
	mesh.faceCapacityInput = uint32_t(faces.size());
	mesh.faces = faces.data();
	if (XrResult res = xrGetSpaceRoomMeshMETA(space, &get_info, &mesh); res != XR_SUCCESS)
	{
		spdlog::info("room_mesh: mesh fetch failed ({})", xr::to_string(res));
		return false;
	}
	// Shrink-wrap to the returned counts: the room can change between the
	// two calls, and stale trailing entries would become phantom zero
	// triangles at the origin.
	verts.resize(mesh.vertexCountOutput);
	faces.resize(mesh.faceCountOutput);
	uint32_t base = uint32_t(probe_verts_.size());
	for (const auto & v: verts)
		probe_verts_.push_back({v.x, v.y, v.z});
	probe_ranges_[probe_idx].first = base;
	probe_ranges_[probe_idx].second = uint32_t(probe_verts_.size()) - base;
	bool any = false;
	for (const auto & f: faces)
	{
		XrRoomMeshFaceIndicesMETA idx{
		        .type = XR_TYPE_ROOM_MESH_FACE_INDICES_META,
		        .next = nullptr,
		};
		if (XrResult res = xrGetSpaceRoomMeshFaceIndicesMETA(space, &f.uuid, &idx); res != XR_SUCCESS)
			continue;
		if (idx.indexCountOutput < 3)
			continue;
		std::vector<uint32_t> face_idx(idx.indexCountOutput);
		idx.indexCapacityInput = uint32_t(face_idx.size());
		idx.indices = face_idx.data();
		if (XrResult res = xrGetSpaceRoomMeshFaceIndicesMETA(space, &f.uuid, &idx); res != XR_SUCCESS)
			continue;
		face_idx.resize(idx.indexCountOutput);
		if (face_idx.size() < 3)
			continue;
		// triangle fan around index 0 (room mesh faces are convex;
		// concave n-gons would triangulate wrong — accepted, the
		// Roombox emits near-planar convex faces)
		for (uint32_t k = 1; k + 1 < uint32_t(face_idx.size()); ++k)
		{
			probe_indices_.push_back(base + face_idx[0]);
			probe_indices_.push_back(base + face_idx[k]);
			probe_indices_.push_back(base + face_idx[k + 1]);
		}
		any = true;
	}
	return any;
}

bool room_mesh::fetch_planes(size_t probe_idx)
{
	// Room planes/volumes fallback for anchors without a triangle mesh
	// (manual Space Setup has walls/floor/ceiling but no mesh). Fetches
	// the semantic labels (logged for forensics), then the tightest
	// available bound: 3D box first (solid volumes), else the 2D boundary
	// polygon, else the 2D bounding rect. Stored anchor-local; baked to
	// world boxes on locate().
	XrSpace space = probe_spaces_[probe_idx];
	std::string label;
	{
		XrSemanticLabelsFB labels{
		        .type = XR_TYPE_SEMANTIC_LABELS_FB,
		        .next = nullptr,
		};
		if (XrResult res = xrGetSpaceSemanticLabelsFB(*sess_, space, &labels); res == XR_SUCCESS && labels.bufferCountOutput > 1)
		{
			std::vector<char> buf(labels.bufferCountOutput, '\0');
			labels.bufferCapacityInput = uint32_t(buf.size());
			labels.buffer = buf.data();
			if (xrGetSpaceSemanticLabelsFB(*sess_, space, &labels) == XR_SUCCESS && labels.bufferCountOutput > 1)
				label.assign(buf.data(), strnlen(buf.data(), buf.size()));
		}
	}
	// 3D box first: solid volumes beat floating polygons for avoidance.
	{
		XrRect3DfFB box{};
		if (xrGetSpaceBoundingBox3DFB(*sess_, space, &box) == XR_SUCCESS &&
		    box.extent.width > 0.f && box.extent.height > 0.f && box.extent.depth > 0.f)
		{
			obstacle o;
			o.space = space;
			o.label = label;
			for (int ix = 0; ix < 2; ++ix)
				for (int iy = 0; iy < 2; ++iy)
					for (int iz = 0; iz < 2; ++iz)
						o.local.push_back({box.offset.x + (ix ? box.extent.width : 0.f),
						                     box.offset.y + (iy ? box.extent.height : 0.f),
						                     box.offset.z + (iz ? box.extent.depth : 0.f)});
			probe_obstacles_.push_back(std::move(o));
			spdlog::info("room_mesh: anchor '{}' cached as 3D box", label.empty() ? "?" : label);
			return true;
		}
	}
	// 2D boundary polygon (walls/floor/ceiling), z=0 in anchor frame.
	{
		XrBoundary2DFB boundary{
		        .type = XR_TYPE_BOUNDARY_2D_FB,
		        .next = nullptr,
		};
		if (xrGetSpaceBoundary2DFB(*sess_, space, &boundary) == XR_SUCCESS && boundary.vertexCountOutput >= 3)
		{
			std::vector<XrVector2f> verts(boundary.vertexCountOutput);
			boundary.vertexCapacityInput = uint32_t(verts.size());
			boundary.vertices = verts.data();
			if (xrGetSpaceBoundary2DFB(*sess_, space, &boundary) == XR_SUCCESS && boundary.vertexCountOutput >= 3)
			{
				verts.resize(boundary.vertexCountOutput);
				obstacle o;
				o.space = space;
				o.label = label;
				for (const auto & v: verts)
					o.local.push_back({v.x, v.y, 0.f});
				probe_obstacles_.push_back(std::move(o));
				spdlog::info("room_mesh: anchor '{}' cached as {}-gon", label.empty() ? "?" : label, verts.size());
				return true;
			}
		}
	}
	// 2D bounding rect fallback.
	{
		XrRect2Df rect{};
		if (xrGetSpaceBoundingBox2DFB(*sess_, space, &rect) == XR_SUCCESS && rect.extent.width > 0.f && rect.extent.height > 0.f)
		{
			obstacle o;
			o.space = space;
			o.label = label;
			o.local.push_back({rect.offset.x, rect.offset.y, 0.f});
			o.local.push_back({rect.offset.x + rect.extent.width, rect.offset.y, 0.f});
			o.local.push_back({rect.offset.x + rect.extent.width, rect.offset.y + rect.extent.height, 0.f});
			o.local.push_back({rect.offset.x, rect.offset.y + rect.extent.height, 0.f});
			probe_obstacles_.push_back(std::move(o));
			spdlog::info("room_mesh: anchor '{}' cached as bounding rect", label.empty() ? "?" : label);
			return true;
		}
	}
	return false;
}

void room_mesh::finish_probe()
{
	// Commit the probed room (or settle empty). Failed re-queries must not
	// wipe a live room; a failed first query backs off for retry.
	if (probe_indices_.empty() && probe_obstacles_.empty())
	{
		for (XrSpace space: probe_spaces_)
			xrDestroySpace(space);
		probe_spaces_.clear();
		probe_ranges_.clear();
		probe_verts_.clear();
		probe_indices_.clear();
		probe_obstacles_.clear();
		if (state_ != state::ready)
		{
			state_ = state::no_data;
			next_poll_ = last_now_ + XrTime(30e9);
		}
		else
		{
			next_poll_ = last_now_ + XrTime(30e9);
		}
		decided_ = true;
		if (saw_spaces_)
			spdlog::warn("room_mesh: anchors found but no usable geometry (manual wall setup?), kiosk flies unconstrained");
		else
			spdlog::warn("room_mesh: no room anchors in this space (no Space Setup here?), kiosk flies unconstrained");
		return;
	}
	// Drop anchors that contributed no geometry: they only produce
	// perennial locate warnings (some anchors never support LOCATABLE).
	{
		std::vector<XrSpace> kept_spaces;
		std::vector<std::pair<uint32_t, uint32_t>> kept_ranges;
		kept_spaces.reserve(probe_spaces_.size());
		kept_ranges.reserve(probe_ranges_.size());
		size_t pruned = 0;
		for (size_t i = 0; i < probe_spaces_.size(); ++i)
		{
			bool used = probe_ranges_[i].second > 0;
			if (not used)
				for (const auto & o: probe_obstacles_)
					if (o.space == probe_spaces_[i])
					{
						used = true;
						break;
					}
			if (used)
			{
				kept_spaces.push_back(probe_spaces_[i]);
				kept_ranges.push_back(probe_ranges_[i]);
			}
			else
			{
				xrDestroySpace(probe_spaces_[i]);
				++pruned;
			}
		}
		if (pruned)
			spdlog::info("room_mesh: pruned {} anchors with no geometry", pruned);
		probe_spaces_.swap(kept_spaces);
		probe_ranges_.swap(kept_ranges);
	}
	// Commit: old spaces die with the old room, new data locates next ensure().
	clear_mesh();
	mesh_spaces_ = std::move(probe_spaces_);
	space_vert_ranges_ = std::move(probe_ranges_);
	local_verts_ = std::move(probe_verts_);
	local_indices_ = std::move(probe_indices_);
	obstacles_ = std::move(probe_obstacles_);
	probe_spaces_.clear();
	probe_ranges_.clear();
	probe_verts_.clear();
	probe_indices_.clear();
	probe_obstacles_.clear();
	located_world_ = XR_NULL_HANDLE;
	next_locate_ = 0; // fresh data deserves an immediate bake attempt, not a stale retry backoff
	state_ = state::ready;
	decided_ = true;
	next_poll_ = last_now_ + XrTime(300e9);
	spdlog::info("room_mesh: cached {} triangles + {} planes from {} anchor(s)", local_indices_.size() / 3, obstacles_.size(), mesh_spaces_.size());
}

bool room_mesh::locate(XrSpace world, XrTime now)
{
	if (mesh_spaces_.empty())
		return false;
	if (now < next_locate_)
		return false;
	// Bake over the live vectors but keep the last-good bake on total
	// failure: a transient loss must not wipe live avoidance (which
	// would fail sampling closed while rendering unpushed).
	std::vector<tri> prev_tris = std::move(world_tris_);
	std::vector<box> prev_boxes = std::move(world_boxes_);
	grid_.clear();
	bake_obstacles(world, now);
	world_tris_.reserve(local_indices_.size() / 3);
	for (size_t s = 0; s < mesh_spaces_.size(); ++s)
	{
		XrSpaceLocation location{.type = XR_TYPE_SPACE_LOCATION, .next = nullptr};
		if (XrResult res = xrLocateSpace(mesh_spaces_[s], world, now, &location); res != XR_SUCCESS)
		{
			spdlog::warn("room_mesh: locate failed ({})", xr::to_string(res));
			continue;
		}
		if ((location.locationFlags & (XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) != (XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT))
			continue;
		// Only the owning space's own vertex range is transformed here;
		// indices are global (base-offset at fetch), so every triangle
		// lands via its own anchor's pose.
		const auto & p = location.pose.position;
		const auto & q = location.pose.orientation;
		glm::mat4 m = glm::translate(glm::mat4(1.f), {p.x, p.y, p.z}) * glm::mat4_cast(glm::quat(q.w, q.x, q.y, q.z));
		auto [vfirst, vcount] = space_vert_ranges_[s];
		for (size_t i = 0; i + 2 < local_indices_.size(); i += 3)
		{
			uint32_t ia = local_indices_[i], ib = local_indices_[i + 1], ic = local_indices_[i + 2];
			if (ia < vfirst || ia >= vfirst + vcount)
				continue;
			glm::vec3 a = glm::vec3(m * glm::vec4(local_verts_[ia], 1.f));
			glm::vec3 b = glm::vec3(m * glm::vec4(local_verts_[ib], 1.f));
			glm::vec3 c = glm::vec3(m * glm::vec4(local_verts_[ic], 1.f));
			world_tris_.push_back({a, b, c});
		}
	}
	if (world_tris_.empty() && world_boxes_.empty())
	{
		// Nothing placed (transient tracking loss or dead anchors):
		// keep serving the previous bake, retry in 2 s, not every frame.
		world_tris_ = std::move(prev_tris);
		world_boxes_ = std::move(prev_boxes);
		build_grid();
		next_locate_ = now + XrTime(2e9);
		return false;
	}
	next_locate_ = 0;
	located_world_ = world;
	build_grid();
	return true;
}

void room_mesh::bake_obstacles(XrSpace world, XrTime now)
{
	// World AABBs from anchor-local polygons/boxes. Planar walls bake to
	// ~zero-thickness boxes; the margin expansion at query time turns them
	// into keep-out slabs. Radial push-out needs no facing info, so rooms
	// of any winding work.
	size_t bake_ok = 0, bake_fail = 0;
	XrResult first_err = XR_SUCCESS;
	for (const auto & o: obstacles_)
	{
		XrSpaceLocation location{.type = XR_TYPE_SPACE_LOCATION, .next = nullptr};
		if (XrResult res = xrLocateSpace(o.space, world, now, &location); res != XR_SUCCESS)
		{
			if (bake_fail == 0)
				first_err = res;
			++bake_fail;
			continue;
		}
		if ((location.locationFlags & (XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) != (XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT))
		{
			++bake_fail;
			continue;
		}
		const auto & p = location.pose.position;
		const auto & q = location.pose.orientation;
		glm::mat4 m = glm::translate(glm::mat4(1.f), {p.x, p.y, p.z}) * glm::mat4_cast(glm::quat(q.w, q.x, q.y, q.z));
		glm::vec3 mn{std::numeric_limits<float>::max()}, mx{std::numeric_limits<float>::lowest()};
		for (const auto & v: o.local)
		{
			glm::vec3 w = glm::vec3(m * glm::vec4(v, 1.f));
			mn = glm::min(mn, w);
			mx = glm::max(mx, w);
		}
		if (mn.x <= mx.x)
		{
			world_boxes_.push_back({mn, mx});
			++bake_ok;
		}
		else
			++bake_fail;
	}
	if (bake_fail)
		spdlog::info("room_mesh: baked {} boxes ({} anchors unlocatable, first: {})", bake_ok, bake_fail, xr::to_string(first_err));
}

void room_mesh::build_grid()
{
	grid_.clear();
	if (world_tris_.empty())
		return;
	glm::vec3 mn = world_tris_.front().a, mx = mn;
	for (const auto & t: world_tris_)
		for (const glm::vec3 * v: {&t.a, &t.b, &t.c})
		{
			mn = glm::min(mn, *v);
			mx = glm::max(mx, *v);
		}
	grid_min_ = mn;
	glm::vec3 extent = mx - mn;
	grid_dim_ = glm::ivec3(
	        std::min(int(std::ceil(extent.x / grid_cell_)) + 1, 64),
	        std::min(int(std::ceil(extent.y / grid_cell_)) + 1, 64),
	        std::min(int(std::ceil(extent.z / grid_cell_)) + 1, 64));
	grid_dim_ = glm::max(grid_dim_, glm::ivec3(1));
	grid_.resize(size_t(grid_dim_.x) * size_t(grid_dim_.y) * size_t(grid_dim_.z));
	auto cell_of = [&](const glm::vec3 & v) {
		glm::ivec3 c = glm::ivec3((v - grid_min_) / grid_cell_);
		return glm::clamp(c, glm::ivec3(0), grid_dim_ - 1);
	};
	for (size_t i = 0; i < world_tris_.size(); ++i)
	{
		const auto & t = world_tris_[i];
		glm::vec3 tmn = glm::min(glm::min(t.a, t.b), t.c);
		glm::vec3 tmx = glm::max(glm::max(t.a, t.b), t.c);
		glm::ivec3 c0 = cell_of(tmn), c1 = cell_of(tmx);
		for (int x = c0.x; x <= c1.x; ++x)
			for (int y = c0.y; y <= c1.y; ++y)
				for (int z = c0.z; z <= c1.z; ++z)
					grid_[(size_t(x) * size_t(grid_dim_.y) + size_t(y)) * size_t(grid_dim_.z) + size_t(z)].push_back(uint32_t(i));
	}
}

// Closest point on triangle to p (Ericson 5.1.5), squared distance out.
static float closest_point_tri(const glm::vec3 & p, const glm::vec3 & a, const glm::vec3 & b, const glm::vec3 & c, glm::vec3 & closest_out)
{
	glm::vec3 ab = b - a, ac = c - a, ap = p - a;
	float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
	if (d1 <= 0.f && d2 <= 0.f)
	{
		closest_out = a;
		return glm::dot(ap, ap);
	}
	glm::vec3 bp = p - b;
	float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
	if (d3 >= 0.f && d4 <= d3)
	{
		closest_out = b;
		return glm::dot(bp, bp);
	}
	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
	{
		float v = d1 / (d1 - d3);
		closest_out = a + ab * v;
		return glm::dot(p - closest_out, p - closest_out);
	}
	glm::vec3 cp = p - c;
	float d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
	if (d6 >= 0.f && d5 <= d6)
	{
		closest_out = c;
		return glm::dot(cp, cp);
	}
	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
	{
		float w = d2 / (d2 - d6);
		closest_out = a + ac * w;
		return glm::dot(p - closest_out, p - closest_out);
	}
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
	{
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		closest_out = b + (c - b) * w;
		return glm::dot(p - closest_out, p - closest_out);
	}
	float denom = 1.f / (va + vb + vc);
	float v = vb * denom, w = vc * denom;
	closest_out = a + ab * v + ac * w;
	return glm::dot(p - closest_out, p - closest_out);
}

bool room_mesh::closest(const glm::vec3 & p, float max_dist, glm::vec3 & closest_out) const
{
	if (world_tris_.empty())
		return false;
	float best2 = max_dist * max_dist;
	bool found = false;
	auto test_tri = [&](uint32_t i) {
		glm::vec3 q;
		float d2 = closest_point_tri(p, world_tris_[i].a, world_tris_[i].b, world_tris_[i].c, q);
		if (d2 < best2)
		{
			best2 = d2;
			closest_out = q;
			found = true;
		}
	};
	if (not grid_.empty())
	{
		glm::vec3 lo = p - glm::vec3(max_dist), hi = p + glm::vec3(max_dist);
		glm::ivec3 c0 = glm::clamp(glm::ivec3((lo - grid_min_) / grid_cell_), glm::ivec3(0), grid_dim_ - 1);
		glm::ivec3 c1 = glm::clamp(glm::ivec3((hi - grid_min_) / grid_cell_), glm::ivec3(0), grid_dim_ - 1);
		for (int x = c0.x; x <= c1.x; ++x)
			for (int y = c0.y; y <= c1.y; ++y)
				for (int z = c0.z; z <= c1.z; ++z)
					for (uint32_t i: grid_[(size_t(x) * size_t(grid_dim_.y) + size_t(y)) * size_t(grid_dim_.z) + size_t(z)])
						test_tri(i);
	}
	else
	{
		for (uint32_t i = 0; i < uint32_t(world_tris_.size()); ++i)
			test_tri(i);
	}
	return found;
}

bool room_mesh::clear_of(const glm::vec3 & p, float margin) const
{
	return clearance(p, margin) >= margin;
}

bool room_mesh::push_out(glm::vec3 & p, float margin) const
{
	if (state_ != state::ready)
		return false;
	// Committed-but-unbaked room data fails closed: without baked
	// geometry there is nothing to push away from, so report unmoved and
	// let clear_of (which reads 0 clearance here) reject the sample.
	if (awaiting_bake())
		return false;
	// Mesh and boxes constrain together in one pass: a partial triangle
	// mesh must not silence the manual-wall boxes covering its holes, nor
	// vice versa.
	bool moved = false;
	if (not world_tris_.empty())
	{
		glm::vec3 q;
		if (closest(p, margin, q))
		{
			glm::vec3 push = p - q;
			float d = glm::length(push);
			if (d < 1e-6f)
				push = {0.f, 1.f, 0.f}; // exactly on the surface: lift up
			else
				push /= d;
			p = q + push * margin;
			moved = true;
		}
	}
	for (const auto & b: world_boxes_)
	{
		glm::vec3 e{margin, margin, margin};
		glm::vec3 q2 = glm::clamp(p, b.mn - e, b.mx + e);
		glm::vec3 d2 = p - q2;
		float l = glm::length(d2);
		if (l >= margin)
			continue;
		if (l > 1e-6f)
		{
			p = q2 + d2 / l * margin;
		}
		else
		{
			glm::vec3 pen = glm::min(p - (b.mn - e), (b.mx + e) - p);
			if (pen.x <= pen.y && pen.x <= pen.z)
				p.x = (p.x - b.mn.x < b.mx.x - p.x) ? b.mn.x - margin : b.mx.x + margin;
			else if (pen.y <= pen.z)
				p.y = (p.y - b.mn.y < b.mx.y - p.y) ? b.mn.y - margin : b.mx.y + margin;
			else
				p.z = (p.z - b.mn.z < b.mx.z - p.z) ? b.mn.z - margin : b.mx.z + margin;
		}
		moved = true;
	}
	return moved;
}

float room_mesh::clearance(const glm::vec3 & p, float max_dist) const
{
	if (state_ != state::ready)
		return max_dist; // no room data: never constrain on missing data
	if (awaiting_bake())
		return 0.f; // committed data, nothing baked: fail closed
	float best = max_dist;
	if (not world_tris_.empty())
	{
		glm::vec3 q;
		if (closest(p, best, q))
			best = glm::length(p - q);
	}
	for (const auto & b: world_boxes_)
	{
		glm::vec3 q2 = glm::clamp(p, b.mn, b.mx);
		float l = glm::length(p - q2);
		best = std::min(best, l);
	}
	return best;
}

bool room_mesh::can_capture()
{
#ifdef __ANDROID__
	return have_capture_ && have_query_ && xrRequestSceneCaptureFB && permission_granted();
#else
	return false;
#endif
}

void room_mesh::request_capture()
{
	if (not can_capture() || not xrRequestSceneCaptureFB)
	{
		spdlog::warn("room_mesh: scene capture unavailable (extension {} / permission {})",
		             have_capture_ ? "present" : "missing",
		             permission_granted() ? "granted" : "missing");
		return;
	}
	XrSceneCaptureRequestInfoFB info{
	        .type = XR_TYPE_SCENE_CAPTURE_REQUEST_INFO_FB,
	        .next = nullptr,
	        .requestByteCount = 0,
	        .request = nullptr,
	};
	XrAsyncRequestIdFB request = 0;
	XrResult res = xrRequestSceneCaptureFB(*sess_, &info, &request);
	if (res == XR_SUCCESS)
		spdlog::info("room_mesh: scene capture requested (system UX takes over)");
	else
		spdlog::warn("room_mesh: scene capture request failed ({})", xr::to_string(res));
}

std::string room_mesh::status() const
{
	switch (state_)
	{
	case state::unsupported:
		return "Room mesh unavailable on this device";
	case state::need_permission:
		return "Waiting for spatial permission…";
	case state::idle:
		return "Preparing room scan…";
	case state::querying:
		return "Scanning room…";
	case state::ready:
		if (not world_tris_.empty())
			return "Room mesh active (" + std::to_string(world_tris_.size()) + " triangles)";
		if (not world_boxes_.empty())
			return "Room planes active (" + std::to_string(world_boxes_.size()) + " walls/volumes)";
		return "Room data loaded, locating…";
	case state::no_data:
		return "No room set up here";
	case state::failed:
		return "Room scan failed: " + fail_reason_;
	}
	return "?";
}

} // namespace wivrn
