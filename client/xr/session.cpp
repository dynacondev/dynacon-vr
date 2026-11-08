/*
 * WiVRn VR streaming
 * Copyright (C) 2022  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2022  Patrick Nicolas <patricknicolas@laposte.net>
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

#include "session.h"

#include "color_space.h"
#include "details/enumerate.h"
#include "openxr/openxr.h"
#include "utils/contains.h"
#include "xr/instance.h"
#include "xr/system.h"
#include "xr/to_string.h"
#include <ranges>
#include <vulkan/vulkan.h>
#include <openxr/openxr_platform.h>

xr::session::session(xr::instance & inst, xr::system & sys, vk::raii::Instance & vk_inst, vk::raii::PhysicalDevice & pdev, vk::raii::Device & dev, thread_safe<vk::raii::Queue> & queue, int queue_family_index) :
        inst(&inst),
        queue(&queue)
{
	XrGraphicsBindingVulkan2KHR vulkan_binding{
	        .type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR,

	        .instance = *vk_inst,
	        .physicalDevice = *pdev,
	        .device = *dev,
	        .queueFamilyIndex = (uint32_t)queue_family_index,
	        .queueIndex = 0,
	};

	XrSessionCreateInfo session_info{
	        .type = XR_TYPE_SESSION_CREATE_INFO,
	        .next = &vulkan_binding,
	        .systemId = sys,
	};

	CHECK_XR(xrCreateSession(inst, &session_info, &id));

	if (inst.has_extension(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME))
	{
		xrGetDisplayRefreshRateFB = inst.get_proc<PFN_xrGetDisplayRefreshRateFB>("xrGetDisplayRefreshRateFB");
		xrEnumerateDisplayRefreshRatesFB = inst.get_proc<PFN_xrEnumerateDisplayRefreshRatesFB>("xrEnumerateDisplayRefreshRatesFB");
		xrRequestDisplayRefreshRateFB = inst.get_proc<PFN_xrRequestDisplayRefreshRateFB>("xrRequestDisplayRefreshRateFB");
	}

	if (inst.has_extension(XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME))
		xrPerfSettingsSetPerformanceLevelEXT = inst.get_proc<PFN_xrPerfSettingsSetPerformanceLevelEXT>("xrPerfSettingsSetPerformanceLevelEXT");

	if (inst.has_extension(XR_FB_COLOR_SPACE_EXTENSION_NAME))
	{
		xrEnumerateColorSpacesFB = inst.get_proc<PFN_xrEnumerateColorSpacesFB>("xrEnumerateColorSpacesFB");
		xrSetColorSpaceFB = inst.get_proc<PFN_xrSetColorSpaceFB>("xrSetColorSpaceFB");
	}

	if (inst.has_extension(XR_META_BOUNDARY_VISIBILITY_EXTENSION_NAME) and sys.boundary_visibility_properties().supportsBoundaryVisibility)
		xrRequestBoundaryVisibilityMETA = inst.get_proc<PFN_xrRequestBoundaryVisibilityMETA>("xrRequestBoundaryVisibilityMETA");
}

std::vector<XrReferenceSpaceType> xr::session::get_reference_spaces() const
{
	return details::enumerate<XrReferenceSpaceType>(xrEnumerateReferenceSpaces, id);
}

xr::space xr::session::create_reference_space(XrReferenceSpaceType ref, const XrPosef & pose)
{
	XrReferenceSpaceCreateInfo create_info{
	        .type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
	        .referenceSpaceType = ref,
	        .poseInReferenceSpace = pose,
	};

	XrSpace s;
	CHECK_XR(xrCreateReferenceSpace(id, &create_info, &s));

	return s;
}

xr::space xr::session::create_action_space(XrAction action, const XrPosef & pose)
{
	XrActionSpaceCreateInfo create_info{
	        .type = XR_TYPE_ACTION_SPACE_CREATE_INFO,
	        .action = action,
	        .subactionPath = XR_NULL_PATH,
	        .poseInActionSpace = pose,
	};

	XrSpace s;
	CHECK_XR(xrCreateActionSpace(id, &create_info, &s));

	return s;
}

xr::hand_tracker xr::session::create_hand_tracker(XrHandEXT hand, XrHandJointSetEXT hand_joint_set)
{
	return {
	        *inst,
	        *this,
	        XrHandTrackerCreateInfoEXT{
	                .type = XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT,
	                .next = nullptr,
	                .hand = hand,
	                .handJointSet = hand_joint_set,
	        }};
}

std::vector<vk::Format> xr::session::get_swapchain_formats() const
{
	std::vector<vk::Format> formats;
	for (auto i: details::enumerate<int64_t>(xrEnumerateSwapchainFormats, id))
		formats.push_back(static_cast<vk::Format>(i));
	return formats;
}

XrFrameState xr::session::wait_frame()
{
	XrFrameWaitInfo wait_info{
	        .type = XR_TYPE_FRAME_WAIT_INFO,
	};

	XrFrameState state{
	        .type = XR_TYPE_FRAME_STATE,
	};

	CHECK_XR(xrWaitFrame(id, &wait_info, &state));

	return state;
}

void xr::session::begin_frame()
{
	XrFrameBeginInfo begin_info{
	        .type = XR_TYPE_FRAME_BEGIN_INFO,
	};

	auto lock = queue->lock();
	CHECK_XR(xrBeginFrame(id, &begin_info));
}

void xr::session::end_frame(XrTime display_time, const std::vector<XrCompositionLayerBaseHeader *> & layers, XrEnvironmentBlendMode blend_mode)
{
	const XrLocalDimmingFrameEndInfoMETA local_dimming_info{
	        .type = XR_TYPE_LOCAL_DIMMING_FRAME_END_INFO_META,
	        .localDimmingMode = XR_LOCAL_DIMMING_MODE_ON_META,
	};

	const XrFrameEndInfo end_info{
	        .type = XR_TYPE_FRAME_END_INFO,
	        .next = inst->has_extension(XR_META_LOCAL_DIMMING_EXTENSION_NAME) ? &local_dimming_info : nullptr,
	        .displayTime = display_time,
	        .environmentBlendMode = blend_mode,
	        .layerCount = (uint32_t)layers.size(),
	        .layers = layers.data(),
	};

	{
		auto lock = queue->lock();
		CHECK_XR(xrEndFrame(id, &end_info));
	}

	// Gate suppression on layers actually submitted this frame, not just
	// the passthrough handle being active: the runtime refuses suppression
	// (SUPPRESSION_NOT_ALLOWED) and forces visible when the app fully
	// obscures reality (opaque video-only), so video-only must expect
	// visible. Projected counts too: camera imagery is showing.
	bool passthrough_visible = (blend_mode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND);
	for (auto * l: layers)
	{
		if (l->type == XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB ||
		    l->type == XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_HTC)
		{
			passthrough_visible = true;
			break;
		}
	}
	update_boundary_visibility(passthrough_visible);
}

void xr::session::begin_session(XrViewConfigurationType view_config)
{
	XrSessionBeginInfo begin_info{
	        .type = XR_TYPE_SESSION_BEGIN_INFO,
	        .primaryViewConfigurationType = view_config,
	};

	CHECK_XR(xrBeginSession(id, &begin_info));
}

void xr::session::end_session()
{
	CHECK_XR(xrEndSession(id));
}

XrViewStateFlags xr::session::locate_views(XrViewConfigurationType view_config_type,
                                           XrTime display_time,
                                           XrSpace space,
                                           std::vector<XrView> & views)
{
	XrViewLocateInfo view_locate_info{
	        .type = XR_TYPE_VIEW_LOCATE_INFO,
	        .viewConfigurationType = view_config_type,
	        .displayTime = display_time,
	        .space = space,
	};

	XrViewState view_state{
	        .type = XR_TYPE_VIEW_STATE,
	};

	details::enumerate(xrLocateViews, views, id, &view_locate_info, &view_state);

	return view_state.viewStateFlags;
}

std::pair<XrViewStateFlags, std::vector<XrView>> xr::session::locate_views(XrViewConfigurationType view_config_type,
                                                                           XrTime display_time,
                                                                           XrSpace space)
{
	std::vector<XrView> views(2, XrView{.type = XR_TYPE_VIEW});
	return {locate_views(view_config_type, display_time, space, views), std::move(views)};
}

std::string xr::session::get_current_interaction_profile(const std::string & path)
{
	XrInteractionProfileState state{
	        .type = XR_TYPE_INTERACTION_PROFILE_STATE,
	};

	CHECK_XR(xrGetCurrentInteractionProfile(id, inst->string_to_path(path), &state));

	return inst->path_to_string(state.interactionProfile);
}

void xr::session::attach_actionsets(const std::vector<XrActionSet> & actionsets)
{
	XrSessionActionSetsAttachInfo attach_info{
	        .type = XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO,
	        .countActionSets = (uint32_t)actionsets.size(),
	        .actionSets = actionsets.data(),
	};

	CHECK_XR(xrAttachSessionActionSets(id, &attach_info));
}

std::vector<std::string> xr::session::sources_for_action(XrAction action)
{
	XrBoundSourcesForActionEnumerateInfo action_info{
	        .type = XR_TYPE_BOUND_SOURCES_FOR_ACTION_ENUMERATE_INFO,
	        .action = action,
	};

	auto sources = xr::details::enumerate<XrPath>(xrEnumerateBoundSourcesForAction, id, &action_info);

	std::vector<std::string> sources_name;
	sources_name.reserve(sources.size());
	for (XrPath path: sources)
	{
		sources_name.push_back(inst->path_to_string(path));
	}

	return sources_name;
}

std::vector<std::string> xr::session::localized_sources_for_action(XrAction action, XrInputSourceLocalizedNameFlags components)
{
	XrBoundSourcesForActionEnumerateInfo action_info{
	        .type = XR_TYPE_BOUND_SOURCES_FOR_ACTION_ENUMERATE_INFO,
	        .action = action,
	};

	auto sources = xr::details::enumerate<XrPath>(xrEnumerateBoundSourcesForAction, id, &action_info);

	std::vector<std::string> sources_name;
	sources_name.reserve(sources.size());
	for (XrPath path: sources)
	{
		XrInputSourceLocalizedNameGetInfo name_info{
		        .type = XR_TYPE_INPUT_SOURCE_LOCALIZED_NAME_GET_INFO,
		        .sourcePath = path,
		        .whichComponents = components,
		};

		sources_name.push_back(details::enumerate<char>(xrGetInputSourceLocalizedName, id, &name_info));
	}

	return sources_name;
}

float xr::session::get_current_refresh_rate()
{
	assert(inst->has_extension(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME));

	float refresh_rate = 0;
	if (xrGetDisplayRefreshRateFB)
		CHECK_XR(xrGetDisplayRefreshRateFB(id, &refresh_rate));

	return refresh_rate;
}

std::vector<float> xr::session::get_refresh_rates()
{
	if (xrEnumerateDisplayRefreshRatesFB)
	{
		try
		{
			return xr::details::enumerate<float>(xrEnumerateDisplayRefreshRatesFB, id);
		}
		catch (...)
		{
			// Return no available refresh rate in case of error
		}
	}

	return {};
}

void xr::session::set_refresh_rate(float refresh_rate)
{
	if (xrRequestDisplayRefreshRateFB)
	{
		spdlog::info("set refresh rate to {}Hz", refresh_rate);
		if (auto res = xrRequestDisplayRefreshRateFB(id, refresh_rate); res != XR_SUCCESS)
			spdlog::warn("Refresh rate change failed: {}", xr::to_string(res));
	}
}

bool xr::session::color_space_supported() const
{
	return xrEnumerateColorSpacesFB and xrSetColorSpaceFB;
}

std::vector<XrColorSpaceFB> xr::session::get_color_spaces()
{
	if (not color_space_supported())
		return {};
	if (not color_spaces_queried)
	{
		color_spaces_queried = true;
		try
		{
			color_spaces_cache = xr::details::enumerate<XrColorSpaceFB>(xrEnumerateColorSpacesFB, id);
		}
		catch (...)
		{
			// Return no available color space in case of error
		}
	}
	return color_spaces_cache;
}

bool xr::session::set_color_space(XrColorSpaceFB color_space)
{
	if (not color_space_supported())
		return false;
	if (auto res = xrSetColorSpaceFB(id, color_space); res != XR_SUCCESS)
	{
		spdlog::warn("Color space change failed: {}", xr::to_string(res));
		return false;
	}
	active_color_space = color_space;
	spdlog::info("set color space to {}", color_space_name(color_space));
	return true;
}

void xr::session::sync_actions(std::span<XrActionSet> action_sets)
{
	std::vector<XrActiveActionSet> active_action_sets(action_sets.size());

	for (auto && [i, j]: std::views::zip(active_action_sets, action_sets))
	{
		i.actionSet = j;
		i.subactionPath = XR_NULL_PATH;
	}

	XrActionsSyncInfo sync_info{
	        .type = XR_TYPE_ACTIONS_SYNC_INFO,
	        .countActiveActionSets = (uint32_t)active_action_sets.size(),
	        .activeActionSets = active_action_sets.data(),
	};

	CHECK_XR(xrSyncActions(id, &sync_info));
}

void xr::session::sync_actions(XrActionSet action_set, XrPath subaction_path)
{
	XrActiveActionSet active_action_set{
	        .actionSet = action_set,
	        .subactionPath = subaction_path};

	XrActionsSyncInfo sync_info{
	        .type = XR_TYPE_ACTIONS_SYNC_INFO,
	        .countActiveActionSets = 1,
	        .activeActionSets = &active_action_set,
	};

	CHECK_XR(xrSyncActions(id, &sync_info));
}

void xr::session::sync_actions(XrActionSet action_set, const std::string & subaction_path)
{
	sync_actions(action_set, inst->string_to_path(subaction_path));
}

void xr::session::enable_passthrough(xr::system & system)
{
	// Already live: do NOT re-apply style here. enable_passthrough() runs
	// every frame from stream::render; re-applying (2 IPCs + logs) each
	// frame stalls the frame and was observed to precede a
	// CALL_ORDER_INVALID on swapchain release. Style is applied once on
	// creation below and on explicit GUI edits via
	// apply_passthrough_effects().
	if (not std::holds_alternative<std::monostate>(passthrough))
		return;

	if (system.passthrough_supported() == xr::passthrough_type::none)
		return;

	if (inst->has_extension(XR_FB_PASSTHROUGH_EXTENSION_NAME))
	{
		// Guard the emplace: a throwing ctor would leave the variant
		// valueless_by_exception, after which every frame throws
		// bad_variant_access in add_passthrough_layer, skips end_frame,
		// and all later acquire/release calls fail with
		// CALL_ORDER_INVALID. Fall back to no passthrough instead.
		// set_effects is inside the guard too: it only allocates on LUT
		// preset changes, but a throw here would poison the frame loop
		// the same way, so keep creation + first style atomic.
		try
		{
			passthrough.emplace<xr::passthrough_fb>(*inst, *this);
			if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
				fb->set_effects(effects_state);
		}
		catch (std::exception & e)
		{
			spdlog::warn("Passthrough creation failed, disabling: {}", e.what());
			passthrough.emplace<std::monostate>();
			return;
		}
	}
	else if (inst->has_extension(XR_HTC_PASSTHROUGH_EXTENSION_NAME))
	{
		passthrough.emplace<xr::passthrough_htc>(*inst, *this);
	}
	else if (utils::contains(system.environment_blend_modes(XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO), XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND))
	{
		passthrough.emplace<xr::passthrough_alpha_blend>();
	}
}

void xr::session::apply_passthrough_effects()
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		fb->set_effects(effects_state);
}

bool xr::session::passthrough_color_lut_supported() const
{
#ifdef XR_META_PASSTHROUGH_COLOR_LUT_EXTENSION_NAME
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		return fb->color_lut_supported();
	// No live fb yet (passthrough currently off): report extension
	// availability so the GUI can enable LUT mode before first enable.
	return inst && inst->has_extension(XR_META_PASSTHROUGH_COLOR_LUT_EXTENSION_NAME);
#else
	return false;
#endif
}

void xr::session::disable_passthrough()
{
	if (std::holds_alternative<std::monostate>(passthrough))
		return;
	passthrough.emplace<std::monostate>();
}

bool xr::session::projected_passthrough_supported()
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		return fb->projected_mesh_supported();
	return false;
}

void xr::session::set_projected_passthrough_mesh(
        std::span<const XrVector3f> vertices,
        std::span<const uint32_t> indices,
        XrSpace base_space,
        const XrPosef & pose,
        const XrVector3f & scale)
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		fb->set_projected_mesh(vertices, indices, base_space, pose, scale);
}

void xr::session::clear_projected_passthrough_mesh()
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		fb->clear_projected_mesh();
}

bool xr::session::has_projected_passthrough_mesh()
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		return fb->has_projected_mesh();
	return false;
}

void xr::session::update_projected_passthrough_transform(XrSpace base_space, XrTime time, const XrPosef & pose, const XrVector3f & scale)
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		fb->update_projected_transform(base_space, time, pose, scale);
}

XrCompositionLayerBaseHeader * xr::session::projected_passthrough_layer()
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
	{
		if (fb->has_projected_mesh() or fb->has_projected_instances())
			return fb->projected_layer_header();
	}
	return nullptr;
}

void xr::session::set_passthrough_boundary_enabled(bool enabled)
{
	passthrough_boundary_enabled = enabled;
	boundary_request_failed = false;
	last_suppress_attempt = {};
}

void xr::session::on_boundary_visibility_changed(XrBoundaryVisibilityMETA v)
{
	runtime_boundary_visibility = v;
}

// Suppression is only eligible while passthrough layers are actually
// submitted: the runtime refuses (SUPPRESSION_NOT_ALLOWED) and forces
// visible when opaque. A refusal backs off to a 2s retry instead of a
// per-frame hot loop; unsuppress requests are never throttled.
void xr::session::update_boundary_visibility(bool passthrough_visible)
{
	if (not xrRequestBoundaryVisibilityMETA)
		return;

	auto v = not passthrough_boundary_enabled and passthrough_visible
	                 ? XR_BOUNDARY_VISIBILITY_SUPPRESSED_META
	                 : XR_BOUNDARY_VISIBILITY_NOT_SUPPRESSED_META;
	if (v == runtime_boundary_visibility)
		return;

	if (v == XR_BOUNDARY_VISIBILITY_SUPPRESSED_META and boundary_request_failed)
	{
		auto now = std::chrono::steady_clock::now();
		if (now - last_suppress_attempt < std::chrono::seconds(2))
			return;
		last_suppress_attempt = now;
	}

	if (auto res = xrRequestBoundaryVisibilityMETA(id, v); res == XR_SUCCESS)
	{
		runtime_boundary_visibility = v;
		boundary_request_failed = false;
	}
	else if (res == XR_BOUNDARY_VISIBILITY_SUPPRESSION_NOT_ALLOWED_META)
	{
		// Expected while opaque / outside guardian: stay visible, retry
		// on backoff. Info, not warn: fighting it every frame is what
		// made the ring flicker.
		last_suppress_attempt = std::chrono::steady_clock::now();
		boundary_request_failed = true;
		spdlog::info("Boundary suppression not allowed (opaque or outside guardian), staying visible");
	}
	else if (not boundary_request_failed)
	{
		spdlog::warn("Boundary visibility change failed: {}", xr::to_string(res));
		boundary_request_failed = true;
	}
}

void xr::session::set_performance_level(XrPerfSettingsDomainEXT domain, XrPerfSettingsLevelEXT level)
{
	if (xrPerfSettingsSetPerformanceLevelEXT)
		xrPerfSettingsSetPerformanceLevelEXT(*this, domain, level);
}

void xr::session::set_projected_instance_mesh(const std::string & key,
                                              std::span<const XrVector3f> vertices,
                                              std::span<const uint32_t> indices,
                                              XrSpace base_space,
                                              const XrPosef & pose,
                                              const XrVector3f & scale)
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		fb->set_projected_instance_mesh(key, vertices, indices, base_space, pose, scale);
}

void xr::session::remove_projected_instance(const std::string & key)
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		fb->remove_projected_instance(key);
}

void xr::session::clear_projected_instances()
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		fb->clear_projected_instances();
}

bool xr::session::has_projected_instance(const std::string & key)
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		return fb->has_projected_instance(key);
	return false;
}

bool xr::session::has_projected_instances()
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		return fb->has_projected_instances();
	return false;
}

void xr::session::update_projected_instance_transform(const std::string & key,
                                                      XrSpace base_space,
                                                      XrTime time,
                                                      const XrPosef & pose,
                                                      const XrVector3f & scale)
{
	if (auto * fb = std::get_if<xr::passthrough_fb>(&passthrough))
		fb->update_projected_instance_transform(key, base_space, time, pose, scale);
}
