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

#pragma once

#include "openxr/openxr.h"
#include "utils/handle.h"
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace xr
{
class instance;
class session;

// Demo passthrough color effects (Quest, XR_FB_passthrough +
// XR_META_passthrough_color_lut). Session-scoped desired state, applied to
// the flat (reconstruction) layer via
// xrPassthroughLayerSetStyleFB. Only one color struct in the Style next
// chain at a time (runtime validation), so the GUI exposes a mode radio.
enum class passthrough_effect_mode
{
	off,
	bcs,
	mono_rgba,
	lut,
	interpolated_lut,
};

enum class passthrough_mono_preset
{
	gray,
	thermal,
	posterize,
};

enum class passthrough_lut_preset
{
	identity,
	sepia,
	invert,
	high_contrast,
	night_vision,
};

struct passthrough_effects
{
	bool enabled = false; // fullscreen demo passthrough (like depth overlay)
	float opacity = 1.0f; // XrPassthroughStyleFB::textureOpacityFactor [0,1]
	float edge_alpha = 0.0f; // edge overlay strength [0,1], 0 disables
	// Edge tint rgb (alpha comes from edge_alpha above)
	float edge_r = 1.0f;
	float edge_g = 1.0f;
	float edge_b = 1.0f;

	float brightness = 0.0f; // BCS: [-100,100] def 0
	float contrast = 1.0f; // BCS: [0,inf] def 1
	float saturation = 1.0f; // BCS: [0,inf] def 1

	passthrough_effect_mode mode = passthrough_effect_mode::bcs;
	passthrough_mono_preset mono_preset = passthrough_mono_preset::thermal;

	passthrough_lut_preset lut_preset = passthrough_lut_preset::sepia;
	passthrough_lut_preset lut_preset_b = passthrough_lut_preset::invert;
	float lut_weight = 1.0f; // Lut: blend original->LUT [0,1]
	float lut_mix = 0.5f; // Interpolated: blend LUT A->B [0,1]
};

class passthrough_layer_fb : public utils::handle<XrPassthroughLayerFB>
{
public:
	passthrough_layer_fb(instance &, session &, const XrPassthroughLayerCreateInfoFB &);
};

class passthrough_fb : public utils::handle<XrPassthroughFB>
{
	PFN_xrPassthroughStartFB xrPassthroughStartFB{};
	PFN_xrPassthroughPauseFB xrPassthroughPauseFB{};
	PFN_xrPassthroughLayerPauseFB xrPassthroughLayerPauseFB{};
	PFN_xrPassthroughLayerResumeFB xrPassthroughLayerResumeFB{};
	PFN_xrPassthroughLayerSetStyleFB xrPassthroughLayerSetStyleFB{};

#ifdef XR_META_PASSTHROUGH_COLOR_LUT_EXTENSION_NAME
	PFN_xrCreatePassthroughColorLutMETA xrCreatePassthroughColorLutMETA{};
	PFN_xrDestroyPassthroughColorLutMETA xrDestroyPassthroughColorLutMETA{};
#endif
	bool lut_supported = false;

	struct lut_entry
	{
		XrPassthroughColorLutMETA handle = XR_NULL_HANDLE;
		std::vector<uint8_t> data;
		uint32_t resolution = 0;
	};
	std::map<std::string, lut_entry> lut_cache;
	bool lut_warn_once = false;

	XrPassthroughColorLutMETA get_or_create_lut(passthrough_lut_preset preset, uint32_t resolution);
	// Build + apply XrPassthroughStyleFB chain for the flat layer.
	// LUT handles are cached per preset; missing LUT ext falls back to
	// unstyled with a one-shot warn.
	void apply_style(const passthrough_effects & fx);

	passthrough_layer_fb passthrough_layer;
	XrCompositionLayerPassthroughFB composition_layer;

public:
	passthrough_fb(instance &, session &);
	~passthrough_fb();
	// Move-only: moving assigns the LUT map without destroying the
	// target's handles, so never move-assign a live instance (the
	// variant only ever emplaces/destroys; no moves happen in practice).
	passthrough_fb(passthrough_fb &&) noexcept = default;
	passthrough_fb & operator=(passthrough_fb &&) noexcept = default;
	passthrough_fb(const passthrough_fb &) = delete;
	passthrough_fb & operator=(const passthrough_fb &) = delete;

	void start();
	void pause();

	// Demo color effects: apply desired style to the flat layer.
	// Safe to call when LUT ext is missing (LUT modes fall back to plain
	// style with a warn). Called by session on enable + on GUI change.
	void set_effects(const passthrough_effects & fx);
	bool color_lut_supported() const
	{
		return lut_supported;
	}

	XrCompositionLayerBaseHeader * layer()
	{
		return (XrCompositionLayerBaseHeader *)&composition_layer;
	}

	// Forget any chained structs from a previous frame's submit. Our
	// chains (alpha blend, settings) live in per-frame storage while this
	// layer lives in the session: a stale next pointer would otherwise
	// dangle after the frame ends.
	void reset_chain()
	{
		composition_layer.next = nullptr;
	}

};

class passthrough_htc : public utils::handle<XrPassthroughHTC>
{
	XrCompositionLayerPassthroughHTC composition_layer;

public:
	passthrough_htc(instance &, session &);
	XrCompositionLayerBaseHeader * layer()
	{
		return (XrCompositionLayerBaseHeader *)&composition_layer;
	}
};

class passthrough_alpha_blend
{
};

using passthrough = std::variant<std::monostate, passthrough_fb, passthrough_htc, passthrough_alpha_blend>;

} // namespace xr
