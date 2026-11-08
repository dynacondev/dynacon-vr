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
#include "encoder_settings.h"

#include "driver/configuration.h"
#include "driver/wivrn_session.h"
#include "util/u_logging.h"
#include "utils/wivrn_vk_bundle.h"
#include "video_encoder.h"
#include "wivrn_packets.h"

#include <magic_enum.hpp>
#include <span>
#include <string>
#include <vulkan/vulkan.hpp>

#include "wivrn_config.h"

#if WIVRN_USE_NVENC
#include "video_encoder_nvenc.h"
#endif
#if WIVRN_USE_VAAPI
#include "ffmpeg/video_encoder_va.h"
#include <libavutil/ffversion.h>
#endif
#if WIVRN_USE_VULKAN_ENCODE
#include "video_encoder_vulkan_h265.h"
#endif

namespace wivrn
{

static const double passthrough_bitrate_factor = 0.05;

// Depth stream (index 3, half-res SBS virtual depth) is pinned to 15% of
// the total bitrate: it is larger than before but low-entropy 10-bit
// luma, and must not starve color. Weights for 0..2 are area-based as
// before; depth weight is derived from their sum so the final share is
// exactly depth_share.
static const double depth_bitrate_share = 0.15;

static void split_bitrate(std::array<wivrn::encoder_settings, 4> & encoders, uint64_t bitrate)
{
	assert(bitrate > 0);
	double color_weight = 0;
	for (auto [i, encoder]: std::ranges::enumerate_view(encoders))
	{
		if (i == 3)
			continue;
		double w = encoder.width * encoder.height;
		if (i == 2)
			w *= passthrough_bitrate_factor;
		switch (encoder.codec)
		{
			case wivrn::h264:
				w *= 2;
				break;
			case wivrn::h265:
			case wivrn::av1:
			case wivrn::raw:
				break;
		}
		encoder.bitrate = w;
		color_weight += w;
	}
	encoders[3].bitrate = color_weight * depth_bitrate_share / (1.0 - depth_bitrate_share);
	double total_weight = color_weight + encoders[3].bitrate;

	for (auto & encoder: encoders)
	{
		encoder.bitrate_multiplier = encoder.bitrate / total_weight;
		encoder.bitrate = encoder.bitrate_multiplier * bitrate;
	}
}

void print_encoders(const std::array<wivrn::encoder_settings, 4> & encoders)
{
	std::stringstream str;
	str << "Encoder configuration:";
	for (auto & encoder: encoders)
	{
		str << "\n\t* " << encoder.encoder_name << " (" << magic_enum::enum_name(encoder.codec) << " " << encoder.bit_depth << "-bit)"
		    << (encoder.enabled ? "" : " (disabled)")
		    << "\n\t  size: " << encoder.width << "x" << encoder.height
		    << "\n\t  bitrate: " << int(encoder.bitrate / 100'000) / 10. << "Mbit/s";
	}
	U_LOG_I("%s", str.str().c_str());
}

static void check_video_size(std::string_view encoder_name, video_codec codec, uint16_t & width, uint16_t & height)
{
#if WIVRN_USE_NVENC
	if (encoder_name == encoder_nvenc)
	{
		auto max = video_encoder_nvenc::get_max_size(codec);
		width = std::min<uint16_t>(max[0], width);
		height = std::min<uint16_t>(max[1], height);
	}
#endif
}

namespace
{
class prober
{
	wivrn::vk_bundle & vk;
	const from_headset::headset_info_packet & info;
	const bool nvidia;

#if WIVRN_USE_VAAPI
	std::unordered_map<video_codec, bool> vaapi_support;

	bool check_vaapi(video_codec codec)
	{
		if (auto it = vaapi_support.find(codec); it != vaapi_support.end())
			return it->second;
		try
		{
			video_encoder_va test(
			        vk,
			        encoder_settings{
			                .width = 800,
			                .height = 800,
			                .codec = codec,
			                .fps = 60,
			                .bitrate = 50'000'000,
			                .bit_depth = 8,
			        },
			        0);
			vaapi_support[codec] = true;
			return true;
		}
		catch (std::exception & e)
		{
			vaapi_support[codec] = false;
			U_LOG_I("vaapi not supported for %s", std::string(magic_enum::enum_name(codec)).c_str());
			return false;
		}
	}
#endif

#if WIVRN_USE_NVENC
	std::unordered_map<video_codec, bool> nvenc_support;

	bool check_nvenc(video_codec codec)
	{
		if (auto it = nvenc_support.find(codec); it != nvenc_support.end())
			return it->second;
		try
		{
			video_encoder_nvenc test(
			        vk,
			        encoder_settings{
			                .width = 800,
			                .height = 800,
			                .codec = codec,
			                .fps = 60,
			                .bitrate = 50'000'000,
			                .bit_depth = 8,
			        },
			        0);
			nvenc_support[codec] = true;
			return true;
		}
		catch (std::exception & e)
		{
			nvenc_support[codec] = false;
			U_LOG_I("nvenc not supported for %s", std::string(magic_enum::enum_name(codec)).c_str());
			return false;
		}
	}
#endif

	static bool is_nvidia(vk::raii::PhysicalDevice & physical_device)
	{
		auto props = physical_device.getProperties();
		return props.vendorID == 0x10DE;
	}

#if WIVRN_USE_VULKAN_ENCODE

	bool has_vk(video_codec codec)
	{
		if (vk.encode_queues.empty())
			return false;
		if (not std::get<vk::PhysicalDeviceVideoMaintenance1FeaturesKHR>(vk.feat).videoMaintenance1)
		{
			U_LOG_I("Cannot use vulkan video encode without VideoMaintenance1 feature");
			return false;
		}
		auto prop = vk.physical_device.getQueueFamilyProperties2<vk::StructureChain<vk::QueueFamilyProperties2, vk::QueueFamilyVideoPropertiesKHR>>();
		const auto flags = prop.at(vk.encode_queues[0].family_index).get<vk::QueueFamilyVideoPropertiesKHR>().videoCodecOperations;
		switch (codec)
		{
			case h264: {
				auto res = vk.has_device_ext(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME) and flags & vk::VideoCodecOperationFlagBitsKHR::eEncodeH264;
				if (not res)
					U_LOG_I("GPU does not support H.264 Vulkan video encode");
				return res;
			}
			case h265: {
				auto res = vk.has_device_ext(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME) and flags & vk::VideoCodecOperationFlagBitsKHR::eEncodeH265;
				if (not res)
					U_LOG_I("GPU does not support H.265 Vulkan video encode");
				return res;
			}
			case av1:
				U_LOG_D("Vulkan video encode for AV1 is not implemented in WiVRn");
			case raw:
				return false;
		}
		U_LOG_E("Invalid codec %d", int(codec));
		return false;
	}
#endif

public:
	prober(wivrn::vk_bundle & vk, const from_headset::headset_info_packet & info) :
	        vk(vk), info(info), nvidia(is_nvidia(vk.physical_device)) {}

	std::pair<std::string, video_codec> select_encoder(const configuration::encoder & config)
	{
		if (config.codec == video_codec::raw or config.name == encoder_raw)
			return {encoder_raw, video_codec::raw};

		std::vector<video_codec> codecs;
		if (config.codec)
		{
			codecs.push_back(*config.codec);
		}
		else
		{
			codecs.reserve(info.supported_codecs.size());
			for (const auto & capability: info.supported_codecs)
				codecs.push_back(capability.codec);
		}

#if WIVRN_USE_VULKAN_ENCODE
		if (config.name.empty() or config.name == encoder_vulkan)
		{
			for (auto codec: codecs)
			{
				if (has_vk(codec))
					return {encoder_vulkan, codec};
			}
		}
#endif

#if WIVRN_USE_NVENC
		if ((nvidia and config.name.empty()) or config.name == encoder_nvenc)
		{
			for (auto codec: codecs)
			{
				if (check_nvenc(codec))
					return {encoder_nvenc, codec};
			}
		}
#endif

#if WIVRN_USE_VAAPI
		if (config.name.empty() or config.name == encoder_vaapi)
		{
			for (auto codec: codecs)
			{
				if (check_vaapi(codec))
					return {encoder_vaapi, codec};
			}
		}
#endif
		U_LOG_W("No suitable hardware accelerated codec found");
#if WIVRN_USE_X264
		if (config.name.empty() or config.name == encoder_x264)
			return {encoder_x264, video_codec::h264};
#endif

		throw std::runtime_error("Failed to find a suitable video encoder");
	}

	// 10-bit H.265 construction probe for the depth stream (index 3):
	// mirrors the 8-bit probes above but exercises the Main10 path, so
	// a backend advertising H.265 without 10-bit support disables depth
	// instead of throwing later at encoder construction.
	bool check_10bit_h265(const std::string & encoder_name)
	{
#if WIVRN_USE_NVENC
		if (encoder_name == encoder_nvenc)
		{
			try
			{
				video_encoder_nvenc test(
				        vk,
				        encoder_settings{
				                .width = 800,
				                .height = 800,
				                .codec = video_codec::h265,
				                .fps = 60,
				                .bitrate = 50'000'000,
				                .bit_depth = 10,
				        },
				        3);
				return true;
			}
			catch (std::exception & e)
			{
				U_LOG_I("nvenc 10-bit H.265 not supported for depth stream: %s", e.what());
				return false;
			}
		}
#endif
#if WIVRN_USE_VAAPI
		if (encoder_name == encoder_vaapi)
		{
			try
			{
				video_encoder_va test(
				        vk,
				        encoder_settings{
				                .width = 800,
				                .height = 800,
				                .codec = video_codec::h265,
				                .fps = 60,
				                .bitrate = 50'000'000,
				                .bit_depth = 10,
				        },
				        3);
				return true;
			}
			catch (std::exception & e)
			{
				U_LOG_I("vaapi 10-bit H.265 not supported for depth stream: %s", e.what());
				return false;
			}
		}
#endif
#if WIVRN_USE_VULKAN_ENCODE
		if (encoder_name == encoder_vulkan)
		{
			try
			{
				auto encoder = video_encoder_vulkan_h265::create(
				        vk,
				        encoder_settings{
				                .width = 800,
				                .height = 800,
				                .codec = video_codec::h265,
				                .fps = 60,
				                .bitrate = 50'000'000,
				                .bit_depth = 10,
				        },
				        3);
				return true;
			}
			catch (std::exception & e)
			{
				U_LOG_I("vulkan 10-bit H.265 not supported for depth stream: %s", e.what());
				return false;
			}
		}
#endif
		// x264, raw and unknown backends have no 10-bit H.265 path.
		return false;
	}
};
} // namespace

static uint16_t align(uint16_t value, uint16_t alignment)
{
	return ((value + alignment - 1) / alignment) * alignment;
}

std::array<encoder_settings, 4> get_encoder_settings(wivrn::vk_bundle & bundle, wivrn_session & session)
{
	configuration config;

	std::array<wivrn::encoder_settings, 4> res;
	const auto & info = session.get_info();
	const auto settings = *session.get_settings();

	prober prober{bundle, info};

	for (auto [i, src_dst]: std::ranges::enumerate_view(std::ranges::zip_view(config.encoders, res)))
	{
		auto & [src, dst] = src_dst;
		dst.fps = session.default_fps();
		dst.options = src.options;
		dst.device = src.device;

		std::tie(dst.encoder_name, dst.codec) = prober.select_encoder(src);

		// Depth stream (index 3) requires H.265 Main10: 10-bit luma
		// carrying 2mm/LSB virtual depth. The headset must advertise
		// 10-bit-capable H.265 and the backend must construct a Main10
		// encoder; otherwise the depth stream is disabled (the encoder
		// keeps benign fallback settings and is never fed) instead of
		// failing the session. There is no 8-bit fallback: the client
		// decodes depth as 10-bit unconditionally.
		if (i == 3)
		{
			// An explicit non-H.265 pin (including global raw test
			// mode) disables depth with a warning instead of failing:
			// depth is an optional enhancement, never worth the
			// session.
			dst.enabled = false;
			if (src.codec and *src.codec != wivrn::h265)
			{
				U_LOG_W("virtual depth stream disabled: pinned %s codec cannot carry 10-bit H.265 depth",
				        std::string(magic_enum::enum_name(*src.codec)).c_str());
				continue;
			}
			auto h265 = std::ranges::find(
			        info.supported_codecs, wivrn::h265, &video_codec_capability::codec);
			if (h265 != info.supported_codecs.end() and h265->supports_10bit)
			{
				try
				{
					configuration::encoder depth_cfg = src;
					depth_cfg.codec = wivrn::h265;
					std::tie(dst.encoder_name, dst.codec) = prober.select_encoder(depth_cfg);
					dst.enabled = dst.codec == wivrn::h265 and
					              prober.check_10bit_h265(dst.encoder_name);
				}
				catch (std::exception &)
				{
					// Leave depth disabled below.
				}
			}
			if (not dst.enabled)
				U_LOG_W("virtual depth stream disabled: H.265 Main10 unavailable (headset or GPU)");
		}
	}

	auto width = align(info.stream_eye_width, 64);
	auto height = align(info.stream_eye_height, 64);
	// Ensure we don't try to encode too large images (only for left/right, ignore alpha)
	for (size_t i = 0; i < 2; ++i)
		check_video_size(res[i].encoder_name, res[i].codec, width, height);

	for (auto [i, dst]: std::ranges::enumerate_view(res))
	{
		if (i == 3) // virtual depth SBS: half stream resolution per eye
		            // (SBS width = full stream width, height = half).
		            // width/height are 64-aligned above, so both stay
		            // %32 for NVENC.
		{
			dst.width = width;
			dst.height = height / 2;
			uint16_t w = dst.width, h = dst.height;
			check_video_size(dst.encoder_name, dst.codec, w, h);
			dst.width = w;
			dst.height = h;
			continue;
		}
		dst.width = width;
		dst.height = height;
		if (i == 2) // alpha channel
			dst.height /= 2;
	}

	auto bit_depth = config.bit_depth ? config.bit_depth : info.bit_depth;

	if (bit_depth and bit_depth != 8 and bit_depth != 10)
		throw std::runtime_error("invalid bit-depth setting. supported values: 8, 10");

	// Depth (index 3) is 10-bit only when enabled (see above): it must
	// not veto 10-bit color, and a disabled depth stream is excluded
	// from the 10-bit requirement entirely.
	const bool all_encoders_10bit = std::ranges::all_of(
	        std::span(res).first(3), [&](const encoder_settings & encoder) {
		        auto it = std::ranges::find(info.supported_codecs, encoder.codec, &video_codec_capability::codec);
		        return it != info.supported_codecs.end() and it->supports_10bit;
	        });

	if (bit_depth == 10 and not all_encoders_10bit)
	{
		U_LOG_I("Using 8-bit encoding due to mixed encoder support.");
		bit_depth = 8;
	}
	else if (not bit_depth)
	{
		bit_depth = all_encoders_10bit ? 10 : 8;
	}

	auto check_format = [&](vk::Format format) {
		try
		{
			auto props = bundle.physical_device.getImageFormatProperties(
			        format,
			        vk::ImageType::e2D,
			        vk::ImageTiling::eOptimal,
			        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc);
			return props.maxArrayLayers >= 3 and
			       props.maxExtent.depth >= 1 and
			       props.maxExtent.width >= width and
			       props.maxExtent.height >= height;
		}
		catch (vk::FormatNotSupportedError &)
		{
			return false;
		}
	};

#if WIVRN_USE_VAAPI
	auto check_vaapi = [&](int bit_depth) {
		for (const auto [i, encoder]: std::ranges::enumerate_view(res))
		{
			// Disabled depth is excluded: it keeps 8-bit fallback
			// settings (assigned below), probed nowhere. Enabled depth
			// is always 10-bit; color rides bit_depth.
			if (i == 3 and not res[3].enabled)
				continue;
			int depth = (i == 3) ? 10 : bit_depth;
			if (encoder.encoder_name == encoder_vaapi)
			{
				try
				{
					video_encoder_va{
					        bundle,
					        encoder_settings{
					                .width = 800,
					                .height = 800,
					                .codec = encoder.codec,
					                .fps = 60,
					                .bitrate = 50'000'000,
					                .bit_depth = depth,
					        },
					        0};
				}
				catch (std::exception & e)
				{
					U_LOG_I("vaapi not supported for %s %d bits", std::string(magic_enum::enum_name(encoder.codec)).c_str(), depth);
					return false;
				}
			}
		}
		return true;
	};
#else
	auto check_vaapi = [&](int) {
		return true;
	};
#endif

	// R16 storage images back 10-bit color and the depth staging image
	// alike; depth staging needs R16 unconditionally.
	bool have_r16 = check_format(vk::Format::eR16Unorm);
	if (bit_depth == 10 and not(have_r16 and check_vaapi(*bit_depth)))
	{
		U_LOG_W("GPU does not have sufficient support for 10-bit images, reverting to 8");
		bit_depth = 8;
	}
	if (bit_depth == 8 and not check_format(vk::Format::eR8Unorm))
	{
		U_LOG_W("GPU does not have sufficient support for 8-bit images");
	}
	if (res[3].enabled and not have_r16)
	{
		U_LOG_W("virtual depth stream disabled: GPU lacks R16 storage images");
		res[3].enabled = false;
	}

	for (auto & i: res)
		i.bit_depth = bit_depth.value_or(10);
	// Virtual depth is 10-bit 2mm/LSB luma when enabled; a disabled depth
	// stream keeps 8-bit fallback settings (always constructible) and is
	// never fed.
	res[3].bit_depth = res[3].enabled ? 10 : 8;

	split_bitrate(res, settings.bitrate_bps);
	if (not res[3].enabled)
	{
		// Reclaim the depth share for color instead of reserving 15%
		// for an encoder that never runs.
		res[3].bitrate = 0;
		res[3].bitrate_multiplier = 0;
		double color_sum = res[0].bitrate_multiplier + res[1].bitrate_multiplier +
		                   res[2].bitrate_multiplier;
		if (color_sum > 0)
		{
			for (int i = 0; i < 3; ++i)
			{
				res[i].bitrate_multiplier /= color_sum;
				res[i].bitrate = uint64_t(res[i].bitrate_multiplier * settings.bitrate_bps);
			}
		}
	}
	return res;
}
} // namespace wivrn
