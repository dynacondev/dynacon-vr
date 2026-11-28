/*
 * WiVRn VR streaming
 * Copyright (C) 2025  Patrick Nicolas <patricknicolas@laposte.net>
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

#include "wivrn_packets.h"

#include "wivrn_config.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vulkan/vulkan_raii.hpp>

namespace scenes
{
class stream;
}

namespace wivrn
{
class shard_accumulator;
class decoder
{
public:
	// YCbCr conversion definition for an image view created with
	// VkSamplerYcbcrConversionInfo in its pNext chain (hardware decoder
	// imports: Android AHB, v4l2 DMA-buf, raw multi-planar). Consumers
	// that sample such a view through a combined image sampler MUST use
	// immutable samplers baked with an identically defined conversion
	// (VUID-VkWriteDescriptorSet-descriptorType-02738); this struct is
	// the comparable definition they build from. Populated per blit
	// handle at map time and immutable afterwards, so the definition
	// always agrees with its view — even across buffer slot reuse and
	// format changes (each handle carries the definition its view was
	// mapped under). has_conversion=false means a plain view: any
	// compatible sampler works.
	struct ycbcr_definition
	{
		bool has_conversion = false;
		// External format for Android AHB imports (0 = explicit format
		// in `format`, raw/v4l2 style).
		uint64_t external_format = 0;
		vk::Format format = vk::Format::eUndefined;
		vk::SamplerYcbcrModelConversion ycbcr_model = vk::SamplerYcbcrModelConversion::eYcbcr709;
		vk::SamplerYcbcrRange ycbcr_range = vk::SamplerYcbcrRange::eItuFull;
		vk::ComponentMapping components{};
		vk::ChromaLocation x_chroma_offset = vk::ChromaLocation::eMidpoint;
		vk::ChromaLocation y_chroma_offset = vk::ChromaLocation::eMidpoint;
		vk::Filter chroma_filter = vk::Filter::eNearest;

		bool operator==(const ycbcr_definition &) const = default;
	};

	struct blit_handle
	{
		wivrn::from_headset::feedback feedback;
		wivrn::to_headset::video_stream_data_shard::view_info_t view_info;
		vk::ImageView image_view = nullptr;
		vk::Image image = nullptr;
		vk::Extent2D extent{};
		vk::ImageLayout & current_layout;
		vk::Semaphore semaphore = nullptr;
		uint64_t * semaphore_val = nullptr;
		// Generation sampler for image_view: the sampler of the format
		// generation this frame was mapped with, set once before the
		// handle is published and immutable after. Kept alive by the by the decoder-specific payload
		// (android: vk_data pins the generation that owns the sampler
		// + conversion). The render thread must use THIS, never a
		// decoder-owned sampler: decoders recreate their live sampler
		// on format renegotiation (decoder thread), so reading it
		// per-frame is both a data race and a use-after-free vector
		// (stale view + destroyed conversion reads as garbage and
		// segfaults Adreno in updateDescriptorSets).
		vk::Sampler sampler = nullptr;
#if WIVRN_USE_V4L2
		// DMA-buf images from V4L2 require queue-family ownership
		// transfers between Vulkan and external engine
		uint32_t foreign_queue_family = vk::QueueFamilyIgnored;
#endif
		// Conversion definition for image_view, set by the producing
		// decoder at map time (see above). Value-initialized here,
		// assigned once before the handle is published.
		ycbcr_definition ycbcr{};
	};

public:
	static std::shared_ptr<decoder> make(
	        vk::raii::Device &,
	        vk::raii::PhysicalDevice &,
	        uint32_t vk_queue_family_index,
	        const wivrn::to_headset::video_stream_description & description,
	        uint8_t stream_index,
	        std::weak_ptr<scenes::stream>,
	        shard_accumulator *);
	virtual ~decoder();
	virtual void push_data(std::span<std::span<const uint8_t>> data, uint64_t frame_index, bool partial) = 0;

	virtual void frame_completed(
	        const from_headset::feedback & feedback,
	        const to_headset::video_stream_data_shard::view_info_t & view_info) = 0;

	static const std::vector<wivrn::video_codec_capability> & supported_codecs();
	static bool supports_10bit(wivrn::video_codec codec);
};
} // namespace wivrn
