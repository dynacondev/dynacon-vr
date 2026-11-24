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

#include "render/union_merge.h"

#include "render/feather_mask.h"
#include "vk/shader.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <spdlog/spdlog.h>

namespace
{
// Mirrors union_merge_mv.frag.glsl Push: active input count (1..8)
// for the uniform tap early-out. 4B, far below the 128-byte Vulkan
// minimum (no maxPushConstantsSize check needed); FRAG-only range,
// and the vertex shader declares no push block, so the mask_mv
// offset trap (mismatched vert/frag blocks under a shared range) is
// structurally impossible.
struct push
{
	int32_t count = 0;
};
static_assert(sizeof(push) == 4);

// Common-layer geometry push: per-eye MVP pair, VERT-only. 128B is
// exactly the Vulkan-guaranteed minimum, so no device-limit throw is
// needed; the fragment declares no push block, so the shared-range
// offset trap cannot fire.
struct geom_push
{
	std::array<glm::mat4, 2> mvp;
};
static_assert(sizeof(geom_push) == 128);

// Expand compact foveation runs to a direct packed-index map per merge
// pixel, baking the merge->render scale. Layout: [x0(W)] [x1(W)]
// [y0(H)] [y1(H)] uint32s, W/H = merge extent. Runs describe packed
// texels: entry i holds n_i texels covering n_i * ratio_i source
// pixels (ratio_i = |n_ratio - i| + 1, middle run 1:1). Inverting gives
// render->packed; the merge scale then picks the packed texel per
// merge pixel (uniform floor, clamped). Empty runs fall back to a
// uniform map over the packed size (straight / pre-foveation data).
static void expand_foveation_axis(const std::vector<uint16_t> & runs, int render_n, int merge_n, int packed_n, uint32_t * out)
{
	if (render_n <= 0 or merge_n <= 0)
	{
		for (int u = 0; u < merge_n; ++u)
			out[u] = 0;
		return;
	}
	if (runs.empty())
	{
		for (int u = 0; u < merge_n; ++u)
			out[u] = packed_n > 0 ? (uint32_t)std::min<int64_t>((int64_t)u * packed_n / merge_n, packed_n - 1) : 0;
		return;
	}
	// Direct map: render pixel -> packed texel.
	std::vector<uint32_t> r2p;
	r2p.reserve((size_t)render_n);
	const int n_ratio = (int)(runs.size() - 1) / 2;
	uint32_t packed = 0;
	for (size_t i = 0; i < runs.size() and (int)r2p.size() < render_n; ++i)
	{
		int ratio = std::abs(n_ratio - (int)i) + 1;
		for (uint16_t k = 0; k < runs[i] and (int)r2p.size() < render_n; ++k)
		{
			for (int r = 0; r < ratio and (int)r2p.size() < render_n; ++r)
				r2p.push_back(packed + k);
		}
		packed += runs[i];
	}
	while ((int)r2p.size() < render_n)
		r2p.push_back(r2p.empty() ? 0 : r2p.back());
	const uint32_t packed_max = packed_n > 0 ? (uint32_t)packed_n - 1 : 0;
	for (int u = 0; u < merge_n; ++u)
	{
		int64_t r = (int64_t)u * render_n / merge_n;
		uint32_t v = r2p[(size_t)std::min<int64_t>(r, render_n - 1)];
		// Clamp to the packed range: the runs should tile exactly it,
		// but an overrun here would read the other eye's half (eye 0)
		// or ClampToEdge texels (eye 1) — silent wrong depth.
		out[u] = packed_n > 0 ? std::min(v, packed_max) : 0;
	}
}

uint32_t find_memory_type(vk::raii::PhysicalDevice & physical_device, uint32_t type_bits, vk::MemoryPropertyFlags properties)
{
	auto mem = physical_device.getMemoryProperties();
	for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
	{
		if ((type_bits & (1u << i)) and (mem.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	throw std::runtime_error("union_merge: no suitable memory type");
}

// Common depth layer (R16F linear meters, MIN-blended) and binary
// verdict (R8) formats. R16F needs COLOR_ATTACHMENT + BLEND on this
// device — checked loudly in init_pipelines (no silent fallback: a
// missing MIN blend would under/over-reveal silently).
constexpr vk::Format layer_format = vk::Format::eR16Sfloat;
constexpr float layer_far = 65504.f;
constexpr vk::Format verdict_format = vk::Format::eR8Unorm;
} // namespace

union_merge_renderer::union_merge_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device) :
        device(&device),
        physical_device(&physical_device)
{
}

std::shared_ptr<vk::raii::PipelineCache> union_merge_renderer::pipeline_cache(vk::raii::Device & device)
{
	// Same process-wide object shared with the feather mask renderer
	// (one entry per VkDevice).
	return feather_mask_renderer::pipeline_cache(device);
}

void union_merge_renderer::warm_pipelines(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device, vk::Format format)
{
	// Throwaway renderer: its pipeline compile populates the shared
	// cache above, so the first real merge pays ~nothing.
	union_merge_renderer warm(device, physical_device);
	warm.init_pipelines(format);
}

bool union_merge_renderer::ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h)
{
	if (ready() and format_ == format and extent_.width == (uint32_t)w and extent_.height == (uint32_t)h)
		return false;
	// Rare path (first merge, video resize): nothing outstanding (we
	// acquire + release within one frame, always paired below).
	device->waitIdle();
	// Free the descriptor sets while their pool is still alive:
	// init_pipelines below recreates the pool, which implicitly frees
	// (destroys the use of) any sets still held.
	descriptor_sets.clear();
	verdict_descriptor_sets.clear();
	apply_descriptor_sets.clear();
	// format_/extent_ commit AFTER the fallible rebuild below: if the
	// swapchain or pipeline construction throws, the old values stay
	// and the next active frame retries instead of the early-out
	// above suppressing recovery forever against stale targets.
	vk::Format new_format = format;
	vk::Extent2D new_extent{(uint32_t)w, (uint32_t)h};
	swapchain_ = xr::swapchain(inst, sess, *device, new_format, w, h, 1, 2);
	spdlog::info("Union merge swapchain: {}x{} ({} images)", w, h, swapchain_.image_count());
	init_pipelines(new_format);
	// One multiview target per swapchain image: array color view +
	// layers=1 framebuffer (the 2-layer routing comes from the
	// multiview view mask, not the framebuffer); one draw broadcasts
	// to both eyes.
	targets.clear();
	for (vk::Image image: swapchain_.images())
	{
		targets.emplace_back();
		vk::ImageViewCreateInfo iv_info{
		        .image = image,
		        .viewType = vk::ImageViewType::e2DArray,
		        .format = format,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .baseMipLevel = 0,
		                .levelCount = 1,
		                .baseArrayLayer = 0,
		                .layerCount = 2,
		        },
		};
		targets.back().view = vk::raii::ImageView(*device, iv_info);
		vk::ImageView color_view = *targets.back().view;
		vk::FramebufferCreateInfo fb_info{
		        .renderPass = *renderpass,
		        .attachmentCount = 1,
		        .pAttachments = &color_view,
		        .width = new_extent.width,
		        .height = new_extent.height,
		        .layers = 1,
		};
		targets.back().framebuffer = vk::raii::Framebuffer(*device, fb_info);
	}
	// Common depth layer + merged + verdict targets (owned images, not
	// swapchain): all full merge res since the foveated verdict compares
	// per-pixel. Rebuilt under the same waitIdle above; the mesh buffers
	// survive (extent-independent).
	auto make_owned = [&](vk::Format fmt, vk::Extent2D e, vk::ImageUsageFlags usage, vk::raii::RenderPass & rp) {
		layer_target t;
		vk::ImageCreateInfo image_info{
		        .imageType = vk::ImageType::e2D,
		        .format = fmt,
		        .extent = {e.width, e.height, 1},
		        .mipLevels = 1,
		        .arrayLayers = 2,
		        .samples = vk::SampleCountFlagBits::e1,
		        .tiling = vk::ImageTiling::eOptimal,
		        .usage = usage,
		        .sharingMode = vk::SharingMode::eExclusive,
		        .initialLayout = vk::ImageLayout::eUndefined,
		};
		t.image = vk::raii::Image(*device, image_info);
		auto requirements = t.image.getMemoryRequirements();
		vk::MemoryAllocateInfo alloc_info{
		        .allocationSize = requirements.size,
		        .memoryTypeIndex = find_memory_type(*physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
		};
		t.memory = vk::raii::DeviceMemory(*device, alloc_info);
		t.image.bindMemory(*t.memory, 0);
		vk::ImageViewCreateInfo iv_info{
		        .image = *t.image,
		        .viewType = vk::ImageViewType::e2DArray,
		        .format = fmt,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .baseMipLevel = 0,
		                .levelCount = 1,
		                .baseArrayLayer = 0,
		                .layerCount = 2,
		        },
		};
		t.view_array = vk::raii::ImageView(*device, iv_info);
		vk::ImageView view = *t.view_array;
		vk::FramebufferCreateInfo fb_info{
		        .renderPass = *rp,
		        .attachmentCount = 1,
		        .pAttachments = &view,
		        .width = e.width,
		        .height = e.height,
		        .layers = 1,
		};
		t.framebuffer = vk::raii::Framebuffer(*device, fb_info);
		return t;
	};
	layer_ = make_owned(layer_format, new_extent, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, geom_renderpass);
	merged_ = make_owned(new_format, new_extent, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, renderpass);
	verdict_ = make_owned(verdict_format, new_extent, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, verdict_renderpass);
	layer_extent_ = new_extent;
	// Foveation index maps (SSBO pair, host-visible): per-eye packed
	// index per merge pixel, [x0(W)] [x1(W)] [y0(H)] [y1(H)] uint32s.
	verdict_tables_bytes = vk::DeviceSize(2 * (new_extent.width + new_extent.height)) * sizeof(uint32_t);
	for (auto & t: verdict_tables)
	{
		t = buffer_allocation(
		        *device,
		        vk::BufferCreateInfo{
		                .size = verdict_tables_bytes,
		                .usage = vk::BufferUsageFlagBits::eStorageBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "union merge verdict tables");
	}
	spdlog::info("Union merge layer targets: layer {}x{} merged {}x{} verdict {}x{} tables {}B",
	             new_extent.width, new_extent.height, new_extent.width, new_extent.height,
	             new_extent.width, new_extent.height, verdict_tables_bytes);
	// Commit only after the whole rebuild succeeded (see above).
	format_ = new_format;
	extent_ = new_extent;
	return true;
}

int union_merge_renderer::acquire()
{
	int idx = swapchain_.acquire();
	if (not swapchain_.wait(100'000'000))
	{
		// Never park forever: release the untouched acquisition to keep
		// pairing and skip the frame (mirrors the mask groups).
		swapchain_.release();
		return -1;
	}
	return idx;
}

void union_merge_renderer::release()
{
	swapchain_.release();
}

void union_merge_renderer::init_pipelines(vk::Format format)
{
	pipe_cache = pipeline_cache(*device);
	// 8 static sampler2DArray bindings, fragment-only (no vertex
	// input, 4B FRAG-only count push — the reduction is
	// count-independent apart from the early-out).
	vk::DescriptorSetLayoutBinding bindings[max_inputs];
	for (uint32_t i = 0; i < max_inputs; ++i)
	{
		bindings[i] = vk::DescriptorSetLayoutBinding{
		        .binding = i,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 1,
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		};
	}
	descriptor_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                 .bindingCount = max_inputs,
	                                                                 .pBindings = bindings,
	                                                         });
	// Linear + ClampToEdge (feather sampler precedent): the bilinear
	// tap IS the upscale for half/quarter-res inputs. maxLod 0, no
	// mipmap chain on swapchain images. NEVER nearest here: it would
	// alias the feather band.
	vk::SamplerCreateInfo sampler_info{
	        .magFilter = vk::Filter::eLinear,
	        .minFilter = vk::Filter::eLinear,
	        .mipmapMode = vk::SamplerMipmapMode::eLinear,
	        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
	        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
	        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
	        .mipLodBias = 0,
	        .maxAnisotropy = 1,
	        .compareEnable = VK_FALSE,
	        .minLod = 0,
	        .maxLod = 0,
	};
	sampler = vk::raii::Sampler(*device, sampler_info);
	// Two sets (ping-pong by record parity, see record()): one may
	// still be read by a prior in-flight command buffer while the
	// next frame rewrites the other.
	vk::DescriptorPoolSize pool_size{
	        .type = vk::DescriptorType::eCombinedImageSampler,
	        .descriptorCount = 2 * max_inputs,
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	        .maxSets = 2,
	        .poolSizeCount = 1,
	        .pPoolSizes = &pool_size,
	};
	descriptor_pool = vk::raii::DescriptorPool(*device, pool_info);
	for (int i = 0; i < 2; ++i)
	{
		vk::DescriptorSetAllocateInfo set_alloc{
		        .descriptorPool = *descriptor_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*descriptor_layout,
		};
		descriptor_sets.emplace_back(std::move(device->allocateDescriptorSets(set_alloc)[0]));
	}

	auto vert = load_shader(*device, "blur.vert");
	auto frag = load_shader(*device, "union_merge_mv.frag");
	vk::PipelineShaderStageCreateInfo stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **frag,
	                .pName = "main",
	        },
	};
	// No vertex input: the fullscreen triangle comes from
	// gl_VertexIndex (blur.vert precedent).
	vk::PipelineVertexInputStateCreateInfo vertex_input{};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly{
	        .topology = vk::PrimitiveTopology::eTriangleList,
	};
	vk::PipelineViewportStateCreateInfo viewport_state{
	        .viewportCount = 1,
	        .scissorCount = 1,
	};
	vk::PipelineRasterizationStateCreateInfo rasterization{
	        .polygonMode = vk::PolygonMode::eFill,
	        .cullMode = vk::CullModeFlagBits::eNone,
	        .frontFace = vk::FrontFace::eCounterClockwise,
	        .lineWidth = 1,
	};
	vk::PipelineMultisampleStateCreateInfo multisample{
	        .rasterizationSamples = vk::SampleCountFlagBits::e1,
	};
	vk::PipelineDepthStencilStateCreateInfo depth_stencil{
	        .depthTestEnable = VK_FALSE,
	        .depthWriteEnable = VK_FALSE,
	};
	// No blending: a single MAX write per pixel; the submit-time
	// REPLACE builds the dst-alpha union (like every mask target).
	vk::PipelineColorBlendAttachmentState blend_attachment{
	        .blendEnable = VK_FALSE,
	        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
	};
	vk::PipelineColorBlendStateCreateInfo blend{
	        .attachmentCount = 1,
	        .pAttachments = &blend_attachment,
	};
	vk::DynamicState dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic{
	        .dynamicStateCount = 2,
	        .pDynamicStates = dynamics,
	};
	// Descriptor set only, plus a 4B FRAG-only count range (the
	// vertex shader declares no push block — a VERT|FRAG shared range
	// with mismatched vert/frag blocks would re-trigger the mask_mv
	// offset trap, so the range stays FRAG-only by design).
	vk::PushConstantRange push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = sizeof(push),
	};
	vk::PipelineLayoutCreateInfo layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*descriptor_layout,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &push_range,
	};
	pipeline_layout = vk::raii::PipelineLayout(*device, layout_info);

	// Single-attachment multiview renderpass (DONT_CARE/STORE/OPTIMAL:
	// submitted image, keeps Adreno UBWC live like the feather groups).
	// DONT_CARE is exact: the fullscreen triangle overwrites every
	// pixel, so a CLEAR would only be a wasted fill.
	vk::AttachmentDescription attachment{
	        .format = format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eDontCare,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference color_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &color_ref,
	};
	uint32_t mv_view_mask = 0b11, mv_corr_mask = 0b11;
	vk::RenderPassMultiviewCreateInfo mv_rp_info{
	        .subpassCount = 1,
	        .pViewMasks = &mv_view_mask,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &mv_corr_mask,
	};
	vk::RenderPassCreateInfo renderpass_info{
	        .pNext = &mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &attachment,
	        .subpassCount = 1,
	        .pSubpasses = &subpass,
	};
	renderpass = vk::raii::RenderPass(*device, renderpass_info);

	vk::GraphicsPipelineCreateInfo pipeline_info{
	        .stageCount = 2,
	        .pStages = stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pDepthStencilState = &depth_stencil,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *pipeline_layout,
	        .renderPass = *renderpass,
	};
	pipeline = vk::raii::Pipeline(*device, *pipe_cache, pipeline_info);

	// Common depth layer format gate (see above): R16F must support
	// COLOR_ATTACHMENT + BLEND on this device. No silent fallback.
	{
		auto props = physical_device->getFormatProperties(layer_format);
		bool ok_color = (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachment) != vk::FormatFeatureFlags{};
		bool ok_blend = (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachmentBlend) != vk::FormatFeatureFlags{};
		if (not ok_color or not ok_blend)
			throw std::runtime_error("union_merge: R16F lacks COLOR_ATTACHMENT+BLEND (common depth layer unsupported on this device)");
	}

	// Geometry raster: position-only vertex input, MIN blend over the
	// FAR clear (near-wins, order-independent), CULL_NONE like the
	// silhouette raster (windows reveal from both sides). VERT-only
	// MVP-pair push range; the fragment declares no push block.
	auto geom_vert = load_shader(*device, "geom_depth_mv.vert");
	auto geom_frag = load_shader(*device, "geom_depth_mv.frag");
	vk::PipelineShaderStageCreateInfo geom_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **geom_vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **geom_frag,
	                .pName = "main",
	        },
	};
	vk::VertexInputBindingDescription geom_binding{
	        .binding = 0,
	        .stride = sizeof(XrVector3f),
	        .inputRate = vk::VertexInputRate::eVertex,
	};
	vk::VertexInputAttributeDescription geom_attribute{
	        .location = 0,
	        .binding = 0,
	        .format = vk::Format::eR32G32B32Sfloat,
	        .offset = 0,
	};
	vk::PipelineVertexInputStateCreateInfo geom_vertex_input{
	        .vertexBindingDescriptionCount = 1,
	        .pVertexBindingDescriptions = &geom_binding,
	        .vertexAttributeDescriptionCount = 1,
	        .pVertexAttributeDescriptions = &geom_attribute,
	};
	vk::PipelineRasterizationStateCreateInfo geom_rasterization{
	        .polygonMode = vk::PolygonMode::eFill,
	        .cullMode = vk::CullModeFlagBits::eNone,
	        .frontFace = vk::FrontFace::eCounterClockwise,
	        .lineWidth = 1,
	};
	vk::PipelineColorBlendAttachmentState geom_blend_attachment{
	        .blendEnable = VK_TRUE,
	        .srcColorBlendFactor = vk::BlendFactor::eOne,
	        .dstColorBlendFactor = vk::BlendFactor::eOne,
	        .colorBlendOp = vk::BlendOp::eMin,
	        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
	        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
	        .alphaBlendOp = vk::BlendOp::eMin,
	        .colorWriteMask = vk::ColorComponentFlagBits::eR,
	};
	vk::PipelineColorBlendStateCreateInfo geom_blend{
	        .attachmentCount = 1,
	        .pAttachments = &geom_blend_attachment,
	};
	vk::PushConstantRange geom_push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eVertex,
	        .offset = 0,
	        .size = sizeof(geom_push),
	};
	vk::PipelineLayoutCreateInfo geom_layout_info{
	        .setLayoutCount = 0,
	        .pSetLayouts = nullptr,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &geom_push_range,
	};
	geom_layout = vk::raii::PipelineLayout(*device, geom_layout_info);
	vk::AttachmentDescription geom_attachment{
	        .format = layer_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference geom_color_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription geom_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &geom_color_ref,
	};
	uint32_t geom_view_mask = 0b11, geom_corr_mask = 0b11;
	vk::RenderPassMultiviewCreateInfo geom_mv_rp_info{
	        .subpassCount = 1,
	        .pViewMasks = &geom_view_mask,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &geom_corr_mask,
	};
	vk::RenderPassCreateInfo geom_renderpass_info{
	        .pNext = &geom_mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &geom_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &geom_subpass,
	};
	geom_renderpass = vk::raii::RenderPass(*device, geom_renderpass_info);
	vk::GraphicsPipelineCreateInfo geom_pipeline_info{
	        .stageCount = 2,
	        .pStages = geom_stages,
	        .pVertexInputState = &geom_vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &geom_rasterization,
	        .pMultisampleState = &multisample,
	        .pDepthStencilState = &depth_stencil,
	        .pColorBlendState = &geom_blend,
	        .pDynamicState = &dynamic,
	        .layout = *geom_layout,
	        .renderPass = *geom_renderpass,
	};
	geom_pipeline = vk::raii::Pipeline(*device, *pipe_cache, geom_pipeline_info);

	// Verdict renderpass (R8 binary, DONT_CARE/STORE/OPTIMAL,
	// multiview): the fullscreen draw overwrites every pixel, so no
	// CLEAR is needed. Runs only with paired depth; mask-only frames
	// skip the pass entirely instead of clearing to 1.
	vk::AttachmentDescription verdict_attachment{
	        .format = verdict_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eDontCare,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference verdict_color_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription verdict_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &verdict_color_ref,
	};
	uint32_t verdict_view_mask = 0b11, verdict_corr_mask = 0b11;
	vk::RenderPassMultiviewCreateInfo verdict_mv_rp_info{
	        .subpassCount = 1,
	        .pViewMasks = &verdict_view_mask,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &verdict_corr_mask,
	};
	vk::RenderPassCreateInfo verdict_renderpass_info{
	        .pNext = &verdict_mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &verdict_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &verdict_subpass,
	};
	verdict_renderpass = vk::raii::RenderPass(*device, verdict_renderpass_info);

	// Verdict + apply sampler/descriptor state (static 2-slot CIS
	// layouts for layer/virt, FRAG-only, same Adreno-safe pattern as
	// the merge): verdict reads layer NEAREST + SBS virt (decoder
	// sampler); apply reads merged NEAREST + verdict NEAREST (both
	// plain texture() taps at full res — no upscale, filter matters).
	vk::SamplerCreateInfo nearest_info{
	        .magFilter = vk::Filter::eNearest,
	        .minFilter = vk::Filter::eNearest,
	        .mipmapMode = vk::SamplerMipmapMode::eNearest,
	        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
	        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
	        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
	        .mipLodBias = 0,
	        .maxAnisotropy = 1,
	        .compareEnable = VK_FALSE,
	        .minLod = 0,
	        .maxLod = 0,
	};
	sampler_nearest = vk::raii::Sampler(*device, nearest_info);
	auto make_dual_cis = [&](vk::raii::DescriptorSetLayout & layout, vk::raii::DescriptorPool & pool,
	                         std::vector<vk::raii::DescriptorSet> & sets) {
		vk::DescriptorSetLayoutBinding bindings[2];
		for (uint32_t i = 0; i < 2; ++i)
		{
			bindings[i] = vk::DescriptorSetLayoutBinding{
			        .binding = i,
			        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			        .descriptorCount = 1,
			        .stageFlags = vk::ShaderStageFlagBits::eFragment,
			};
		}
		layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
		                                                              .bindingCount = 2,
		                                                              .pBindings = bindings,
		                                                      });
		vk::DescriptorPoolSize pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 4,
		};
		vk::DescriptorPoolCreateInfo pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 2,
		        .poolSizeCount = 1,
		        .pPoolSizes = &pool_size,
		};
		pool = vk::raii::DescriptorPool(*device, pool_info);
		for (int i = 0; i < 2; ++i)
		{
			vk::DescriptorSetAllocateInfo set_alloc{
			        .descriptorPool = *pool,
			        .descriptorSetCount = 1,
			        .pSetLayouts = &*layout,
			};
			sets.emplace_back(std::move(device->allocateDescriptorSets(set_alloc)[0]));
		}
	};
	// Verdict pool (sets/layout/pipeline built by create_verdict_state
	// below so rebuilds share one path; the pool itself is
	// format-agnostic and never changes). Sized for the 2 image
	// bindings + the foveation-table storage buffer on each set.
	{
		vk::DescriptorPoolSize pool_sizes[2] = {
		        {
		                .type = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 4,
		        },
		        {
		                .type = vk::DescriptorType::eStorageBuffer,
		                .descriptorCount = 2,
		        },
		};
		vk::DescriptorPoolCreateInfo pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 2,
		        .poolSizeCount = 2,
		        .pPoolSizes = pool_sizes,
		};
		verdict_descriptor_pool = vk::raii::DescriptorPool(*device, pool_info);
	}
	make_dual_cis(apply_descriptor_layout, apply_descriptor_pool, apply_descriptor_sets);

	// Initial plain verdict state (also resets verdict_ycbcr_built after
	// a swapchain rebuild, so a stale YCbCr build can never persist).
	create_verdict_state(wivrn::decoder::ycbcr_definition{}, vk::Sampler{});

	auto apply_frag = load_shader(*device, "vr_apply_direct_mv.frag");
	vk::PipelineShaderStageCreateInfo apply_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **apply_frag,
	                .pName = "main",
	        },
	};
	vk::PipelineLayoutCreateInfo apply_layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*apply_descriptor_layout,
	        .pushConstantRangeCount = 0,
	        .pPushConstantRanges = nullptr,
	};
	apply_layout = vk::raii::PipelineLayout(*device, apply_layout_info);
	vk::GraphicsPipelineCreateInfo apply_pipeline_info{
	        .stageCount = 2,
	        .pStages = apply_stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pDepthStencilState = &depth_stencil,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *apply_layout,
	        .renderPass = *renderpass,
	};
	apply_pipeline = vk::raii::Pipeline(*device, *pipe_cache, apply_pipeline_info);

	// Mask-only apply: same vertex stage, renderpass and layout (the
	// fragment shader samples binding 0 only; binding 1 is never
	// written or read on this path).
	auto apply_mask_frag = load_shader(*device, "vr_apply_mask_mv.frag");
	vk::PipelineShaderStageCreateInfo apply_mask_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **apply_mask_frag,
	                .pName = "main",
	        },
	};
	vk::GraphicsPipelineCreateInfo apply_mask_pipeline_info{
	        .stageCount = 2,
	        .pStages = apply_mask_stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pDepthStencilState = &depth_stencil,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *apply_layout,
	        .renderPass = *renderpass,
	};
	apply_mask_pipeline = vk::raii::Pipeline(*device, *pipe_cache, apply_mask_pipeline_info);
}

bool union_merge_renderer::record(vk::raii::CommandBuffer & cmd,
                                  vk::Extent2D extent, std::span<const input> inputs)
{
	if (inputs.empty() or inputs.size() > max_inputs)
		return false;
	for (auto & in: inputs)
	{
		// Borrowed views only (never cached here): a null view means
		// the owning renderer retired the image without recording it
		// this frame — that input contributes nothing, but silently
		// dropping coverage under-reveals, so refuse the whole merge
		// loudly instead (caller releases + skips B this frame).
		if (in.view_array == vk::ImageView{})
			return false;
	}

	// Write→read hazard (same command buffer): every input was just
	// written as a color attachment and ends OPTIMAL. Transition to
	// SHADER_READ for the merge; every next record re-begins
	// UNDEFINED+CLEAR, so no transition back is needed.
	std::vector<vk::ImageMemoryBarrier> barriers;
	barriers.reserve(inputs.size());
	for (auto & in: inputs)
	{
		barriers.push_back(vk::ImageMemoryBarrier{
		        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
		        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .image = in.image,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .levelCount = 1,
		                .layerCount = 2,
		        },
		});
	}
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    vk::PipelineStageFlagBits::eFragmentShader,
	                    {},
	                    {},
	                    {},
	                    barriers);

	// Descriptor write (per call: contributor images cycle
	// independently, so nothing is stable across frames). Unused
	// slots re-bind input 0's view (same view in several slots is
	// legal; max() over duplicates is neutral). Ping-pong parity:
	// the other set may still be read by a prior in-flight command
	// buffer (inline fallback while the async CB flies).
	set_parity = !set_parity;
	vk::DescriptorSet set = *descriptor_sets[set_parity ? 1 : 0];
	vk::DescriptorImageInfo infos[max_inputs];
	for (uint32_t i = 0; i < max_inputs; ++i)
	{
		auto & in = inputs[std::min<size_t>(i, inputs.size() - 1)];
		infos[i] = vk::DescriptorImageInfo{
		        .sampler = *sampler,
		        .imageView = in.view_array,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
	}
	std::vector<vk::WriteDescriptorSet> writes;
	writes.reserve(max_inputs);
	for (uint32_t i = 0; i < max_inputs; ++i)
	{
		writes.push_back(vk::WriteDescriptorSet{
		        .dstSet = set,
		        .dstBinding = i,
		        .descriptorCount = 1,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .pImageInfo = &infos[i],
		});
	}
	device->updateDescriptorSets(writes, {});

	auto & tgt = merged_;
	cmd.setViewport(0, vk::Viewport{
	                        .x = 0,
	                        .y = 0,
	                        .width = (float)extent.width,
	                        .height = (float)extent.height,
	                        .minDepth = 0,
	                        .maxDepth = 1,
	                });
	// Full-extent scissor, set explicitly: the pipeline enables the
	// scissor dynamic state and prior passes in this command buffer
	// leave narrowed boxes behind (per-draw unions, content boxes).
	// Inheriting one would clip the merge (under-reveal) with no
	// validation complaint.
	cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extent});
	// DONT_CARE load: no clear values (the fullscreen triangle below
	// overwrites every pixel in the render area).
	vk::RenderPassBeginInfo begin{
	        .renderPass = *renderpass,
	        .framebuffer = *tgt.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = extent},
	        .clearValueCount = 0,
	        .pClearValues = nullptr,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, set, {});
	// Active input count for the uniform tap early-out: identical for
	// every invocation in the draw, so skipped taps never execute.
	// inputs.size() is 1..8 (checked on entry).
	cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{.count = (int32_t)inputs.size()});
	// Fullscreen triangle (blur.vert: 3 verts, no vertex buffers).
	cmd.draw(3, 1, 0, 0);
	cmd.endRenderPass();

	// Post-write bookkeeping: merged ends SHADER_READ for the
	// verdict apply below (same command buffer); the next record
	// re-begins UNDEFINED+CLEAR, so no transition back is needed.
	vk::ImageMemoryBarrier out_barrier{
	        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
	        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
	        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .image = *merged_.image,
	        .subresourceRange = {
	                .aspectMask = vk::ImageAspectFlagBits::eColor,
	                .levelCount = 1,
	                .layerCount = 2,
	        },
	};
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    vk::PipelineStageFlagBits::eFragmentShader,
	                    {},
	                    {},
	                    {},
	                    out_barrier);
	return true;
}

void union_merge_renderer::record_base(vk::raii::CommandBuffer & cmd, std::span<const geom_draw> draws)
{
	// Pending mesh uploads first (staging→device + VertexInput
	// barrier), mirroring the feather renderer: uploads are rare
	// (map-change only), so this is a no-op most frames.
	flush_uploads(cmd);

	// Even with no draws the CLEAR runs: a LOAD of undefined contents
	// would read as geometry and wrongly gate the verdict — fail-open
	// demands the explicit FAR clear.
	vk::ClearValue clear{};
	clear.color.float32 = std::array<float, 4>{layer_far, 0, 0, 0};
	vk::RenderPassBeginInfo begin{
	        .renderPass = *geom_renderpass,
	        .framebuffer = *layer_.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = layer_extent_},
	        .clearValueCount = 1,
	        .pClearValues = &clear,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	cmd.setViewport(0, vk::Viewport{
	                        .x = 0,
	                        .y = 0,
	                        .width = (float)layer_extent_.width,
	                        .height = (float)layer_extent_.height,
	                        .minDepth = 0,
	                        .maxDepth = 1,
	                });
	// Full-layer scissor, set explicitly: prior passes in this command
	// buffer may leave narrowed boxes behind (per-draw unions). The
	// raster covers the whole layer by design.
	cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = layer_extent_});
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *geom_pipeline);
	for (const auto & d: draws)
	{
		auto mit = layer_meshes.find(d.mesh);
		if (mit == layer_meshes.end() or mit->second.index_count == 0)
			continue;
		cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
		cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
		cmd.pushConstants<geom_push>(*geom_layout, vk::ShaderStageFlagBits::eVertex, 0,
		                             geom_push{.mvp = {{d.mvp[0], d.mvp[1]}}});
		cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
	}
	cmd.endRenderPass();

	// Write→read hazard (same command buffer): the layer is sampled by
	// the verdict in record_finish below.
	vk::ImageMemoryBarrier barrier{
	        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
	        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
	        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .image = *layer_.image,
	        .subresourceRange = {
	                .aspectMask = vk::ImageAspectFlagBits::eColor,
	                .levelCount = 1,
	                .layerCount = 2,
	        },
	};
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    vk::PipelineStageFlagBits::eFragmentShader,
	                    {},
	                    {},
	                    {},
	                    barrier);
}

void union_merge_renderer::create_verdict_state(const wivrn::decoder::ycbcr_definition & def, vk::Sampler decoder_sampler)
{
	// Full teardown in dependency order: sets reference pool + set
	// layout, pipeline references pipeline layout, pipeline layout
	// references the set layout. Caller guarantees no in-flight readers
	// (fresh object, or a waitIdle'd rebuild).
	verdict_descriptor_sets.clear();
	verdict_pipeline.clear();
	verdict_layout.clear();
	verdict_descriptor_layout.clear();
	// Bake the decoder's live sampler as the immutable sampler
	// (defoveator pattern): a separately built conversion proved
	// unidentifiable to validation despite matching parameters, so the
	// layout shares the decoder's object. verdict_immutable_src tracks
	// what the layout was baked with; a decoder-side sampler change
	// rebuilds before the next bind.
	verdict_immutable_src = def.has_conversion ? decoder_sampler : vk::Sampler{};
	// Binding 0 (common layer, plain view) stays mutable; binding 1
	// (virt) is immutable exactly when the view carries a conversion.
	// Binding 2 (foveation index maps) is a plain storage buffer,
	// rewritten per verdict frame. The immutable handle is copied by
	// layout creation (defoveator precedent: pointer to a local is fine).
	vk::Sampler immutable = verdict_immutable_src;
	vk::DescriptorSetLayoutBinding bindings[3] = {
	        {
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	        {
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	                .pImmutableSamplers = def.has_conversion ? &immutable : nullptr,
	        },
	        {
	                .binding = 2,
	                .descriptorType = vk::DescriptorType::eStorageBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	verdict_descriptor_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                                 .bindingCount = 3,
	                                                                                 .pBindings = bindings,
	                                                                         });
	for (int i = 0; i < 2; ++i)
	{
		vk::DescriptorSetAllocateInfo set_alloc{
		        .descriptorPool = *verdict_descriptor_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*verdict_descriptor_layout,
		};
		verdict_descriptor_sets.emplace_back(std::move(device->allocateDescriptorSets(set_alloc)[0]));
	}
	// Pipeline bakes the set layout: rebuilt alongside it, never apart
	// (binding fresh sets with a stale pipeline is a use error).
	vk::PipelineLayoutCreateInfo verdict_layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*verdict_descriptor_layout,
	        .pushConstantRangeCount = 0,
	        .pPushConstantRanges = nullptr,
	};
	verdict_layout = vk::raii::PipelineLayout(*device, verdict_layout_info);
	auto vert = load_shader(*device, "blur.vert");
	auto verdict_frag = load_shader(*device, "vr_verdict_mv.frag");
	vk::PipelineShaderStageCreateInfo verdict_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **verdict_frag,
	                .pName = "main",
	        },
	};
	// Static fullscreen state, identical to the shared init block.
	vk::PipelineVertexInputStateCreateInfo vertex_input{};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly{
	        .topology = vk::PrimitiveTopology::eTriangleList,
	};
	vk::PipelineViewportStateCreateInfo viewport_state{
	        .viewportCount = 1,
	        .scissorCount = 1,
	};
	vk::PipelineRasterizationStateCreateInfo rasterization{
	        .polygonMode = vk::PolygonMode::eFill,
	        .cullMode = vk::CullModeFlagBits::eNone,
	        .frontFace = vk::FrontFace::eCounterClockwise,
	        .lineWidth = 1,
	};
	vk::PipelineMultisampleStateCreateInfo multisample{
	        .rasterizationSamples = vk::SampleCountFlagBits::e1,
	};
	vk::PipelineDepthStencilStateCreateInfo depth_stencil{
	        .depthTestEnable = VK_FALSE,
	        .depthWriteEnable = VK_FALSE,
	};
	vk::PipelineColorBlendAttachmentState blend_attachment{
	        .blendEnable = VK_FALSE,
	        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
	};
	vk::PipelineColorBlendStateCreateInfo blend{
	        .attachmentCount = 1,
	        .pAttachments = &blend_attachment,
	};
	vk::DynamicState dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic{
	        .dynamicStateCount = 2,
	        .pDynamicStates = dynamics,
	};
	vk::GraphicsPipelineCreateInfo verdict_pipeline_info{
	        .stageCount = 2,
	        .pStages = verdict_stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pDepthStencilState = &depth_stencil,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *verdict_layout,
	        .renderPass = *verdict_renderpass,
	};
	verdict_pipeline = vk::raii::Pipeline(*device, *pipe_cache, verdict_pipeline_info);
	verdict_ycbcr_built = def;
}

void union_merge_renderer::ensure_verdict_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler decoder_sampler)
{
	// Rebuild when the definition changes OR the decoder sampler was
	// recreated (same params, new object: the baked immutable handle
	// must track the live object). Plain layouts don't bake a sampler,
	// so only converted state compares handles.
	if (verdict_ycbcr_built == def and (not def.has_conversion or verdict_immutable_src == decoder_sampler))
		return;
	// Depth format change (rare: renegotiation only): retire prior
	// submissions (including the verdict's own previous frame) before
	// rebuilding what they may still read.
	device->waitIdle();
	create_verdict_state(def, decoder_sampler);
	spdlog::info("Union merge verdict layout rebuilt (YCbCr {}, decoder-shared immutable sampler)", def.has_conversion ? "immutable" : "plain");
}

bool union_merge_renderer::record_finish(vk::raii::CommandBuffer & cmd,
                                         size_t image_index,
                                         vk::Extent2D extent,
                                         const virt_input & virt_in)
{
	if (image_index >= targets.size())
		return false;

	// Single merge size: the submit image, the owned merged/layer/verdict
	// targets and the verdict index maps all share it (ensure() rebuilds
	// them together). The apply below reads extent, the verdict reads
	// layer_extent_ — divergence would silently misalign the two.
	assert(extent.width == layer_extent_.width and extent.height == layer_extent_.height);

	// Verdict pass at full merge res: fused foveated-decode + exact
	// compare. Mask-only frames (!valid: toggle off, unpaired depth,
	// transition frames) skip the pass + barrier entirely and apply
	// merged directly below (pixel-identical to a cleared-to-1
	// verdict). DONT_CARE load: the fullscreen draw overwrites every
	// pixel, so no CLEAR is needed.
	bool do_verdict = virt_in.valid and virt_in.view and virt_in.sampler;
	if (do_verdict)
	{
		vk::RenderPassBeginInfo verdict_begin{
		        .renderPass = *verdict_renderpass,
		        .framebuffer = *verdict_.framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = layer_extent_},
		        .clearValueCount = 0,
		        .pClearValues = nullptr,
		};
		cmd.beginRenderPass(verdict_begin, vk::SubpassContents::eInline);
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)layer_extent_.width,
		                        .height = (float)layer_extent_.height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = layer_extent_});
			// YCbCr immutable-sampler layout (VUID-02738): rebuild when the
			// view's conversion definition or the decoder sampler handle
			// changes. waitIdle'd, rare.
			ensure_verdict_layout(virt_in.ycbcr, virt_in.sampler);
			verdict_set_parity = !verdict_set_parity;
			// Foveation index maps, CPU-expanded per frame into the
			// ping-ponged SSBO (depth_overlay UBO discipline: memcpy +
			// flush, rewritten before the draw that reads them). Merge->render
			// scale baked in; packed dims from the paired depth frame.
			auto & tables = verdict_tables[verdict_set_parity ? 1 : 0];
			{
				const int W = (int)layer_extent_.width, H = (int)layer_extent_.height;
				const int render_w = virt_in.render_w, render_h = virt_in.render_h;
				const int packed_w = virt_in.packed_w, packed_h = virt_in.packed_h;
				assert(W > 0 and H > 0);
				verdict_map_scratch.resize((size_t)2 * (W + H));
				uint32_t * map = verdict_map_scratch.data();
				for (int eye = 0; eye < 2; ++eye)
				{
					// Server invariant: compute_params always emits odd,
					// non-empty runs (fill_param_2d ends count*2-1; the
					// non-foveated fallback emits one element).
					assert(virt_in.foveation[eye].x.size() % 2 == 1);
					assert(virt_in.foveation[eye].y.size() % 2 == 1);
					expand_foveation_axis(virt_in.foveation[eye].x, render_w, W, packed_w, map + eye * W);
					expand_foveation_axis(virt_in.foveation[eye].y, render_h, H, packed_h, map + 2 * W + eye * H);
				}
				std::memcpy(tables.map(), map, verdict_map_scratch.size() * sizeof(uint32_t));
				tables.unmap();
				// Non-coherent host-visible memory: flush, or the GPU reads
				// stale tables forever (depth_overlay precedent).
				vmaFlushAllocation(vk_allocator::instance(), tables, 0, verdict_map_scratch.size() * sizeof(uint32_t));
			}
			vk::DescriptorSet vset = *verdict_descriptor_sets[verdict_set_parity ? 1 : 0];
			vk::DescriptorImageInfo layer_info{
			        .sampler = *sampler_nearest,
			        .imageView = *layer_.view_array,
			        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			};
			// Decoder SBS image: GENERAL layout at frame top (same
			// steady-state discipline as the video path). The write sampler
			// is ignored when the layout baked an immutable one (defoveator
			// pattern); pass the live decoder sampler either way.
			vk::DescriptorImageInfo virt_info{
			        .sampler = virt_in.sampler,
			        .imageView = virt_in.view,
			        .imageLayout = vk::ImageLayout::eGeneral,
			};
			vk::DescriptorBufferInfo tables_info{
			        .buffer = vk::Buffer(tables),
			        .offset = 0,
			        .range = verdict_tables_bytes,
			};
			std::vector<vk::WriteDescriptorSet> vwrites;
			vwrites.reserve(3);
			vwrites.push_back(vk::WriteDescriptorSet{
			        .dstSet = vset,
			        .dstBinding = 0,
			        .descriptorCount = 1,
			        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			        .pImageInfo = &layer_info,
			});
			vwrites.push_back(vk::WriteDescriptorSet{
			        .dstSet = vset,
			        .dstBinding = 1,
			        .descriptorCount = 1,
			        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			        .pImageInfo = &virt_info,
			});
			vwrites.push_back(vk::WriteDescriptorSet{
			        .dstSet = vset,
			        .dstBinding = 2,
			        .descriptorCount = 1,
			        .descriptorType = vk::DescriptorType::eStorageBuffer,
			        .pBufferInfo = &tables_info,
			});
			device->updateDescriptorSets(vwrites, {});
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *verdict_pipeline);
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *verdict_layout, 0, vset, {});
			cmd.draw(3, 1, 0, 0);
		cmd.endRenderPass();
		vk::ImageMemoryBarrier verdict_barrier{
		        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
		        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .image = *verdict_.image,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .levelCount = 1,
		                .layerCount = 2,
		        },
		};
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                    vk::PipelineStageFlagBits::eFragmentShader,
		                    {},
		                    {},
		                    {},
		                    verdict_barrier);
	}

	// Apply pass into the acquired submit image. Verdict path:
	// merged (NEAREST) x verdict (NEAREST, already full-res — no
	// upscale). Mask-only path (no verdict pass above): merged straight
	// through (one tap, pixel-identical to x1). RGB zero; the
	// submit-time REPLACE preserves dst color.
	apply_set_parity = !apply_set_parity;
	vk::DescriptorSet aset = *apply_descriptor_sets[apply_set_parity ? 1 : 0];
	vk::DescriptorImageInfo merged_info{
	        .sampler = *sampler_nearest,
	        .imageView = *merged_.view_array,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	std::vector<vk::WriteDescriptorSet> apply_writes;
	apply_writes.reserve(do_verdict ? 2 : 1);
	apply_writes.push_back(vk::WriteDescriptorSet{
	        .dstSet = aset,
	        .dstBinding = 0,
	        .descriptorCount = 1,
	        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	        .pImageInfo = &merged_info,
	});
	if (do_verdict)
	{
		vk::DescriptorImageInfo apply_verdict_info{
		        .sampler = *sampler_nearest,
		        .imageView = *verdict_.view_array,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		apply_writes.push_back(vk::WriteDescriptorSet{
		        .dstSet = aset,
		        .dstBinding = 1,
		        .descriptorCount = 1,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .pImageInfo = &apply_verdict_info,
		});
	}
	device->updateDescriptorSets(apply_writes, {});
	auto & tgt = targets[image_index];
	cmd.setViewport(0, vk::Viewport{
	                        .x = 0,
	                        .y = 0,
	                        .width = (float)extent.width,
	                        .height = (float)extent.height,
	                        .minDepth = 0,
	                        .maxDepth = 1,
	                });
	cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extent});
	// DONT_CARE load: no clear values (the fullscreen draw below
	// overwrites every pixel in the render area).
	vk::RenderPassBeginInfo begin{
	        .renderPass = *renderpass,
	        .framebuffer = *tgt.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = extent},
	        .clearValueCount = 0,
	        .pClearValues = nullptr,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, do_verdict ? *apply_pipeline : *apply_mask_pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *apply_layout, 0, aset, {});
	cmd.draw(3, 1, 0, 0);
	cmd.endRenderPass();

	// Post-write bookkeeping (feather submit-barrier precedent):
	// color write out, compositor read in, layout unchanged.
	vk::ImageMemoryBarrier out_barrier{
	        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
	        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
	        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
	        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .image = swapchain_.image(image_index),
	        .subresourceRange = {
	                .aspectMask = vk::ImageAspectFlagBits::eColor,
	                .levelCount = 1,
	                .layerCount = 2,
	        },
	};
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    vk::PipelineStageFlagBits::eBottomOfPipe,
	                    {},
	                    {},
	                    {},
	                    out_barrier);
	return true;
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> union_merge_renderer::make_buffer(vk::DeviceSize size,
                                                                                      vk::BufferUsageFlags usage,
                                                                                      vk::MemoryPropertyFlags properties)
{
	vk::BufferCreateInfo buffer_info{
	        .size = size,
	        .usage = usage,
	        .sharingMode = vk::SharingMode::eExclusive,
	};
	vk::raii::Buffer buffer(*device, buffer_info);
	auto requirements = buffer.getMemoryRequirements();
	vk::MemoryAllocateInfo alloc_info{
	        .allocationSize = requirements.size,
	        .memoryTypeIndex = find_memory_type(*physical_device, requirements.memoryTypeBits, properties),
	};
	vk::raii::DeviceMemory memory(*device, alloc_info);
	buffer.bindMemory(*memory, 0);
	return {std::move(buffer), std::move(memory)};
}

void union_merge_renderer::flush_uploads(vk::raii::CommandBuffer & cmd)
{
	for (auto & [key, mesh]: layer_meshes)
	{
		(void)key;
		if (not mesh.pending_upload)
			continue;
		auto [vertex_bytes, index_bytes] = *mesh.pending_upload;
		mesh.pending_upload.reset();
		vk::BufferCopy vertex_copy{
		        .srcOffset = 0,
		        .dstOffset = 0,
		        .size = vertex_bytes,
		};
		cmd.copyBuffer(*mesh.staging_buffer, *mesh.vertex_buffer, vertex_copy);
		vk::BufferCopy index_copy{
		        .srcOffset = vertex_bytes,
		        .dstOffset = 0,
		        .size = index_bytes,
		};
		cmd.copyBuffer(*mesh.staging_buffer, *mesh.index_buffer, index_copy);
		std::vector<vk::BufferMemoryBarrier> barriers = {
		        vk::BufferMemoryBarrier{
		                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
		                .dstAccessMask = vk::AccessFlagBits::eVertexAttributeRead,
		                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                .buffer = *mesh.vertex_buffer,
		                .offset = 0,
		                .size = vertex_bytes,
		        },
		        vk::BufferMemoryBarrier{
		                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
		                .dstAccessMask = vk::AccessFlagBits::eIndexRead,
		                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                .buffer = *mesh.index_buffer,
		                .offset = 0,
		                .size = index_bytes,
		        },
		};
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                    vk::PipelineStageFlagBits::eVertexInput,
		                    {},
		                    {},
		                    barriers,
		                    {});
	}
}

void union_merge_renderer::set_layer_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup)
{
	if (soup.indices.empty() or soup.vertices.empty())
	{
		remove_layer_mesh(key);
		return;
	}
	// Rare path (map change): make sure no in-flight frame still reads
	// the old buffers before replacing them.
	device->waitIdle();
	mesh_buffers & mesh = layer_meshes[key];
	vk::DeviceSize vertex_bytes = soup.vertices.size() * sizeof(XrVector3f);
	vk::DeviceSize index_bytes = soup.indices.size() * sizeof(uint32_t);
	vk::DeviceSize total = vertex_bytes + index_bytes;
	if (mesh.staging_size < total)
	{
		std::tie(mesh.staging_buffer, mesh.staging_memory) = make_buffer(
		        total,
		        vk::BufferUsageFlagBits::eTransferSrc,
		        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		mesh.staging_size = total;
	}
	void * mapped = mesh.staging_memory.mapMemory(0, total);
	std::memcpy(mapped, soup.vertices.data(), vertex_bytes);
	std::memcpy((char *)mapped + vertex_bytes, soup.indices.data(), index_bytes);
	mesh.staging_memory.unmapMemory();
	std::tie(mesh.vertex_buffer, mesh.vertex_memory) = make_buffer(
	        vertex_bytes,
	        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
	        vk::MemoryPropertyFlagBits::eDeviceLocal);
	std::tie(mesh.index_buffer, mesh.index_memory) = make_buffer(
	        index_bytes,
	        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
	        vk::MemoryPropertyFlagBits::eDeviceLocal);
	mesh.pending_upload = {vertex_bytes, index_bytes};
	mesh.index_count = (uint32_t)soup.indices.size();
	glm::vec3 lo{soup.vertices[0].x, soup.vertices[0].y, soup.vertices[0].z};
	glm::vec3 hi = lo;
	for (const auto & v: soup.vertices)
	{
		lo = glm::min(lo, glm::vec3(v.x, v.y, v.z));
		hi = glm::max(hi, glm::vec3(v.x, v.y, v.z));
	}
	mesh.aabb_min = lo;
	mesh.aabb_max = hi;
}

void union_merge_renderer::remove_layer_mesh(const std::string & key)
{
	auto it = layer_meshes.find(key);
	if (it == layer_meshes.end())
		return;
	// Same in-flight hazard as replace.
	device->waitIdle();
	layer_meshes.erase(it);
}

std::optional<std::pair<glm::vec3, glm::vec3>> union_merge_renderer::layer_mesh_bounds(const std::string & key) const
{
	auto it = layer_meshes.find(key);
	if (it == layer_meshes.end())
		return std::nullopt;
	return std::make_pair(it->second.aabb_min, it->second.aabb_max);
}
