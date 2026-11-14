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

#include "render/feather_mask.h"

#include "vk/shader.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace
{
uint32_t find_memory_type(vk::raii::PhysicalDevice & physical_device, uint32_t type_bits, vk::MemoryPropertyFlags properties)
{
	auto mem = physical_device.getMemoryProperties();
	for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
	{
		if ((type_bits & (1u << i)) and (mem.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	throw std::runtime_error("feather_mask: no suitable memory type");
}

struct blur_push
{
	float texel[2];
	float dir[2];
	float spread;
	float pad = 0;
};

// Silhouette raster push block. Mirrors mask_mv.vert/frag.glsl: per-eye
// mvp selected by gl_ViewIndex. 132B exceeds the 128-byte Vulkan minimum —
// Adreno 740 (Quest 3) provides 256B; the ctor and ensure_hidden throw
// loudly below when the device reports less.
struct silhouette_mv_push
{
	std::array<glm::mat4, 2> mvp;
	float opacity = 1;
};
static_assert(sizeof(silhouette_mv_push) == 132);
static_assert(sizeof(silhouette_mv_push) <= 256);

// Erosion push block: source texel size, pass direction and box radius
// (matches blur_push at 24 bytes, so all fullscreen passes share one
// layout).
struct erode_push
{
	float texel[2];
	float dir[2];
	float radius = 0;
	float pad = 0;
};
static_assert(sizeof(erode_push) == 24);

// Erosion radius selector for internal-only feathering: the blur support
// reaches ~2*spread past the silhouette (spread is defined in blur-input
// texels), so eroding by that much pulls the band inside. Returns 0
// (skip: sub-full-pixel band, erosion would only shrink windows) or
// ceil(2*spread) otherwise (always rounds outward: a sliver of inset
// beats any outward halo for mapped windows). Two separable 1D min
// passes compose the exact box, so any feather width stays within a
// texel of the ideal inset by construction. Clamped to the shader's
// tap range (8).
int erode_radius(float spread, float fullpx_per_texel)
{
	float bleed = 2.f * spread;
	if (bleed * fullpx_per_texel < 1.f)
		return 0;
	int r = (int)std::ceil(bleed - 1e-6f);
	return std::clamp(r, 0, 8);
}

// RAII section timer for the record() diagnostics (mask_stage_cpu): adds
// the scoped wall time in milliseconds to *acc on destruction.
struct section_clock
{
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	double * acc;
	explicit section_clock(double * a) :
	        acc(a) {}
	~section_clock()
	{
		*acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	}
};

// Single-channel intermediates (mask alpha only): quarter the bytes of
// RGBA8 at identical 8-bit precision. Submitted images stay RGBA8 (the
// blend-factor vocabulary needs an alpha channel).
constexpr vk::Format blur_format = vk::Format::eR8Unorm;

// Depth for the reversed-draw raster path (D16 depth-test-only: never
// sampled, device-local like the intermediates; D16 attachment support is
// mandatory per spec, so no format fallback).
constexpr vk::Format depth_format = vk::Format::eD16Unorm;

// Downsample level extent (level 1 = half): exact halving, so the 2x2 box
// mapping stays exact. Shared with ensure_targets: sizes must match.
// The mask extent is 64-quantized upstream, hence divisible throughout.
vk::Extent2D level_extent(vk::Extent2D full, int level)
{
	uint32_t w = full.width;
	uint32_t h = full.height;
	for (int i = 0; i < level; ++i)
	{
		w = std::max(8u, w / 2);
		h = std::max(8u, h / 2);
	}
	return {w, h};
}
} // namespace

std::shared_ptr<vk::raii::PipelineCache> feather_mask_renderer::pipeline_cache(vk::raii::Device & device)
{
	static std::map<VkDevice, std::weak_ptr<vk::raii::PipelineCache>> caches;
	VkDevice key = *device;
	if (auto hit = caches[key].lock())
		return hit;
	auto built = std::make_shared<vk::raii::PipelineCache>(device, vk::PipelineCacheCreateInfo{}, nullptr);
	caches[key] = built;
	return built;
}

void feather_mask_renderer::warm_pipelines(vk::raii::Device & device,
                                           vk::raii::PhysicalDevice & physical_device,
                                           vk::Format format)
{
	// Throwaway renderer: its pipeline compiles populate the shared
	// cache above (and the page cache), so the first real group pays
	// ~nothing. The descriptor pool etc. built here dies with the
	// temporary.
	feather_mask_renderer(device, physical_device, format);
}

feather_mask_renderer::feather_mask_renderer(vk::raii::Device & device_,
                                             vk::raii::PhysicalDevice & physical_device_,
                                             vk::Format format_) :
        device(device_),
        physical_device(physical_device_),
        format(format_),
        pipe_cache(pipeline_cache(device_))
{
	vk::VertexInputBindingDescription binding{
	        .binding = 0,
	        .stride = sizeof(XrVector3f),
	        .inputRate = vk::VertexInputRate::eVertex,
	};
	vk::VertexInputAttributeDescription attribute{
	        .location = 0,
	        .binding = 0,
	        .format = vk::Format::eR32G32B32Sfloat,
	        .offset = 0,
	};
	vk::PipelineVertexInputStateCreateInfo vertex_input{
	        .vertexBindingDescriptionCount = 1,
	        .pVertexBindingDescriptions = &binding,
	        .vertexAttributeDescriptionCount = 1,
	        .pVertexAttributeDescriptions = &attribute,
	};

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

	// Same vertex input/assembly/viewport/rasterization/multisample/
	// dynamic state throughout (reused locals); only blend
	// and depth-stencil state vary (consumed by the MV variants below).
	vk::PipelineColorBlendAttachmentState min_attachment{
	        .blendEnable = VK_TRUE,
	        .srcColorBlendFactor = vk::BlendFactor::eOne,
	        .dstColorBlendFactor = vk::BlendFactor::eOne,
	        .colorBlendOp = vk::BlendOp::eMin,
	        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
	        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
	        .alphaBlendOp = vk::BlendOp::eMin,
	        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
	};
	vk::PipelineColorBlendStateCreateInfo min_blend{
	        .attachmentCount = 1,
	        .pAttachments = &min_attachment,
	};
	vk::PipelineDepthStencilStateCreateInfo depth_write{
	        .depthTestEnable = VK_TRUE,
	        .depthWriteEnable = VK_TRUE,
	        // GREATER, not LESS: scene::projection_matrix is reversed-Z
	        // (near = 1, far = 0), matching scene_renderer. LESS here
	        // inverts the interference (behind cuts, front doesn't).
	        .depthCompareOp = vk::CompareOp::eGreater,
	};
	vk::PipelineDepthStencilStateCreateInfo depth_test{
	        .depthTestEnable = VK_TRUE,
	        .depthWriteEnable = VK_FALSE,
	        .depthCompareOp = vk::CompareOp::eGreater,
	};

	// Multiview raster path (one draw broadcasts to both eyes):
	// CLEAR/STORE/OPTIMAL attachments plus viewMask=both. Uses
	// mask_mv.vert/frag (plain gl_ViewIndex indexing, no mat3/ternary)
	// with the 132B MVP-pair push (silhouette_mv_push shape); layout is
	// dedicated here so the raster MV pipes never depend on the lazy
	// hidden init.
	if (physical_device.getProperties().limits.maxPushConstantsSize < sizeof(silhouette_mv_push))
		throw std::runtime_error("feather silhouette multiview needs 132B push constants");
	{
		vk::PushConstantRange mv_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(silhouette_mv_push),
		};
		vk::PipelineLayoutCreateInfo mv_layout_info{
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &mv_push_range,
		};
		silhouette_mv_layout = vk::raii::PipelineLayout(device, mv_layout_info);
	}
	auto mv_vert = load_shader(device, "mask_mv.vert");
	auto mv_frag = load_shader(device, "mask_mv.frag");
	vk::PipelineShaderStageCreateInfo mv_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **mv_vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **mv_frag,
	                .pName = "main",
	        },
	};
	uint32_t raster_mv_mask = 0b11, raster_mv_corr = 0b11;
	vk::RenderPassMultiviewCreateInfo raster_mv_info{
	        .subpassCount = 1,
	        .pViewMasks = &raster_mv_mask,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &raster_mv_corr,
	};
	vk::AttachmentDescription raster_mv_att{
	        .format = format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentDescription raster_r8_mv_att{
	        .format = blur_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference raster_mv_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription raster_mv_sub{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &raster_mv_ref,
	};
	renderpass_mv = vk::raii::RenderPass(device, vk::RenderPassCreateInfo{
	        .pNext = &raster_mv_info,
	        .attachmentCount = 1,
	        .pAttachments = &raster_mv_att,
	        .subpassCount = 1,
	        .pSubpasses = &raster_mv_sub,
	});
	raster_r8_renderpass_mv = vk::raii::RenderPass(device, vk::RenderPassCreateInfo{
	        .pNext = &raster_mv_info,
	        .attachmentCount = 1,
	        .pAttachments = &raster_r8_mv_att,
	        .subpassCount = 1,
	        .pSubpasses = &raster_mv_sub,
	});
	auto make_mv_depth_rp = [&](vk::Format color_format) {
		vk::AttachmentDescription atts[2] = {
		        {
		                .format = color_format,
		                .samples = vk::SampleCountFlagBits::e1,
		                .loadOp = vk::AttachmentLoadOp::eClear,
		                .storeOp = vk::AttachmentStoreOp::eStore,
		                .initialLayout = vk::ImageLayout::eUndefined,
		                .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        },
		        {
		                .format = depth_format,
		                .samples = vk::SampleCountFlagBits::e1,
		                .loadOp = vk::AttachmentLoadOp::eClear,
		                .storeOp = vk::AttachmentStoreOp::eDontCare,
		                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
		                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		                .initialLayout = vk::ImageLayout::eUndefined,
		                .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		        },
		};
		vk::AttachmentReference mv_color_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference mv_depth_ref{
		        .attachment = 1,
		        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::SubpassDescription mv_depth_sub{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &mv_color_ref,
		        .pDepthStencilAttachment = &mv_depth_ref,
		};
		return vk::raii::RenderPass(device, vk::RenderPassCreateInfo{
		        .pNext = &raster_mv_info,
		        .attachmentCount = 2,
		        .pAttachments = atts,
		        .subpassCount = 1,
		        .pSubpasses = &mv_depth_sub,
		});
	};
	depth_renderpass_mv = make_mv_depth_rp(format);
	raster_r8_depth_renderpass_mv = make_mv_depth_rp(blur_format);
	auto make_raster_mv_variant = [&](vk::raii::RenderPass & rp, vk::PipelineColorBlendStateCreateInfo & bl,
	                                 vk::PipelineDepthStencilStateCreateInfo * ds) {
		vk::GraphicsPipelineCreateInfo info{
		        .stageCount = 2,
		        .pStages = mv_stages,
		        .pVertexInputState = &vertex_input,
		        .pInputAssemblyState = &input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pDepthStencilState = ds,
		        .pColorBlendState = &bl,
		        .pDynamicState = &dynamic,
		        .layout = *silhouette_mv_layout,
		        .renderPass = *rp,
		};
		return vk::raii::Pipeline(device, *pipeline_cache(device), info);
	};
	pipeline_mv = make_raster_mv_variant(renderpass_mv, blend, nullptr);
	depth_pipeline_mv = make_raster_mv_variant(depth_renderpass_mv, blend, &depth_write);
	reverse_pipeline_mv = make_raster_mv_variant(depth_renderpass_mv, min_blend, &depth_test);
	raster_r8_pipeline_mv = make_raster_mv_variant(raster_r8_renderpass_mv, blend, nullptr);
	raster_r8_depth_pipeline_mv = make_raster_mv_variant(raster_r8_depth_renderpass_mv, blend, &depth_write);
	raster_r8_reverse_pipeline_mv = make_raster_mv_variant(raster_r8_depth_renderpass_mv, min_blend, &depth_test);

	// Blur render pass: fully overwritten every use (fullscreen triangle),
	// so no clear and no prior contents needed.
	vk::AttachmentDescription blur_attachment{
	        .format = format,
	        .samples = vk::SampleCountFlagBits::e1,
	        // CLEAR: the V draw is scissored to the content box, so the
	        // exterior must be defined here (mask 0). Interior and band
	        // are fully overwritten by the draw below.
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference blur_color_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription blur_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &blur_color_ref,
	};
	// Multiview blur passes (viewMask = both eyes, correlated): one
	// fullscreen draw broadcasts to both layers. Fully overwritten
	// every use, so no clear and no prior contents needed.
	uint32_t mv_view_mask = 0b11, mv_corr_mask = 0b11;
	vk::RenderPassMultiviewCreateInfo mv_rp_info{
	        .subpassCount = 1,
	        .pViewMasks = &mv_view_mask,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &mv_corr_mask,
	};
	vk::RenderPassCreateInfo blur_rp_mv_info{
	        .pNext = &mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &blur_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &blur_subpass,
	};
	blur_renderpass_mv = vk::raii::RenderPass(device, blur_rp_mv_info);

	// R8 variant of the blur pass for the intermediates (same DONT_CARE
	// shape). The RGBA8 pass above serves only the V upscale into
	// swapchain images now.
	vk::AttachmentDescription blur_r8_attachment{
	        .format = blur_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eDontCare,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference blur_r8_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription blur_r8_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &blur_r8_ref,
	};
	vk::RenderPassCreateInfo blur_r8_rp_mv_info{
	        .pNext = &mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &blur_r8_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &blur_r8_subpass,
	};
	blur_r8_renderpass_mv = vk::raii::RenderPass(device, blur_r8_rp_mv_info);

	// Dedicated compare pass (see header): same R8 MV attachment shape,
	// loadOp=CLEAR so every texel the merge upscales is defined.
	vk::AttachmentDescription compare_attachment{
	        .format = blur_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference compare_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription compare_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &compare_ref,
	};
	vk::RenderPassCreateInfo compare_rp_mv_info{
	        .pNext = &mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &compare_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &compare_subpass,
	};
	compare_renderpass_mv = vk::raii::RenderPass(device, compare_rp_mv_info);

	// Merged erode-H/erode-V/blur-H triple: one renderpass, 3
	// subpasses, attachments [src, tmp] (tier-1: A,B; tiered:
	// h,eblur — identical specs, shared object). src enters live
	// (LOAD) and is never read after subpass 2; tmp is scratch at
	// entry (DONT_CARE) and stored for the V upscale after. All refs
	// are GENERAL: sampler reads across subpasses need a samplable
	// layout, and attachments unused in a subpass do NOT transition,
	// so ATTACH refs would be illegally sampled. Tile residency (not
	// compression) is the win inside; GMEM traffic stays compressed
	// via the OPTIMAL raster/downsample passes outside.
	vk::AttachmentDescription triple_attachments[2] = {
	        {
	                .format = blur_format,
	                .samples = vk::SampleCountFlagBits::e1,
	                .loadOp = vk::AttachmentLoadOp::eLoad,
	                .storeOp = vk::AttachmentStoreOp::eDontCare,
	                .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	                .finalLayout = vk::ImageLayout::eGeneral,
	        },
	        {
	                .format = blur_format,
	                .samples = vk::SampleCountFlagBits::e1,
	                .loadOp = vk::AttachmentLoadOp::eDontCare,
	                .storeOp = vk::AttachmentStoreOp::eStore,
	                .initialLayout = vk::ImageLayout::eUndefined,
	                .finalLayout = vk::ImageLayout::eGeneral,
	        },
	};
	vk::AttachmentReference triple_color0{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eGeneral,
	};
	vk::AttachmentReference triple_color1{
	        .attachment = 1,
	        .layout = vk::ImageLayout::eGeneral,
	};
	vk::SubpassDescription triple_subpasses[3] = {
	        {
	                .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	                .colorAttachmentCount = 1,
	                .pColorAttachments = &triple_color1,
	        },
	        {
	                .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	                .colorAttachmentCount = 1,
	                .pColorAttachments = &triple_color0,
	        },
	        {
	                .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	                .colorAttachmentCount = 1,
	                .pColorAttachments = &triple_color1,
	        },
	};
	vk::SubpassDependency triple_deps[2] = {
	        {
	                .srcSubpass = 0,
	                .dstSubpass = 1,
	                .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eColorAttachmentWrite,
	        },
	        {
	                .srcSubpass = 1,
	                .dstSubpass = 2,
	                .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eColorAttachmentWrite,
	        },
	};
	uint32_t triple_view_masks[3] = {0b11, 0b11, 0b11};
	uint32_t triple_corr_mask = 0b11;
	vk::RenderPassMultiviewCreateInfo triple_mv{
	        .subpassCount = 3,
	        .pViewMasks = triple_view_masks,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &triple_corr_mask,
	};
	vk::RenderPassCreateInfo triple_rp_info{
	        .pNext = &triple_mv,
	        .attachmentCount = 2,
	        .pAttachments = triple_attachments,
	        .subpassCount = 3,
	        .pSubpasses = triple_subpasses,
	        .dependencyCount = 2,
	        .pDependencies = triple_deps,
	};
	triple_rp = vk::raii::RenderPass(device, triple_rp_info);

	auto blur_vert = load_shader(device, "blur.vert");

	vk::PipelineVertexInputStateCreateInfo blur_vertex_input{};
	vk::PipelineInputAssemblyStateCreateInfo blur_input_assembly{
	        .topology = vk::PrimitiveTopology::eTriangleList,
	};
	vk::PipelineViewportStateCreateInfo blur_viewport_state{
	        .viewportCount = 1,
	        .scissorCount = 1,
	};
	vk::PipelineRasterizationStateCreateInfo blur_rasterization{
	        .polygonMode = vk::PolygonMode::eFill,
	        .cullMode = vk::CullModeFlagBits::eNone,
	        .frontFace = vk::FrontFace::eCounterClockwise,
	        .lineWidth = 1,
	};
	vk::PipelineMultisampleStateCreateInfo blur_multisample{
	        .rasterizationSamples = vk::SampleCountFlagBits::e1,
	};
	vk::PipelineColorBlendAttachmentState blur_blend_attachment{
	        .blendEnable = VK_FALSE,
	        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
	};
	vk::PipelineColorBlendStateCreateInfo blur_blend{
	        .attachmentCount = 1,
	        .pAttachments = &blur_blend_attachment,
	};
	vk::DynamicState blur_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo blur_dynamic{
	        .dynamicStateCount = 2,
	        .pDynamicStates = blur_dynamics,
	};

	vk::DescriptorSetLayoutBinding sampler_binding{
	        .binding = 0,
	        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	        .descriptorCount = 1,
	        .stageFlags = vk::ShaderStageFlagBits::eFragment,
	};
	vk::DescriptorSetLayoutCreateInfo set_layout_info{
	        .bindingCount = 1,
	        .pBindings = &sampler_binding,
	};
	descriptor_layout = vk::raii::DescriptorSetLayout(device, set_layout_info);

	vk::PushConstantRange blur_push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = 24,
	};
	vk::PipelineLayoutCreateInfo blur_layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*descriptor_layout,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &blur_push_range,
	};
	blur_layout = vk::raii::PipelineLayout(device, blur_layout_info);

	// Fullscreen pipelines (multiview: one draw broadcasts to both
	// eyes). Built here so warm_pipelines covers them like every
	// other pipe.
	auto blur_mv_frag = load_shader(device, "blur_mv.frag");
	vk::PipelineShaderStageCreateInfo blur_mv_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **blur_vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **blur_mv_frag,
	                .pName = "main",
	        },
	};
	vk::GraphicsPipelineCreateInfo blur_pipeline_mv_info{
	        .stageCount = 2,
	        .pStages = blur_mv_stages,
	        .pVertexInputState = &blur_vertex_input,
	        .pInputAssemblyState = &blur_input_assembly,
	        .pViewportState = &blur_viewport_state,
	        .pRasterizationState = &blur_rasterization,
	        .pMultisampleState = &blur_multisample,
	        .pColorBlendState = &blur_blend,
	        .pDynamicState = &blur_dynamic,
	        .layout = *blur_layout,
	        .renderPass = *blur_renderpass_mv,
	};
	blur_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), blur_pipeline_mv_info);

	// R8 variant for the intermediates (same DONT_CARE shape). The
	// RGBA8 pipe above serves only the V upscale into swapchain images.
	vk::GraphicsPipelineCreateInfo blur_r8_pipeline_mv_info{
	        .stageCount = 2,
	        .pStages = blur_mv_stages,
	        .pVertexInputState = &blur_vertex_input,
	        .pInputAssemblyState = &blur_input_assembly,
	        .pViewportState = &blur_viewport_state,
	        .pRasterizationState = &blur_rasterization,
	        .pMultisampleState = &blur_multisample,
	        .pColorBlendState = &blur_blend,
	        .pDynamicState = &blur_dynamic,
	        .layout = *blur_layout,
	        .renderPass = *blur_r8_renderpass_mv,
	};
	blur_r8_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), blur_r8_pipeline_mv_info);

	// Box-downsample pipeline: same fullscreen vertex shader, same layout
	// (reads only the push prefix); R8 throughout (intermediates only).
	auto downsample_mv_frag = load_shader(device, "downsample_mv.frag");
	vk::PipelineShaderStageCreateInfo downsample_mv_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **blur_vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **downsample_mv_frag,
	                .pName = "main",
	        },
	};
	vk::PipelineVertexInputStateCreateInfo downsample_vertex_input{};
	vk::GraphicsPipelineCreateInfo downsample_pipeline_mv_info{
	        .stageCount = 2,
	        .pStages = downsample_mv_stages,
	        .pVertexInputState = &downsample_vertex_input,
	        .pInputAssemblyState = &blur_input_assembly,
	        .pViewportState = &blur_viewport_state,
	        .pRasterizationState = &blur_rasterization,
	        .pMultisampleState = &blur_multisample,
	        .pColorBlendState = &blur_blend,
	        .pDynamicState = &blur_dynamic,
	        .layout = *blur_layout,
	        .renderPass = *blur_r8_renderpass_mv,
	};
	downsample_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), downsample_pipeline_mv_info);

	// Merged-triple pipes (always multiview; subpass index baked in):
	// erode in subpasses 0/1, blur-H in 2. Same shaders, layout and
	// sampler discipline as the standalone passes (nearest for erode).
	// Built here so warm_pipelines covers them.
	auto triple_blur_frag = load_shader(device, "blur_mv.frag");
	vk::PipelineShaderStageCreateInfo triple_blur_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **blur_vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **triple_blur_frag,
	                .pName = "main",
	        },
	};
	auto triple_erode_frag = load_shader(device, "erode1d_mv.frag");
	vk::PipelineShaderStageCreateInfo triple_erode_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **blur_vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **triple_erode_frag,
	                .pName = "main",
	        },
	};
	auto make_triple_pipe = [&](vk::PipelineShaderStageCreateInfo * triple_stages, uint32_t subpass) {
		vk::GraphicsPipelineCreateInfo triple_pipe_info{
		        .stageCount = 2,
		        .pStages = triple_stages,
		        .pVertexInputState = &downsample_vertex_input,
		        .pInputAssemblyState = &blur_input_assembly,
		        .pViewportState = &blur_viewport_state,
		        .pRasterizationState = &blur_rasterization,
		        .pMultisampleState = &blur_multisample,
		        .pColorBlendState = &blur_blend,
		        .pDynamicState = &blur_dynamic,
		        .layout = *blur_layout,
		        .renderPass = *triple_rp,
		        .subpass = subpass,
		};
		return vk::raii::Pipeline(device, *pipeline_cache(device), triple_pipe_info);
	};
	triple_erode_0 = make_triple_pipe(triple_erode_stages, 0);
	triple_erode_1 = make_triple_pipe(triple_erode_stages, 1);
	triple_blur_h = make_triple_pipe(triple_blur_stages, 2);

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
	sampler = vk::raii::Sampler(device, sampler_info);
	// Point twin of the above for erode taps (identical output at
	// texel centers, single fetch per tap instead of bilinear).
	vk::SamplerCreateInfo nearest_info = sampler_info;
	nearest_info.magFilter = vk::Filter::eNearest;
	nearest_info.minFilter = vk::Filter::eNearest;
	nearest_info.mipmapMode = vk::SamplerMipmapMode::eNearest;
	sampler_nearest = vk::raii::Sampler(device, nearest_info);

	vk::DescriptorPoolSize pool_size{
	        .type = vk::DescriptorType::eCombinedImageSampler,
	        .descriptorCount = 9,
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .maxSets = 9,
	        .poolSizeCount = 1,
	        .pPoolSizes = &pool_size,
	};
	descriptor_pool = vk::raii::DescriptorPool(device, pool_info);
	for (int i = 0; i < 9; ++i)
	{
		vk::DescriptorSetAllocateInfo set_alloc_info{
		        .descriptorPool = *descriptor_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*descriptor_layout,
		};
		descriptor_sets.emplace_back(std::move(device.allocateDescriptorSets(set_alloc_info)[0]));
	}

	// Depth-volume chain (3d-depth-passthrough, fixed 320px, format-fixed:
	// built once here, never rebuilt). Raster encodes furthest-D as R16
	// (MAX blend over a 0 clear, no MIN cuts); compare flattens to binary
	// R8; merge upscales+MAXes into the tier seed ahead of erode/blur.
	// (Legacy high-res VR path removed: exact-depth soft compare covers it.)
	static_assert(sizeof(volume_mv_push) == 136);
	static_assert(sizeof(volume_mv_push) <= 256);
	static_assert(sizeof(volume_homography) == 144);
	{
		vk::VertexInputBindingDescription volume_binding{
		        .binding = 0,
		        .stride = sizeof(XrVector3f),
		        .inputRate = vk::VertexInputRate::eVertex,
		};
		vk::VertexInputAttributeDescription volume_attribute{
		        .location = 0,
		        .binding = 0,
		        .format = vk::Format::eR32G32B32Sfloat,
		        .offset = 0,
		};
		vk::PipelineVertexInputStateCreateInfo volume_vertex_input{
		        .vertexBindingDescriptionCount = 1,
		        .pVertexBindingDescriptions = &volume_binding,
		        .vertexAttributeDescriptionCount = 1,
		        .pVertexAttributeDescriptions = &volume_attribute,
		};
		vk::PipelineInputAssemblyStateCreateInfo volume_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo volume_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo volume_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        // Back faces encode the far side; MAX blending keeps the
		        // furthest surface order-independently (no depth test).
		        .cullMode = vk::CullModeFlagBits::eFront,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo volume_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		// MAX accumulation (order-independent furthest, UNORM-safe like
		// the reversed MIN path): src/dst factors are ignored by MIN/MAX.
		vk::PipelineColorBlendAttachmentState max_attachment{
		        .blendEnable = VK_TRUE,
		        .srcColorBlendFactor = vk::BlendFactor::eOne,
		        .dstColorBlendFactor = vk::BlendFactor::eOne,
		        .colorBlendOp = vk::BlendOp::eMax,
		        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
		        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
		        .alphaBlendOp = vk::BlendOp::eMax,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo volume_blend{
		        .attachmentCount = 1,
		        .pAttachments = &max_attachment,
		};
		vk::DynamicState volume_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo volume_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = volume_dynamics,
		};
		vk::AttachmentDescription volume_raster_attachment{
		        .format = volume_depth_format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eClear,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eUndefined,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference volume_raster_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription volume_raster_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &volume_raster_ref,
		};
		// Exact-depth raster: nearest state instead of furthest (shared pipe):
		// cull-nothing + MIN over far. Shared attachment/subpass shapes
		// below; only the clear value differs (0 vs 1.0) and that rides
		// the begin info, not the pass.
		vk::PipelineRasterizationStateCreateInfo exact_rasterization = volume_rasterization;
		exact_rasterization.cullMode = vk::CullModeFlagBits::eNone;
		vk::PipelineColorBlendAttachmentState min_attachment{
		        .blendEnable = VK_TRUE,
		        .srcColorBlendFactor = vk::BlendFactor::eOne,
		        .dstColorBlendFactor = vk::BlendFactor::eOne,
		        .colorBlendOp = vk::BlendOp::eMin,
		        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
		        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
		        .alphaBlendOp = vk::BlendOp::eMin,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo exact_blend{
		        .attachmentCount = 1,
		        .pAttachments = &min_attachment,
		};
		// Multiview volume/VR raster: same R16 CLEAR/STORE/OPTIMAL attachment plus
		// viewMask=both, volume_mv.vert (MVP pair) + matching fragment
		// twin (push blocks must agree byte-for-byte — mask_mv precedent).
		if (physical_device.getProperties().limits.maxPushConstantsSize < sizeof(volume_mv_push))
			throw std::runtime_error("feather volume multiview needs 136B push constants");
		{
			vk::PushConstantRange mv_push_range{
			        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			        .offset = 0,
			        .size = sizeof(volume_mv_push),
			};
			vk::PipelineLayoutCreateInfo mv_layout_info{
			        .pushConstantRangeCount = 1,
			        .pPushConstantRanges = &mv_push_range,
			};
			volume_raster_layout_mv = vk::raii::PipelineLayout(device, mv_layout_info);
		}
		auto volume_mv_vert = load_shader(device, "volume_mv.vert");
		auto volume_mv_frag = load_shader(device, "volume_raster_mv.frag");
		vk::PipelineShaderStageCreateInfo volume_mv_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **volume_mv_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **volume_mv_frag,
		                .pName = "main",
		        },
		};
		uint32_t vol_mv_mask = 0b11, vol_mv_corr = 0b11;
		vk::RenderPassMultiviewCreateInfo vol_mv_info{
		        .subpassCount = 1,
		        .pViewMasks = &vol_mv_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &vol_mv_corr,
		};
		volume_raster_renderpass_mv = vk::raii::RenderPass(device, vk::RenderPassCreateInfo{
		        .pNext = &vol_mv_info,
		        .attachmentCount = 1,
		        .pAttachments = &volume_raster_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &volume_raster_subpass,
		});
		vk::GraphicsPipelineCreateInfo volume_mv_pipeline_info{
		        .stageCount = 2,
		        .pStages = volume_mv_stages,
		        .pVertexInputState = &volume_vertex_input,
		        .pInputAssemblyState = &volume_input_assembly,
		        .pViewportState = &volume_viewport_state,
		        .pRasterizationState = &volume_rasterization,
		        .pMultisampleState = &volume_multisample,
		        .pColorBlendState = &volume_blend,
		        .pDynamicState = &volume_dynamic,
		        .layout = *volume_raster_layout_mv,
		        .renderPass = *volume_raster_renderpass_mv,
		};
		volume_raster_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), volume_mv_pipeline_info);
		vk::GraphicsPipelineCreateInfo exact_mv_raster_info{
		        .stageCount = 2,
		        .pStages = volume_mv_stages,
		        .pVertexInputState = &volume_vertex_input,
		        .pInputAssemblyState = &volume_input_assembly,
		        .pViewportState = &volume_viewport_state,
		        .pRasterizationState = &exact_rasterization,
		        .pMultisampleState = &volume_multisample,
		        .pColorBlendState = &exact_blend,
		        .pDynamicState = &volume_dynamic,
		        .layout = *volume_raster_layout_mv,
		        .renderPass = *volume_raster_renderpass_mv,
		};
		exact_raster_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), exact_mv_raster_info);
	}
	// Compare: fullscreen _mv binary flatten (sensor D16 + volume R16).
	// Renders into volume_bin.blur_fb_mv under the shared R8 MV pass
	// (identical attachment shape); only the fragment shader + set
	// layout (2 samplers) are new.
	{
		auto compare_frag = load_shader(device, "soft_occlude_mv.frag");
		vk::PipelineShaderStageCreateInfo compare_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **blur_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **compare_frag,
		                .pName = "main",
		        },
		};
		vk::DescriptorSetLayoutBinding compare_bindings[4] = {
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
		        },
		        {
		                // Virtual-depth D canvas (same shape as binding 1).
		                // Unpaired frames rebind the volume canvas here.
		                .binding = 2,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		                {
		                        // Exact-depth D canvas (passthrough + hidden nearest,
		                        // reversed excluded). Empty frames rebind the volume canvas.
		                        .binding = 3,
		                        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                        .descriptorCount = 1,
		                        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		                },
		};
		vk::DescriptorSetLayoutCreateInfo compare_set_info{
		        .bindingCount = 4,
		        .pBindings = compare_bindings,
		};
		// 144B soft-occlusion push (H + B2 corr + bias/near/far) exceeds the
		// 128-byte Vulkan minimum: Adreno 740 (Quest 3) provides 256B; throw
		// loudly on smaller limits like the silhouette/volume paths above.
		if (physical_device.getProperties().limits.maxPushConstantsSize < sizeof(volume_homography))
			throw std::runtime_error("feather soft compare needs 144B push constants");
		compare_set_layout = vk::raii::DescriptorSetLayout(device, compare_set_info);
		vk::PushConstantRange compare_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(volume_homography),
		};
		vk::PipelineLayoutCreateInfo compare_pipe_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*compare_set_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &compare_push_range,
		};
		compare_pipe_layout = vk::raii::PipelineLayout(device, compare_pipe_layout_info);
		vk::DescriptorPoolSize compare_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 5,
		};
		vk::DescriptorPoolCreateInfo compare_pool_info{
		        .maxSets = 1,
		        .poolSizeCount = 1,
		        .pPoolSizes = &compare_pool_size,
		};
		compare_pool = vk::raii::DescriptorPool(device, compare_pool_info);
		vk::DescriptorSetAllocateInfo compare_alloc_info{
		        .descriptorPool = *compare_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*compare_set_layout,
		};
		compare_set = std::move(device.allocateDescriptorSets(compare_alloc_info)[0]);
		vk::PipelineVertexInputStateCreateInfo compare_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo compare_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo compare_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo compare_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo compare_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState compare_blend_attachment{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo compare_blend{
		        .attachmentCount = 1,
		        .pAttachments = &compare_blend_attachment,
		};
		vk::DynamicState compare_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo compare_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = compare_dynamics,
		};
		vk::GraphicsPipelineCreateInfo compare_pipeline_info{
			        .stageCount = 2,
			        .pStages = compare_stages,
			        .pVertexInputState = &compare_vertex_input,
			        .pInputAssemblyState = &compare_input_assembly,
			        .pViewportState = &compare_viewport_state,
		                .pRasterizationState = &compare_rasterization,
		        .pMultisampleState = &compare_multisample,
		        .pColorBlendState = &compare_blend,
		        .pDynamicState = &compare_dynamic,
		        .layout = *compare_pipe_layout,
		        .renderPass = *compare_renderpass_mv,
		};
			compare_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), compare_pipeline_info);
	// VR-cut compare: fullscreen _mv eye-space test (virtual D vs exact
	// nearest D, both sensor-compatible). Renders keep=1 into
	// cut_bin.blur_fb_mv under the shared R8 MV CLEAR pass. No push
	// constants: both inputs are already D, no reprojection.
	{
		auto cut_frag = load_shader(device, "vr_cut_mv.frag");
		vk::PipelineShaderStageCreateInfo cut_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **blur_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **cut_frag,
		                .pName = "main",
		        },
		};
		vk::DescriptorSetLayoutBinding cut_bindings[2] = {
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
		        },
		};
		vk::DescriptorSetLayoutCreateInfo cut_set_info{
		        .bindingCount = 2,
		        .pBindings = cut_bindings,
		};
		cut_set_layout = vk::raii::DescriptorSetLayout(device, cut_set_info);
		vk::PipelineLayoutCreateInfo cut_pipe_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*cut_set_layout,
		};
		cut_pipe_layout = vk::raii::PipelineLayout(device, cut_pipe_layout_info);
		vk::DescriptorPoolSize cut_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 3,
		};
		vk::DescriptorPoolCreateInfo cut_pool_info{
		        .maxSets = 1,
		        .poolSizeCount = 1,
		        .pPoolSizes = &cut_pool_size,
		};
		cut_pool = vk::raii::DescriptorPool(device, cut_pool_info);
		vk::DescriptorSetAllocateInfo cut_alloc_info{
		        .descriptorPool = *cut_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*cut_set_layout,
		};
		cut_set = std::move(device.allocateDescriptorSets(cut_alloc_info)[0]);
		vk::PipelineVertexInputStateCreateInfo cut_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo cut_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo cut_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo cut_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo cut_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState cut_blend_attachment{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo cut_blend{
		        .attachmentCount = 1,
		        .pAttachments = &cut_blend_attachment,
		};
		vk::DynamicState cut_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo cut_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = cut_dynamics,
		};
		vk::GraphicsPipelineCreateInfo cut_pipeline_info{
		        .stageCount = 2,
		        .pStages = cut_stages,
		        .pVertexInputState = &cut_vertex_input,
		        .pInputAssemblyState = &cut_input_assembly,
		        .pViewportState = &cut_viewport_state,
		        .pRasterizationState = &cut_rasterization,
		        .pMultisampleState = &cut_multisample,
		        .pColorBlendState = &cut_blend,
		        .pDynamicState = &cut_dynamic,
		        .layout = *cut_pipe_layout,
		        .renderPass = *compare_renderpass_mv,
		};
		cut_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), cut_pipeline_info);
	}
	}
	// Virtual depth SBS -> sensor-D canvas: fullscreen _mv conversion
	// (meters->D with the volume near/far), so the compare stays in D
	// space with untouched push constants. Renders into
	// volume_virt_fb_mv under its own CLEAR R16 MV pass (the
	// compare pass is R8: format-incompatible). One draw per volume
	// group, only when paired.
	{
		auto virt_frag = load_shader(device, "volume_virt_mv.frag");
		vk::PipelineShaderStageCreateInfo virt_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **blur_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **virt_frag,
		                .pName = "main",
		        },
		};
		vk::DescriptorSetLayoutBinding virt_binding{
		        .binding = 0,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 1,
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		};
		vk::DescriptorSetLayoutCreateInfo virt_set_info{
		        .bindingCount = 1,
		        .pBindings = &virt_binding,
		};
		volume_virt_set_layout = vk::raii::DescriptorSetLayout(device, virt_set_info);
		vk::PushConstantRange virt_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(volume_virt_push),
		};
		vk::PipelineLayoutCreateInfo virt_pipe_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*volume_virt_set_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &virt_push_range,
		};
		volume_virt_layout = vk::raii::PipelineLayout(device, virt_pipe_layout_info);
		vk::DescriptorPoolSize virt_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 1,
		};
		vk::DescriptorPoolCreateInfo virt_pool_info{
		        .maxSets = 1,
		        .poolSizeCount = 1,
		        .pPoolSizes = &virt_pool_size,
		};
		volume_virt_pool = vk::raii::DescriptorPool(device, virt_pool_info);
		vk::DescriptorSetAllocateInfo virt_alloc_info{
		        .descriptorPool = *volume_virt_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*volume_virt_set_layout,
		};
		volume_virt_set = std::move(device.allocateDescriptorSets(virt_alloc_info)[0]);
		uint32_t mv_view_mask = 0b11, mv_corr_mask = 0b11;
		vk::RenderPassMultiviewCreateInfo virt_mv_rp_info{
		        .subpassCount = 1,
		        .pViewMasks = &mv_view_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &mv_corr_mask,
		};
		vk::AttachmentDescription virt_attachment{
		        .format = volume_depth_format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eClear,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eUndefined,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference virt_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription virt_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &virt_ref,
		};
		vk::RenderPassCreateInfo virt_rp_info{
		        .pNext = &virt_mv_rp_info,
		        .attachmentCount = 1,
		        .pAttachments = &virt_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &virt_subpass,
		};
		volume_virt_renderpass = vk::raii::RenderPass(device, virt_rp_info);
		vk::PipelineVertexInputStateCreateInfo virt_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo virt_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo virt_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo virt_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo virt_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState virt_blend_attachment{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo virt_blend{
		        .attachmentCount = 1,
		        .pAttachments = &virt_blend_attachment,
		};
		vk::DynamicState virt_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo virt_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = virt_dynamics,
		};
		vk::GraphicsPipelineCreateInfo virt_pipeline_info{
		        .stageCount = 2,
		        .pStages = virt_stages,
		        .pVertexInputState = &virt_vertex_input,
		        .pInputAssemblyState = &virt_input_assembly,
		        .pViewportState = &virt_viewport_state,
		        .pRasterizationState = &virt_rasterization,
		        .pMultisampleState = &virt_multisample,
		        .pColorBlendState = &virt_blend,
		        .pDynamicState = &virt_dynamic,
		        .layout = *volume_virt_layout,
		        .renderPass = *volume_virt_renderpass,
		};
		volume_virt_pipeline = vk::raii::Pipeline(device, *pipeline_cache(device), virt_pipeline_info);
	}
	// Homography rides push constants (named vec4 members, see header);
	// no buffers, views, copies or barriers per frame.
	// Linear sampler for the compare taps (sensor + volume depth),
	// same filtering as the blur passes: the bilinear ramp is the
	// outline pre-smoothing (see header), not a correctness hazard.
	{
		vk::SamplerCreateInfo volume_linear_info{
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
		volume_sampler_linear = vk::raii::Sampler(device, volume_linear_info);
	}
	// Merge: upscale the 320px binary canvas into the tier seed with MAX
	// blending (LOAD preserves silhouettes + reversed cuts, 1s set where
	// the sensor is inside). Reuses the blur shaders with spread 0 (the
	// linear sampler is the upscale) under the shared blur layout; the
	// seed's own MV framebuffer is reused (loadOp is not part of render
	// pass compatibility). Two renderpasses: R8 intermediates (tier 1 +
	// tiered seeds, COLOR_OPTIMAL) and RGBA8 submitted images (tier 0,
	// COLOR_OPTIMAL).
	{
		uint32_t mv_view_mask = 0b11, mv_corr_mask = 0b11;
		vk::RenderPassMultiviewCreateInfo volume_mv_rp_info{
		        .subpassCount = 1,
		        .pViewMasks = &mv_view_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &mv_corr_mask,
		};
		vk::AttachmentDescription max_r8_attachment{
		        .format = blur_format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eLoad,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference max_r8_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription max_r8_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &max_r8_ref,
		};
		vk::RenderPassCreateInfo max_r8_rp_info{
		        .pNext = &volume_mv_rp_info,
		        .attachmentCount = 1,
		        .pAttachments = &max_r8_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &max_r8_subpass,
		};
		volume_max_r8_renderpass = vk::raii::RenderPass(device, max_r8_rp_info);
		vk::AttachmentDescription max_rgb8_attachment{
		        .format = format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eLoad,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference max_rgb8_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription max_rgb8_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &max_rgb8_ref,
		};
		vk::RenderPassCreateInfo max_rgb8_rp_info{
		        .pNext = &volume_mv_rp_info,
		        .attachmentCount = 1,
		        .pAttachments = &max_rgb8_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &max_rgb8_subpass,
		};
		volume_max_rgb8_renderpass = vk::raii::RenderPass(device, max_rgb8_rp_info);
		vk::PipelineColorBlendAttachmentState volume_max_attachment{
		        .blendEnable = VK_TRUE,
		        .srcColorBlendFactor = vk::BlendFactor::eOne,
		        .dstColorBlendFactor = vk::BlendFactor::eOne,
		        .colorBlendOp = vk::BlendOp::eMax,
		        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
		        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
		        .alphaBlendOp = vk::BlendOp::eMax,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo volume_max_blend{
		        .attachmentCount = 1,
		        .pAttachments = &volume_max_attachment,
		};
		auto make_max_pipe = [&](vk::raii::RenderPass & rp) {
			vk::GraphicsPipelineCreateInfo max_pipe_info{
			        .stageCount = 2,
			        .pStages = blur_mv_stages,
			        .pVertexInputState = &blur_vertex_input,
			        .pInputAssemblyState = &blur_input_assembly,
			        .pViewportState = &blur_viewport_state,
			        .pRasterizationState = &blur_rasterization,
			        .pMultisampleState = &blur_multisample,
			        .pColorBlendState = &volume_max_blend,
			        .pDynamicState = &blur_dynamic,
			        .layout = *blur_layout,
			        .renderPass = *rp,
			};
			return vk::raii::Pipeline(device, *pipeline_cache(device), max_pipe_info);
		};
		volume_max_r8_pipeline = make_max_pipe(volume_max_r8_renderpass);
		volume_max_rgb8_pipeline = make_max_pipe(volume_max_rgb8_renderpass);
		// VR-cut MIN merges: same blur shaders/layout/renderpasses as the
		// MAX merges above, MIN blend instead. LOAD preserves the seed,
		// 0s cut where VR wins over known geometry (and hands).
		vk::PipelineColorBlendAttachmentState cut_min_attachment = volume_max_attachment;
		cut_min_attachment.colorBlendOp = vk::BlendOp::eMin;
		cut_min_attachment.alphaBlendOp = vk::BlendOp::eMin;
		vk::PipelineColorBlendStateCreateInfo cut_min_blend{
		        .attachmentCount = 1,
		        .pAttachments = &cut_min_attachment,
		};
		auto make_cut_min_pipe = [&](vk::raii::RenderPass & rp) {
		        vk::GraphicsPipelineCreateInfo cut_min_pipe_info{
		                .stageCount = 2,
		                .pStages = blur_mv_stages,
		                .pVertexInputState = &blur_vertex_input,
		                .pInputAssemblyState = &blur_input_assembly,
		                .pViewportState = &blur_viewport_state,
		                .pRasterizationState = &blur_rasterization,
		                .pMultisampleState = &blur_multisample,
		                .pColorBlendState = &cut_min_blend,
		                .pDynamicState = &blur_dynamic,
		                .layout = *blur_layout,
		                .renderPass = *rp,
		        };
		        return vk::raii::Pipeline(device, *pipeline_cache(device), cut_min_pipe_info);
		};
		cut_min_r8_pipeline = make_cut_min_pipe(volume_max_r8_renderpass);
		cut_min_rgb8_pipeline = make_cut_min_pipe(volume_max_rgb8_renderpass);
	}
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> feather_mask_renderer::make_buffer(vk::DeviceSize size,
                                                                                       vk::BufferUsageFlags usage,
                                                                                       vk::MemoryPropertyFlags properties)
{
	vk::BufferCreateInfo buffer_info{
	        .size = size,
	        .usage = usage,
	        .sharingMode = vk::SharingMode::eExclusive,
	};
	vk::raii::Buffer buffer(device, buffer_info);
	auto requirements = buffer.getMemoryRequirements();
	vk::MemoryAllocateInfo alloc_info{
	        .allocationSize = requirements.size,
	        .memoryTypeIndex = find_memory_type(physical_device, requirements.memoryTypeBits, properties),
	};
	vk::raii::DeviceMemory memory(device, alloc_info);
	buffer.bindMemory(*memory, 0);
	return {std::move(buffer), std::move(memory)};
}

void feather_mask_renderer::set_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup)
{
	if (soup.indices.empty() or soup.vertices.empty())
	{
		remove_mesh(key);
		return;
	}

	// Rare path (map change): make sure no in-flight frame still reads the
	// old buffers before replacing them.
	device.waitIdle();

	mesh_buffers & mesh = meshes[key];
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
	// Model-space bounds for the stream-side frustum skip (upload-time
	// only; per-frame tests transform the 8 corners by the draw MVP).
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

std::optional<std::pair<glm::vec3, glm::vec3>> feather_mask_renderer::mesh_bounds(const std::string & key) const
{
	auto it = meshes.find(key);
	if (it == meshes.end())
		return std::nullopt;
	return std::make_pair(it->second.aabb_min, it->second.aabb_max);
}

void feather_mask_renderer::remove_mesh(const std::string & key)
{
	auto it = meshes.find(key);
	if (it == meshes.end())
		return;
	// Same in-flight hazard as replace.
	device.waitIdle();
	meshes.erase(it);
}

// In-place vertex rewrite for dynamic meshes (skinned hands): same
// vertex count as the uploaded soup, so every buffer is reused and no
// waitIdle is needed. Staging is host-visible + coherent (same pattern
// as the cutout quads); the pending copy is flushed by the normal
// per-draw flush_upload path with a transfer->vertex barrier.
void feather_mask_renderer::update_mesh_positions(const std::string & key, const std::vector<XrVector3f> & verts)
{
	auto it = meshes.find(key);
	if (it == meshes.end())
		return;
	auto & mesh = it->second;
	vk::DeviceSize vertex_bytes = verts.size() * sizeof(XrVector3f);
	vk::DeviceSize index_bytes = mesh.index_count * sizeof(uint32_t);
	if (mesh.staging_size < vertex_bytes + index_bytes)
		return; // size changed: caller falls back to set_mesh
	if (vertex_bytes == 0)
		return;
	void * mapped = mesh.staging_memory.mapMemory(0, vertex_bytes + index_bytes);
	std::memcpy(mapped, verts.data(), vertex_bytes);
	mesh.staging_memory.unmapMemory();
	mesh.pending_upload = {vertex_bytes, index_bytes};
	glm::vec3 lo{verts[0].x, verts[0].y, verts[0].z};
	glm::vec3 hi = lo;
	for (const auto & v: verts)
	{
		lo = glm::min(lo, glm::vec3(v.x, v.y, v.z));
		hi = glm::max(hi, glm::vec3(v.x, v.y, v.z));
	}
	mesh.aabb_min = lo;
	mesh.aabb_max = hi;
}

vk::ImageView feather_mask_renderer::submitted_view_array(vk::Image image) const
{
	// The array view is built unconditionally with every targets
	// entry (and every hidden entry), so map presence alone proves
	// the view is live.
	if (auto it = targets.find(image); it != targets.end())
		return *it->second.view_array;
	if (auto ht = hidden_targets.find(image); ht != hidden_targets.end())
		return *ht->second.view_array;
	return vk::ImageView{};
}

feather_mask_renderer::depth_entry feather_mask_renderer::make_depth_entry(vk::Extent2D e)
{
	depth_entry entry;
	vk::ImageCreateInfo image_info{
	        .imageType = vk::ImageType::e2D,
	        .format = depth_format,
	        .extent = {e.width, e.height, 1},
	        .mipLevels = 1,
	        .arrayLayers = 2,
	        .samples = vk::SampleCountFlagBits::e1,
	        .tiling = vk::ImageTiling::eOptimal,
	        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
	        .sharingMode = vk::SharingMode::eExclusive,
	        .initialLayout = vk::ImageLayout::eUndefined,
	};
	entry.image = vk::raii::Image(device, image_info);
	auto requirements = entry.image.getMemoryRequirements();
	vk::MemoryAllocateInfo alloc_info{
	        .allocationSize = requirements.size,
	        .memoryTypeIndex = find_memory_type(physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
	};
	entry.memory = vk::raii::DeviceMemory(device, alloc_info);
	entry.image.bindMemory(*entry.memory, 0);
	return entry;
}

vk::ImageView feather_mask_renderer::output_depth_array_view(vk::Extent2D extent)
{
	auto key = std::make_pair(extent.width, extent.height);
	auto it = output_depth.find(key);
	if (it == output_depth.end())
	{
		spdlog::info("Feather mask depth target: {}x{}", extent.width, extent.height);
		it = output_depth.emplace(key, make_depth_entry(extent)).first;
	}
	if (*it->second.view_array == VK_NULL_HANDLE)
	{
		it->second.view_array = vk::raii::ImageView(device,
		        vk::ImageViewCreateInfo{
		                .image = *it->second.image,
		                .viewType = vk::ImageViewType::e2DArray,
		                .format = depth_format,
		                .subresourceRange = {
		                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
		                        .baseMipLevel = 0,
		                        .levelCount = 1,
		                        .baseArrayLayer = 0,
		                        .layerCount = 2,
		                },
		        });
	}
	return *it->second.view_array;
}

// Feather-owned hidden proximity field (P4: one proximity raster per frame).
// (Re)created here in the waitIdle'd scene prewarm on extent change;
// the mask record below renders into it.
bool feather_mask_renderer::ensure_hidden_field(vk::Extent2D extent)
{
	if (hidden_sphere_extent_.width == extent.width and hidden_sphere_extent_.height == extent.height and
	    hidden_sphere_extent_.width != 0)
		return false;
	// Rare path (first hidden frame, video resize): nothing outstanding
	// (acquire + release pair within one frame, always paired below).
	device.waitIdle();
	hidden_sphere_depth_ = image_allocation{
	        device,
	        vk::ImageCreateInfo{
	                .imageType = vk::ImageType::e2D,
	                .format = depth_format,
	                .extent = {extent.width, extent.height, 1},
	                .mipLevels = 1,
	                .arrayLayers = 2,
	                .samples = vk::SampleCountFlagBits::e1,
	                .tiling = vk::ImageTiling::eOptimal,
	                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
	                .sharingMode = vk::SharingMode::eExclusive,
	                .initialLayout = vk::ImageLayout::eUndefined,
	        },
	        VmaAllocationCreateInfo{
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "hidden proximity sphere depth",
	};
	hidden_sphere_view_array_ = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
	        .image = (VkImage)hidden_sphere_depth_,
	        .viewType = vk::ImageViewType::e2DArray,
	        .format = depth_format,
	        .subresourceRange = {
	                .aspectMask = vk::ImageAspectFlagBits::eDepth,
	                .baseMipLevel = 0,
	                .levelCount = 1,
	                .baseArrayLayer = 0,
	                .layerCount = 2,
	        },
	});
	hidden_sphere_extent_ = extent;
	spdlog::info("Hidden proximity field: {}x{}", extent.width, extent.height);
	return true;
}

void feather_mask_renderer::ensure_hidden(vk::Image image, vk::Extent2D extent)
{
	assert(hidden_sphere_extent_.width == extent.width and hidden_sphere_extent_.height == extent.height);

	if (not hidden_ready)
	{
		// Multiview silhouette pushes (132B MVP pair) exceed the 128-byte
		// Vulkan minimum: Adreno 740 (Quest 3) provides 256B. Throw loudly
		// here — not silently on device — when unsupported (see the
		// silhouette_mv_push note above the ctor).
		if (physical_device.getProperties().limits.maxPushConstantsSize < sizeof(silhouette_mv_push))
			throw std::runtime_error("hidden multiview needs 132B push constants");
		hidden_ready = true;
		// Unit-sphere verts (shared proximity-depth generator).
		auto sphere = passthrough_mesh::unit_sphere_positions();
		hidden_sphere_vert_count = (uint32_t)sphere.size();
		hidden_sphere_verts_ = buffer_allocation{
		        device,
		        vk::BufferCreateInfo{
		                .size = sphere.size() * sizeof(XrVector3f),
		                .usage = vk::BufferUsageFlagBits::eVertexBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "hidden proximity sphere",
		};
		std::memcpy(hidden_sphere_verts_.map(), sphere.data(), sphere.size() * sizeof(XrVector3f));
		hidden_sphere_verts_.unmap();
		// Unit-cylinder verts (body column, same depth-only pipe).
		auto cylinder = passthrough_mesh::unit_cylinder_positions();
		hidden_cylinder_vert_count = (uint32_t)cylinder.size();
		hidden_cylinder_verts_ = buffer_allocation{
		        device,
		        vk::BufferCreateInfo{
		                .size = cylinder.size() * sizeof(XrVector3f),
		                .usage = vk::BufferUsageFlagBits::eVertexBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "hidden proximity cylinder",
		};
		std::memcpy(hidden_cylinder_verts_.map(), cylinder.data(), cylinder.size() * sizeof(XrVector3f));
		hidden_cylinder_verts_.unmap();

		// Sphere depth-only RP: CLEAR near (1.0) / STORE, ends in the
		// attachment layout for the silhouette pass below.
		vk::AttachmentDescription sphere_att{
		        .format = depth_format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eClear,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
		        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		        .initialLayout = vk::ImageLayout::eUndefined,
		        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::AttachmentReference sphere_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::SubpassDescription sphere_sub{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .pDepthStencilAttachment = &sphere_ref,
		};
		uint32_t hmv_view_mask = 0b11, hmv_corr_mask = 0b11;
		vk::RenderPassMultiviewCreateInfo hmv_rp_info{
		        .subpassCount = 1,
		        .pViewMasks = &hmv_view_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &hmv_corr_mask,
		};
		hidden_sphere_rp = vk::raii::RenderPass(device, vk::RenderPassCreateInfo{
		                                                          .pNext = &hmv_rp_info,
		                                                          .attachmentCount = 1,
		                                                          .pAttachments = &sphere_att,
		                                                          .subpassCount = 1,
		                                                          .pSubpasses = &sphere_sub,
		                                                  });
		// Silhouette RP: color CLEAR/STORE/OPTIMAL (submitted image, nothing
		// app-side re-reads it — keeps UBWC live) + proximity depth LOAD
		// (written by the pass above), DONT_CARE store.
		vk::AttachmentDescription hidden_atts[2] = {
		        vk::AttachmentDescription{
		                .format = format,
		                .samples = vk::SampleCountFlagBits::e1,
		                .loadOp = vk::AttachmentLoadOp::eClear,
		                .storeOp = vk::AttachmentStoreOp::eStore,
		                .initialLayout = vk::ImageLayout::eUndefined,
		                .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        },
		        vk::AttachmentDescription{
		                .format = depth_format,
		                .samples = vk::SampleCountFlagBits::e1,
		                .loadOp = vk::AttachmentLoadOp::eLoad,
		                .storeOp = vk::AttachmentStoreOp::eDontCare,
		                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
		                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		                .initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		                .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		        },
		};
		vk::AttachmentReference hidden_color_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference hidden_depth_ref{
		        .attachment = 1,
		        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::SubpassDescription hidden_sub{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &hidden_color_ref,
		        .pDepthStencilAttachment = &hidden_depth_ref,
		};
		hidden_color_rp = vk::raii::RenderPass(device, vk::RenderPassCreateInfo{
		                                                         .pNext = &hmv_rp_info,
		                                                         .attachmentCount = 2,
		                                                         .pAttachments = hidden_atts,
		                                                         .subpassCount = 1,
		                                                         .pSubpasses = &hidden_sub,
		                                                 });

		// Multiview twins of the mask modules (mask_mv.vert/frag, 132B
		// silhouette_mv_layout): same raster state as the tier-0 raster plus
		// the proximity depth test.
		auto hvert = load_shader(device, "mask_mv.vert");
		auto hfrag = load_shader(device, "mask_mv.frag");
		// Empty fragment: the proximity pipe is depth-only (no color
		// attachment, color writes off).
		auto hempty = load_shader(device, "empty.frag");
		vk::PipelineShaderStageCreateInfo hstages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **hvert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **hfrag,
		                .pName = "main",
		        },
		};
		vk::PipelineShaderStageCreateInfo hfill_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **hvert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **hempty,
		                .pName = "main",
		        },
		};
		// vec3 positions, shared by mesh verts and proximity verts.
		vk::VertexInputBindingDescription hbinding{
		        .binding = 0,
		        .stride = sizeof(float) * 3,
		        .inputRate = vk::VertexInputRate::eVertex,
		};
		vk::VertexInputAttributeDescription hattribute{
		        .location = 0,
		        .binding = 0,
		        .format = vk::Format::eR32G32B32Sfloat,
		        .offset = 0,
		};
		vk::PipelineVertexInputStateCreateInfo hvertex_input{
		        .vertexBindingDescriptionCount = 1,
		        .pVertexBindingDescriptions = &hbinding,
		        .vertexAttributeDescriptionCount = 1,
		        .pVertexAttributeDescriptions = &hattribute,
		};
		vk::PipelineInputAssemblyStateCreateInfo hinput_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo hviewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineMultisampleStateCreateInfo hmultisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::DynamicState hdynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo hdynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = hdynamics,
		};
		// Sphere fill: cull-front keeps the inside/exit wall from any
		// viewpoint; LESS min-accumulates the furthest exit across
		// spheres (sound over-approximation — see record_hidden).
		vk::PipelineRasterizationStateCreateInfo hsphere_raster{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eFront,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineDepthStencilStateCreateInfo hsphere_depth{
		        .depthTestEnable = VK_TRUE,
		        .depthWriteEnable = VK_TRUE,
		        .depthCompareOp = vk::CompareOp::eLess,
		};
		// Reuse the ctor-built silhouette MV layout (identical 132B
		// silhouette_mv_push range): one layout for all MV mask pipes.
		hidden_sphere_pipe = vk::raii::Pipeline(device, *pipe_cache,
		                                        vk::GraphicsPipelineCreateInfo{
		                                                .stageCount = 2,
		                                                .pStages = hfill_stages,
		                                                .pVertexInputState = &hvertex_input,
		                                                .pInputAssemblyState = &hinput_assembly,
		                                                .pViewportState = &hviewport_state,
		                                                .pRasterizationState = &hsphere_raster,
		                                                .pMultisampleState = &hmultisample,
		                                                .pDepthStencilState = &hsphere_depth,
		                                                .pColorBlendState = nullptr,
		                                                .pDynamicState = &hdynamic,
		                                                .layout = *silhouette_mv_layout,
		                                                .renderPass = *hidden_sphere_rp,
		                                        });
		// Silhouette: tier-0 raster state verbatim (cull-none, blend-off
		// overwrite) plus the proximity test (GREATER, no depth write).
		vk::PipelineRasterizationStateCreateInfo hsil_raster{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineDepthStencilStateCreateInfo hsil_depth{
		        .depthTestEnable = VK_TRUE,
		        .depthWriteEnable = VK_FALSE,
		        .depthCompareOp = vk::CompareOp::eGreater,
		};
		vk::PipelineColorBlendAttachmentState hsil_blend_att{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo hsil_blend{
		        .attachmentCount = 1,
		        .pAttachments = &hsil_blend_att,
		};
		hidden_silhouette_pipe = vk::raii::Pipeline(device, *pipe_cache,
		                                            vk::GraphicsPipelineCreateInfo{
		                                                    .stageCount = 2,
		                                                    .pStages = hstages,
		                                                    .pVertexInputState = &hvertex_input,
		                                                    .pInputAssemblyState = &hinput_assembly,
		                                                    .pViewportState = &hviewport_state,
		                                                    .pRasterizationState = &hsil_raster,
		                                                    .pMultisampleState = &hmultisample,
		                                                    .pDepthStencilState = &hsil_depth,
		                                                    .pColorBlendState = &hsil_blend,
		                                                    .pDynamicState = &hdynamic,
		                                                    .layout = *silhouette_mv_layout,
		                                                    .renderPass = *hidden_color_rp,
		                                            });
	}

	auto tit = hidden_targets.find(image);
	if (tit == hidden_targets.end() or tit->second.extent.width != (int32_t)extent.width or
	    tit->second.extent.height != (int32_t)extent.height)
	{
		hidden_frame_targets tgt;
		tgt.extent = extent;
		// One multiview target: array color view (both layers) + layers=2
		// framebuffers; one draw broadcasts to both eyes. The proximity
		// framebuffer bakes the feather-owned proximity field ARRAY view
		// (see ensure_hidden_field); reset_targets() on field rebuild.
		// Extent agreement is the scene's job — one quarter-res for
		// mask and proximity field.
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
		tgt.view_array = vk::raii::ImageView(device, iv_info);
		vk::ImageView color_view = *tgt.view_array;
		vk::ImageView sphere_view = *hidden_sphere_view_array_;
		vk::ImageView color_atts[2] = {color_view, sphere_view};
		vk::FramebufferCreateInfo color_fb{
		        .renderPass = *hidden_color_rp,
		        .attachmentCount = 2,
		        .pAttachments = color_atts,
		        .width = extent.width,
		        .height = extent.height,
		        .layers = 2,
		};
		tgt.framebuffer = vk::raii::Framebuffer(device, color_fb);
		vk::FramebufferCreateInfo sphere_fb{
		        .renderPass = *hidden_sphere_rp,
		        .attachmentCount = 1,
		        .pAttachments = &sphere_view,
		        .width = extent.width,
		        .height = extent.height,
		        .layers = 2,
		};
		tgt.sphere_framebuffer = vk::raii::Framebuffer(device, sphere_fb);
		hidden_targets[image] = std::move(tgt);
	}
}

void feather_mask_renderer::record_hidden(vk::raii::CommandBuffer & cmd,
                                          vk::Image image,
                                          vk::Extent2D extent,
                                          const std::vector<hidden_draw> & draws,
                                          const std::vector<std::array<glm::mat4, 2>> & spheres,
                                          const std::vector<std::array<glm::mat4, 2>> & cylinders,
                                          hidden_cpu * cpu_stats,
                                          const std::array<vk::Extent2D, 2> & full_extents)
{
	hidden_cpu local{};
	section_clock raster_clk(&local.raster_ms);

	for (auto & [key, mesh]: meshes)
		flush_upload(cmd, mesh);
	ensure_hidden(image, extent);
	auto & tgt = hidden_targets[image];
	VkImage sphere_image = (VkImage)hidden_sphere_depth_;

	// Multiview: one draw broadcasts to both layers (viewMask=both) — the
	// two per-eye passes collapse into two begins total. Push carries both
	// MVPs; each invocation selects its own eye (exact, no cross-eye
	// leakage). Scissors are the both-eyes union (one scissor covers both
	// views): superset-safe, skip only when both eyes degenerate.
	vk::Viewport full_vp{
	        .x = 0,
	        .y = 0,
	        .width = (float)extent.width,
	        .height = (float)extent.height,
	        .minDepth = 0,
	        .maxDepth = 1,
	};
	vk::Rect2D full_sc{.offset = {0, 0}, .extent = extent};
	// Both-eyes union of the containing-cube boxes for a proximity MVP
	// pair (unit sphere or unit cylinder — both are inscribed in
	// [-1,1]^3, so the cube corners stay a rigorous superset either
	// way). Per eye: rigorous superset while every corner is in front
	// (+2px pad, float-domain clamp — see the UB note below); any corner
	// at/behind the camera forces fullscreen for that eye (the eye-inside
	// case always has one; a clipped box there would under-reveal,
	// fail-unsafe).
	// Returns false only when both eyes degenerate (fully off-screen).
	auto field_union = [&](const std::array<glm::mat4, 2> & s, vk::Rect2D & out) {
		bool got = false;
		int32_t ax0 = 0, ay0 = 0, ax1 = 0, ay1 = 0;
		for (int eye = 0; eye < 2; ++eye)
		{
			bool eye_full = false;
			float sx0 = 0, sy0 = 0, sx1 = 0, sy1 = 0;
			bool eye_any = false;
			for (int corner = 0; corner < 8 and not eye_full; ++corner)
			{
				glm::vec4 q4((corner & 1) ? 1.f : -1.f, (corner & 2) ? 1.f : -1.f,
				               (corner & 4) ? 1.f : -1.f, 1);
				glm::vec4 c = s[eye] * q4;
				if (c.w <= 1e-6f)
				{
					eye_full = true;
					break;
				}
				float px = (c.x / c.w * 0.5f + 0.5f) * (float)extent.width;
				float py = (c.y / c.w * 0.5f + 0.5f) * (float)extent.height;
				if (not eye_any)
				{
					sx0 = sx1 = px;
					sy0 = sy1 = py;
					eye_any = true;
				}
				else
				{
					sx0 = std::min(sx0, px);
					sx1 = std::max(sx1, px);
					sy0 = std::min(sy0, py);
					sy1 = std::max(sy1, py);
				}
			}
			if (eye_full)
			{
				out = full_sc;
				return true;
			}
			if (not eye_any)
				continue;
			// Float-domain clamp first: radii are config-controlled with no
			// upper bound, and an out-of-range float->int cast is UB that
			// resolves (x86) to INT_MIN => wrongful skip => under-reveal
			// (fail-unsafe). Clamped floats cast exactly.
			float fx0 = std::min(std::max(sx0 - 2.f, 0.f), (float)extent.width);
			float fy0 = std::min(std::max(sy0 - 2.f, 0.f), (float)extent.height);
			float fx1 = std::min(std::max(sx1 + 2.f, 0.f), (float)extent.width);
			float fy1 = std::min(std::max(sy1 + 2.f, 0.f), (float)extent.height);
			int32_t ix0 = (int32_t)std::floor(fx0);
			int32_t iy0 = (int32_t)std::floor(fy0);
			int32_t ix1 = (int32_t)std::ceil(fx1);
			int32_t iy1 = (int32_t)std::ceil(fy1);
			if (ix1 <= ix0 or iy1 <= iy0)
				continue;
			if (not got)
			{
				ax0 = ix0;
				ay0 = iy0;
				ax1 = ix1;
				ay1 = iy1;
				got = true;
			}
			else
			{
				ax0 = std::min(ax0, ix0);
				ay0 = std::min(ay0, iy0);
				ax1 = std::max(ax1, ix1);
				ay1 = std::max(ay1, iy1);
			}
		}
		if (not got)
			return false;
		out = vk::Rect2D{.offset = {ax0, ay0}, .extent = {(uint32_t)(ax1 - ax0), (uint32_t)(ay1 - ay0)}};
		return true;
	};
	// Both-eyes union of a silhouette draw's per-eye pixel boxes (same
	// full-res bounds scaled into the mask extent +1px pad, clamped).
	// Skip only when both eyes degenerate.
	auto draw_union = [&](const hidden_draw & d, vk::Rect2D & out) {
		bool got = false;
		int32_t ax0 = 0, ay0 = 0, ax1 = 0, ay1 = 0;
		for (int eye = 0; eye < 2; ++eye)
		{
			float msx = (float)extent.width / (float)full_extents[eye].width;
			float msy = (float)extent.height / (float)full_extents[eye].height;
			int32_t x0 = (int32_t)std::max(0.f, d.px[eye][0] * msx - 1.f);
			int32_t y0 = (int32_t)std::max(0.f, d.px[eye][1] * msy - 1.f);
			int32_t x1 = (int32_t)std::min((float)extent.width, d.px[eye][2] * msx + 1.f);
			int32_t y1 = (int32_t)std::min((float)extent.height, d.px[eye][3] * msy + 1.f);
			if (x1 <= x0 or y1 <= y0)
				continue;
			if (not got)
			{
				ax0 = x0;
				ay0 = y0;
				ax1 = x1;
				ay1 = y1;
				got = true;
			}
			else
			{
				ax0 = std::min(ax0, x0);
				ay0 = std::min(ay0, y0);
				ax1 = std::max(ax1, x1);
				ay1 = std::max(ay1, y1);
			}
		}
		if (not got)
			return false;
		out = vk::Rect2D{.offset = {ax0, ay0}, .extent = {(uint32_t)(ax1 - ax0), (uint32_t)(ay1 - ay0)}};
		return true;
	};
	// RP1: proximity field (sphere/cylinder exits, LESS over a
	// near-clear). One multiview begin over the layers=2 framebuffer.
	vk::ClearValue sphere_clear{};
	sphere_clear.depthStencil = vk::ClearDepthStencilValue{1.f, 0};
	vk::RenderPassBeginInfo sphere_begin{
	        .renderPass = *hidden_sphere_rp,
	        .framebuffer = *tgt.sphere_framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = extent},
	        .clearValueCount = 1,
	        .pClearValues = &sphere_clear,
	};
	cmd.beginRenderPass(sphere_begin, vk::SubpassContents::eInline);
	cmd.setViewport(0, full_vp);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *hidden_sphere_pipe);
	cmd.bindVertexBuffers(0, (vk::Buffer)hidden_sphere_verts_, (vk::DeviceSize)0);
	for (const auto & s: spheres)
	{
		vk::Rect2D scissor;
		if (not field_union(s, scissor))
			continue;
		cmd.setScissor(0, scissor);
		cmd.pushConstants<silhouette_mv_push>(*silhouette_mv_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                                 silhouette_mv_push{.mvp = {{s[0], s[1]}}, .opacity = 1});
		cmd.draw(hidden_sphere_vert_count, 1, 0, 0);
	}
	// Body columns: same pipe/push, cylinder verts (same vec3 layout).
	cmd.bindVertexBuffers(0, (vk::Buffer)hidden_cylinder_verts_, (vk::DeviceSize)0);
	for (const auto & c: cylinders)
	{
		vk::Rect2D scissor;
		if (not field_union(c, scissor))
			continue;
		cmd.setScissor(0, scissor);
		cmd.pushConstants<silhouette_mv_push>(*silhouette_mv_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                                 silhouette_mv_push{.mvp = {{c[0], c[1]}}, .opacity = 1});
		cmd.draw(hidden_cylinder_vert_count, 1, 0, 0);
	}
	cmd.endRenderPass();

	// Proximity depth → silhouette test hazard (same attachment layout
	// throughout, write then read).
	vk::ImageMemoryBarrier sph_bar{
	        .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead,
	        .oldLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
	        .newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
	        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .image = sphere_image,
	        .subresourceRange = {
	                .aspectMask = vk::ImageAspectFlagBits::eDepth,
	                .baseMipLevel = 0,
	                .levelCount = 1,
	                .baseArrayLayer = 0,
	                .layerCount = 2,
	        },
	};
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eLateFragmentTests,
	                    vk::PipelineStageFlagBits::eEarlyFragmentTests, {}, {}, {}, sph_bar);

	// RP2: silhouettes, kept only where nearer than the proximity exit
	// (GREATER, no depth write). Hard near-field edge by design. One
	// multiview begin over the layers=2 framebuffer. Per-draw scissor is
	// the both-eyes union (submitted full-rect image: renderArea stays
	// full, correctness).
	vk::ClearValue hidden_clear{};
	hidden_clear.color.float32.fill(0);
	vk::RenderPassBeginInfo color_begin{
	        .renderPass = *hidden_color_rp,
	        .framebuffer = *tgt.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = extent},
	        .clearValueCount = 1,
	        .pClearValues = &hidden_clear,
	};
	cmd.beginRenderPass(color_begin, vk::SubpassContents::eInline);
	cmd.setViewport(0, full_vp);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *hidden_silhouette_pipe);
	for (const auto & d: draws)
	{
		auto mit = meshes.find(d.mesh);
		if (mit == meshes.end() or mit->second.index_count == 0)
			continue;
		vk::Rect2D scissor;
		if (not draw_union(d, scissor))
			continue;
		cmd.setScissor(0, scissor);
		cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
		cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
		cmd.pushConstants<silhouette_mv_push>(*silhouette_mv_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                                 silhouette_mv_push{.mvp = {{d.mvp[0], d.mvp[1]}}, .opacity = d.opacity});
		cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
		++local.draws;
	}
	cmd.endRenderPass();

	if (cpu_stats)
		*cpu_stats = local;
}

void feather_mask_renderer::ensure_targets(vk::Extent2D extent, bool with_intermediates)
{
	if (targets_extent.width == extent.width and targets_extent.height == extent.height and
	    targets_extent.width != 0 and targets_has_intermediates == with_intermediates)
		return;

	// Rare path (extent/tier change): in-flight frames may still read
	// the old images, mirroring setup_reprojection_swapchain.
	device.waitIdle();
	targets_extent = extent;
	targets_has_intermediates = with_intermediates;
	if (not with_intermediates)
	{
		// Tier-0 direct raster never touches the blur/downsample
		// workspaces: it rasters straight into the submitted images
		// (per-image targets map), reversed depth uses output_depth,
		// volume/VR use their own fixed targets. Drop any present
		// (tier downgrade) and skip the ~17 MB alloc. Views die before
		// the images they reference (existing order below).
		for (auto * target: {&target_a, &target_b, &down_targets[0], &down_targets[1],
		                     &down_targets[2], &eblur_targets[0], &eblur_targets[1],
		                     &eblur_targets[2]})
		{
			target->view_array.clear();
			target->blur_fb_mv.clear();
			target->raster_fb_mv.clear();
			target->raster_depth_fb_mv.clear();
			target->depth_view_array.clear();
			target->image.clear();
			target->memory.clear();
			target->depth_image.clear();
			target->depth_memory.clear();
		}
		spdlog::info("Feather mask blur targets: skipped (tier-0 direct raster) {}x{}", extent.width, extent.height);
		return;
	}
	// A/B intermediates at half mask resolution (the mask extent is
	// 64-quantized upstream, hence evenly divisible); tier-1 raster and
	// H run small, V writes the half-res submitted image (the compositor
	// upscales to full).
	vk::Extent2D half_targets = level_extent(extent, 1);
	for (auto * target: {&target_a, &target_b})
	{
		// Views/framebuffers must die before the image they reference
		// (the image assignment below destroys the old image).
		target->view_array.clear();
		target->blur_fb_mv.clear();
		target->raster_fb_mv.clear();
		// Depth companions likewise (before the depth image assignment
		// below destroys the old one). Blur-only target B never
		// rasterizes, so it keeps no depth resources.
		target->raster_depth_fb_mv.clear();
		target->depth_view_array.clear();
		bool needs_depth = (target == &target_a);
		vk::ImageCreateInfo image_info{
		        .imageType = vk::ImageType::e2D,
		        .format = blur_format,
		        .extent = {half_targets.width, half_targets.height, 1},
		        .mipLevels = 1,
		        .arrayLayers = 2,
		        .samples = vk::SampleCountFlagBits::e1,
		        .tiling = vk::ImageTiling::eOptimal,
		        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
		        .sharingMode = vk::SharingMode::eExclusive,
		        .initialLayout = vk::ImageLayout::eUndefined,
		};
		target->image = vk::raii::Image(device, image_info);
		auto requirements = target->image.getMemoryRequirements();
		vk::MemoryAllocateInfo alloc_info{
		        .allocationSize = requirements.size,
		        .memoryTypeIndex = find_memory_type(physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
		};
		target->memory = vk::raii::DeviceMemory(device, alloc_info);
		target->image.bindMemory(*target->memory, 0);
		// Multiview sampling + render target: array view over both
		// layers, plus one layers=2 framebuffer per begun pass for the
		// raster source only (A). Target B is triple scratch: written via
		// the shared triple FB and sampled via its view_array, never
		// begun on its own, so it keeps no framebuffers (null FBs are
		// reset/cleared like the rest, never begun).
		target->view_array = vk::raii::ImageView(device,
		        vk::ImageViewCreateInfo{
		                .image = *target->image,
		                .viewType = vk::ImageViewType::e2DArray,
		                .format = blur_format,
		                .subresourceRange = {
		                        .aspectMask = vk::ImageAspectFlagBits::eColor,
		                        .baseMipLevel = 0,
		                        .levelCount = 1,
		                        .baseArrayLayer = 0,
		                        .layerCount = 2,
		                },
		        });
		vk::ImageView raw_array = *target->view_array;
		if (target == &target_a)
		{
			target->blur_fb_mv = vk::raii::Framebuffer(device,
			        vk::FramebufferCreateInfo{
			                .renderPass = *blur_r8_renderpass_mv,
			                .attachmentCount = 1,
			                .pAttachments = &raw_array,
			                .width = half_targets.width,
			                .height = half_targets.height,
			                .layers = 2,
			        });
			target->raster_fb_mv = vk::raii::Framebuffer(device,
			        vk::FramebufferCreateInfo{
			                .renderPass = *raster_r8_renderpass_mv,
			                .attachmentCount = 1,
			                .pAttachments = &raw_array,
			                .width = half_targets.width,
			                .height = half_targets.height,
			                .layers = 2,
			        });
		}
		if (needs_depth)
		{
			// Depth-tested seed-raster companion: D16 image + array views
			// + one layers=2 color+depth framebuffer (MV raster pass).
			depth_entry depth = make_depth_entry(half_targets);
			target->depth_image = std::move(depth.image);
			target->depth_memory = std::move(depth.memory);
			target->depth_view_array = vk::raii::ImageView(device,
			        vk::ImageViewCreateInfo{
			                .image = (VkImage)*target->depth_image,
			                .viewType = vk::ImageViewType::e2DArray,
			                .format = vk::Format::eD16Unorm,
			                .subresourceRange = {
			                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
			                        .baseMipLevel = 0,
			                        .levelCount = 1,
			                        .baseArrayLayer = 0,
			                        .layerCount = 2,
			                },
			        });
			{
				vk::ImageView mv_atts[2] = {raw_array, *target->depth_view_array};
				target->raster_depth_fb_mv = vk::raii::Framebuffer(device,
				        vk::FramebufferCreateInfo{
				                .renderPass = *raster_r8_depth_renderpass_mv,
				                .attachmentCount = 2,
				                .pAttachments = mv_atts,
				                .width = half_targets.width,
				                .height = half_targets.height,
				                .layers = 2,
				        });
			}
		}
	}
	// Downsample/blur levels at half, quarter and eighth mask resolution
	// (exact halving: the mask extent is 64-quantized upstream). Only the
	// down levels keep framebuffers (seed raster + downsample begins);
	// the eblur workspaces are triple scratch (view_array/image only),
	// like target B above.
	for (int li = 0; li < 3; ++li)
	{
		vk::Extent2D le = level_extent(extent, li + 1);
		for (auto * target: {&down_targets[li], &eblur_targets[li]})
		{
			vk::ImageCreateInfo image_info{
			        .imageType = vk::ImageType::e2D,
		        .format = blur_format,
		        .extent = {le.width, le.height, 1},
			        .mipLevels = 1,
			        .arrayLayers = 2,
			        .samples = vk::SampleCountFlagBits::e1,
			        .tiling = vk::ImageTiling::eOptimal,
			        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			        .sharingMode = vk::SharingMode::eExclusive,
			        .initialLayout = vk::ImageLayout::eUndefined,
			};
			target->view_array.clear();
			target->blur_fb_mv.clear();
			target->raster_fb_mv.clear();
			// Depth companions likewise. Seed raster runs on the down
			// levels (never on the eblur workspaces), so only they keep
			// depth resources.
			target->raster_depth_fb_mv.clear();
			target->depth_view_array.clear();
			bool needs_depth = (target == &down_targets[li]);
			target->image = vk::raii::Image(device, image_info);
			auto requirements = target->image.getMemoryRequirements();
			vk::MemoryAllocateInfo alloc_info{
			        .allocationSize = requirements.size,
			        .memoryTypeIndex = find_memory_type(physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
			};
			target->memory = vk::raii::DeviceMemory(device, alloc_info);
			target->image.bindMemory(*target->memory, 0);
			target->view_array = vk::raii::ImageView(device,
			        vk::ImageViewCreateInfo{
			                .image = *target->image,
			                .viewType = vk::ImageViewType::e2DArray,
			                .format = blur_format,
			                .subresourceRange = {
			                        .aspectMask = vk::ImageAspectFlagBits::eColor,
			                        .baseMipLevel = 0,
			                        .levelCount = 1,
			                        .baseArrayLayer = 0,
			                        .layerCount = 2,
			                },
			        });
			vk::ImageView raw_array = *target->view_array;
			if (target == &down_targets[li])
			{
				target->blur_fb_mv = vk::raii::Framebuffer(device,
				        vk::FramebufferCreateInfo{
				                .renderPass = *blur_r8_renderpass_mv,
				                .attachmentCount = 1,
				                .pAttachments = &raw_array,
				                .width = le.width,
				                .height = le.height,
				                .layers = 2,
				        });
				target->raster_fb_mv = vk::raii::Framebuffer(device,
				        vk::FramebufferCreateInfo{
				                .renderPass = *raster_r8_renderpass_mv,
				                .attachmentCount = 1,
				                .pAttachments = &raw_array,
				                .width = le.width,
				                .height = le.height,
				                .layers = 2,
				        });
			}
			if (needs_depth)
			{
				// Seed raster runs on down_targets[seed-1].raster_fb_mv;
				// the depth path pairs the array color view with a D16
				// array view under the MV R8+depth pass.
				depth_entry depth = make_depth_entry(le);
				target->depth_image = std::move(depth.image);
				target->depth_memory = std::move(depth.memory);
				target->depth_view_array = vk::raii::ImageView(device,
				        vk::ImageViewCreateInfo{
				                .image = (VkImage)*target->depth_image,
				                .viewType = vk::ImageViewType::e2DArray,
				                .format = vk::Format::eD16Unorm,
				                .subresourceRange = {
				                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
				                        .baseMipLevel = 0,
				                        .levelCount = 1,
				                        .baseArrayLayer = 0,
				                        .layerCount = 2,
				                },
				        });
				{
					vk::ImageView mv_atts[2] = {raw_array, *target->depth_view_array};
					target->raster_depth_fb_mv = vk::raii::Framebuffer(device,
					        vk::FramebufferCreateInfo{
					                .renderPass = *raster_r8_depth_renderpass_mv,
					                .attachmentCount = 2,
					                .pAttachments = mv_atts,
					                .width = le.width,
					                .height = le.height,
					                .layers = 2,
					        });
				}
			}
		}
	}
	spdlog::info("Feather mask blur targets: {}x{} (A/B at half, levels below)", extent.width, extent.height);
}

void feather_mask_renderer::ensure_volume_targets(vk::Extent2D working, vk::Extent2D cut)
{
	if (volume_targets_ready and volume_working_extent.width == working.width and
	    volume_working_extent.height == working.height and volume_cut_extent.width == cut.width and
	    volume_cut_extent.height == cut.height)
		return;

	// Rebuilt on extent change (video resize, depth-size change):
	// waitIdle retires in-flight frames using the old views, mirroring
	// the scene swapchain recreation discipline above.
	device.waitIdle();
	vk::Extent2D ve = working;
	vk::Extent2D ce = cut;
	auto make_target_image = [&](vk::Format fmt, vk::Extent2D e) {
		vk::ImageCreateInfo image_info{
		        .imageType = vk::ImageType::e2D,
		        .format = fmt,
		        .extent = {e.width, e.height, 1},
		        .mipLevels = 1,
		        .arrayLayers = 2,
		        .samples = vk::SampleCountFlagBits::e1,
		        .tiling = vk::ImageTiling::eOptimal,
		        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
		        .sharingMode = vk::SharingMode::eExclusive,
		        .initialLayout = vk::ImageLayout::eUndefined,
		};
		vk::raii::Image image(device, image_info);
		auto requirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo alloc_info{
		        .allocationSize = requirements.size,
		        .memoryTypeIndex = find_memory_type(physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
		};
		vk::raii::DeviceMemory memory(device, alloc_info);
		image.bindMemory(*memory, 0);
		return std::make_pair(std::move(image), std::move(memory));
	};

	// Furthest-D canvas (R16, sensor-compatible): array view for the
	// compare sampler + one layers=2 MV raster FB.
	std::tie(volume_depth.image, volume_depth.memory) = make_target_image(ve, volume_depth_format);
	volume_depth.view_array = vk::raii::ImageView(device,
	        vk::ImageViewCreateInfo{
	                .image = *volume_depth.image,
	                .viewType = vk::ImageViewType::e2DArray,
	                .format = volume_depth_format,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .baseMipLevel = 0,
	                        .levelCount = 1,
	                        .baseArrayLayer = 0,
	                        .layerCount = 2,
	                },
	        });
	{
		vk::ImageView raw_mv = *volume_depth.view_array;
		volume_depth.raster_fb_mv = vk::raii::Framebuffer(device,
		        vk::FramebufferCreateInfo{
		                .renderPass = *volume_raster_renderpass_mv,
		                .attachmentCount = 1,
		                .pAttachments = &raw_mv,
		                .width = ve.width,
		                .height = ve.height,
		                .layers = 2,
		        });
	}

	// Virtual-depth D canvas (R16, same 320px 2-layer shape): SBS
	// meters->D conversion output, cleared to far. Own MV framebuffer
	// under volume_virt_renderpass (CLEAR R16 MV).
	std::tie(volume_virt.image, volume_virt.memory) = make_target_image(ce, volume_depth_format);
	volume_virt.view_array = vk::raii::ImageView(device,
	        vk::ImageViewCreateInfo{
	                .image = *volume_virt.image,
	                .viewType = vk::ImageViewType::e2DArray,
	                .format = volume_depth_format,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .baseMipLevel = 0,
	                        .levelCount = 1,
	                        .baseArrayLayer = 0,
	                        .layerCount = 2,
	                },
	        });
	vk::ImageView raw_virt_array = *volume_virt.view_array;
	volume_virt_fb_mv = vk::raii::Framebuffer(device,
	        vk::FramebufferCreateInfo{
	                .renderPass = *volume_virt_renderpass,
	                .attachmentCount = 1,
	                .pAttachments = &raw_virt_array,
	                .width = ce.width,
	                .height = ce.height,
	                .layers = 2,
	        });

	// Binary canvas (R8): compare renders MV into blur_fb_mv under the
	// shared R8 pass; merge samples the array view with LINEAR.
	std::tie(volume_bin.image, volume_bin.memory) = make_target_image(ve, blur_format);
	volume_bin.view_array = vk::raii::ImageView(device,
	        vk::ImageViewCreateInfo{
	                .image = *volume_bin.image,
	                .viewType = vk::ImageViewType::e2DArray,
	                .format = blur_format,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .baseMipLevel = 0,
	                        .levelCount = 1,
	                        .baseArrayLayer = 0,
	                        .layerCount = 2,
	                },
	        });
	vk::ImageView raw_bin_array = *volume_bin.view_array;
	volume_bin.blur_fb_mv = vk::raii::Framebuffer(device,
	        vk::FramebufferCreateInfo{
	                .renderPass = *blur_r8_renderpass_mv,
	                .attachmentCount = 1,
	                .pAttachments = &raw_bin_array,
	                .width = ve.width,
	                .height = ve.height,
	                .layers = 2,
	        });

	// Exact-depth canvas (R16 nearest, same 320px shape): soft-occlusion
	// exact occluders (passthrough + hidden, reversed excluded). Far (1.0)
	// clear, MIN blend via exact_raster_pipeline_mv, sampled as binding 3.
	std::tie(exact_depth.image, exact_depth.memory) = make_target_image(ce, volume_depth_format);
	exact_depth.view_array = vk::raii::ImageView(device,
	        vk::ImageViewCreateInfo{
	                .image = *exact_depth.image,
	                .viewType = vk::ImageViewType::e2DArray,
	                .format = volume_depth_format,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .baseMipLevel = 0,
	                        .levelCount = 1,
	                        .baseArrayLayer = 0,
	                        .layerCount = 2,
	                },
	        });
	{
		vk::ImageView raw_mv = *exact_depth.view_array;
		exact_depth.raster_fb_mv = vk::raii::Framebuffer(device,
		        vk::FramebufferCreateInfo{
		                .renderPass = *volume_raster_renderpass_mv,
		                .attachmentCount = 1,
		                .pAttachments = &raw_mv,
		                .width = ce.width,
		                .height = ce.height,
		                .layers = 2,
		        });
	}

	// VR-cut keep field (R8): compare renders MV into blur_fb_mv under
	// the dedicated CLEAR R8 pass (same shape as the binary canvas).
	std::tie(cut_bin.image, cut_bin.memory) = make_target_image(blur_format, ce);
	cut_bin.view_array = vk::raii::ImageView(device,
	        vk::ImageViewCreateInfo{
	                .image = *cut_bin.image,
	                .viewType = vk::ImageViewType::e2DArray,
	                .format = blur_format,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .baseMipLevel = 0,
	                        .levelCount = 1,
	                        .baseArrayLayer = 0,
	                        .layerCount = 2,
	                },
	        });
	vk::ImageView raw_cut_array = *cut_bin.view_array;
	cut_bin.blur_fb_mv = vk::raii::Framebuffer(device,
	        vk::FramebufferCreateInfo{
	                .renderPass = *compare_renderpass_mv,
	                .attachmentCount = 1,
	                .pAttachments = &raw_cut_array,
	                .width = ce.width,
	                .height = ce.height,
	                .layers = 2,
	        });

	volume_working_extent = working;
	volume_cut_extent = cut;
	volume_targets_ready = true;
	spdlog::info("Feather mask depth targets: work {}x{} cut {}x{}", ve.width, ve.height, ce.width, ce.height);
}

void feather_mask_renderer::flush_upload(vk::raii::CommandBuffer & cmd, mesh_buffers & mesh)
{
	if (not mesh.pending_upload)
		return;
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

void feather_mask_renderer::update_source(vk::ImageView view, uint32_t set, bool nearest,
                                           vk::ImageLayout layout)
{
	vk::DescriptorImageInfo image_info{
	        .sampler = nearest ? *sampler_nearest : *sampler,
	        .imageView = view,
	        .imageLayout = layout,
	};
	vk::WriteDescriptorSet write{
	        .dstSet = *descriptor_sets[set],
	        .dstBinding = 0,
	        .descriptorCount = 1,
	        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	        .pImageInfo = &image_info,
	};
	std::array<vk::CopyDescriptorSet, 0> no_copies{};
	device.updateDescriptorSets(write, no_copies);
}

void feather_mask_renderer::record(vk::raii::CommandBuffer & cmd,
                                   vk::Image image,
                                   vk::Extent2D extent,
                                   const std::vector<instance_draw> & draws,
                                   bool rasterize,
                                   float feather_px,
                                   const std::vector<std::array<glm::vec3, 6>> & cutouts,
                                   const std::array<glm::mat4, 2> & cutout_mvp,
                                   mask_stage_cpu * cpu_stats,
                                   vk::Extent2D out_extent,
                                   const volume_input * volume)
{
	auto t_total0 = std::chrono::steady_clock::now();
	double ms_raster = 0, ms_erode = 0, ms_blur = 0, ms_punch = 0;
	double ms_volume_raster = 0, ms_volume_virt = 0, ms_volume_compare = 0, ms_volume_merge = 0;
	double ms_cut = 0;
	// Depth volume engages exactly when the group carries depth draws
	// with a valid sensor view. The raster is a plain MAX accumulation
	// (no MIN cuts: the merge below MAXes over the reversed result).
	bool use_volume = volume && !volume->draws.empty() && volume->sensor_view != vk::ImageView{};
	// VR cut engages when virtual depth paired and exact geometry exists,
	// independent of use_volume (cut needs no sensor). Extents: sensor
	// path at quarter of full working size (one line to switch to half);
	// cut at min(submit, virt eye) so it is per layer pixel, never
	// upscaled from a coarser grid. Both floor at 64px.
	bool paired_virt = volume && volume->virt_valid && volume->virt_eye_extent.width > 0 && volume->virt_eye_extent.height > 0;
	vk::Extent2D workE{std::max(64u, extent.width / 4), std::max(64u, extent.height / 4)};
	vk::Extent2D cutE = workE;
	if (paired_virt)
	        cutE = {std::max(64u, std::min(out_extent.width, volume->virt_eye_extent.width)),
	                std::max(64u, std::min(out_extent.height, volume->virt_eye_extent.height))};
	bool use_cut = paired_virt && volume && !volume->exact_draws.empty();
	if (use_cut || use_volume)
	        ensure_volume_targets(workE, cutE);
	bool exact_done = false;
	vk::Extent2D exact_done_ext{0, 0};
	float box_frac = 1;
	float volume_box_frac = 1;
	size_t n_draws = 0;
	int tier_out = -1;
	bool has_reversed = std::ranges::any_of(draws, [](const instance_draw & d) { return d.reverse; });
	bool has_positive = std::ranges::any_of(draws, [](const instance_draw & d) { return !d.reverse; });
	// Reversed-only raster is a provable no-op (MIN against clear 0 writes
	// 0), so clear-only and let the cutout punch below run if armed. Depth
	// engages exactly when a reversed draw exists; positive-only calls
	// rasterize as usual.
	auto fill_stats = [&] {
		if (cpu_stats)
		{
			cpu_stats->raster_ms = ms_raster;
			cpu_stats->erode_ms = ms_erode;
			cpu_stats->blur_ms = ms_blur;
			cpu_stats->punch_ms = ms_punch;
			cpu_stats->volume_raster_ms = ms_volume_raster;
			cpu_stats->volume_virt_ms = ms_volume_virt;
			cpu_stats->volume_compare_ms = ms_volume_compare;
			cpu_stats->volume_merge_ms = ms_volume_merge;
			cpu_stats->cut_ms = ms_cut;
			cpu_stats->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total0).count();
			cpu_stats->tier = tier_out;
			cpu_stats->draws = n_draws;
			cpu_stats->box_frac = box_frac;
			cpu_stats->volume_box_frac = volume_box_frac;
		}
	};

	auto set_full_viewport = [&](vk::Extent2D e) {
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)e.width,
		                        .height = (float)e.height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = e});
	};

	// Same-queue read-after-write between passes. Standalone passes
	// end OPTIMAL (UBWC on); the merged triple stays GENERAL across
	// subpasses (cross-subpass sampling).
	// Full-image range: the multiview chain reads both layers together
	// (one barrier).
	auto transfer_barrier = [&](vk::Image img, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
	                            vk::AccessFlags src_access, vk::AccessFlags dst_access,
	                            vk::PipelineStageFlags src_stage, vk::PipelineStageFlags dst_stage) {
	vk::ImageMemoryBarrier barrier{
		        .srcAccessMask = src_access,
		        .dstAccessMask = dst_access,
		        .oldLayout = old_layout,
		        .newLayout = new_layout,
		        .image = img,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .levelCount = 1,
		                .layerCount = 2,
		        },
		};
		cmd.pipelineBarrier(src_stage, dst_stage, {}, {}, {}, barrier);
	};
	// Post-write: color write out, shader read in. old_layout is the
	// pass's final layout (OPTIMAL from standalone passes, GENERAL
	// from the merged triple only); new is always READ_ONLY.
	auto make_readable = [&](vk::Image img, vk::ImageLayout old_layout = vk::ImageLayout::eColorAttachmentOptimal) {
		transfer_barrier(img,
		                 old_layout,
		                 vk::ImageLayout::eShaderReadOnlyOptimal,
		                 vk::AccessFlagBits::eColorAttachmentWrite,
		                 vk::AccessFlagBits::eShaderRead,
		                 vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                 vk::PipelineStageFlagBits::eFragmentShader);
	};
	// Pre-write: prior reads out, color write in. oldLayout Undefined
	// discards (always legal, no first-use tracking); every R8 target
	// is fully overwritten (DONT_CARE load), and the discard drops the
	// tile load on TBDR. Not used before raster (initial Undefined
	// covers) or the OPTIMAL submit upscale.
	auto make_writable = [&](vk::Image img) {
		transfer_barrier(img,
		                 vk::ImageLayout::eUndefined,
		                 vk::ImageLayout::eColorAttachmentOptimal,
		                 vk::AccessFlagBits::eShaderRead,
		                 vk::AccessFlagBits::eColorAttachmentWrite,
		                 vk::PipelineStageFlagBits::eFragmentShader,
		                 vk::PipelineStageFlagBits::eColorAttachmentOutput);
	};

	// Merged-triple framebuffer for this renderer's (src, tmp) image
	// pair (fixed per group: tier is fixed per feather). Creation-only
	// on first record — the pair never changes mid-stream (extent
	// changes rebuild the images themselves, hence new handles), so no
	// waitIdle is needed. Both layers, triple renderpass.
	auto triple_fb_for = [&](vk::Image src_img, vk::Image tmp_img, vk::ImageView src_view, vk::ImageView tmp_view,
	                          vk::Extent2D e) -> vk::Framebuffer {
		if (triple_src != src_img or triple_tmp != tmp_img)
		{
			triple_src = src_img;
			triple_tmp = tmp_img;
			vk::ImageView attachments[2] = {src_view, tmp_view};
			triple_fb = vk::raii::Framebuffer(device,
			        vk::FramebufferCreateInfo{
			                .renderPass = *triple_rp,
			                .attachmentCount = 2,
			                .pAttachments = attachments,
			                .width = e.width,
			                .height = e.height,
			                .layers = 2,
			        });
		}
		return *triple_fb;
	};

	auto it = targets.find(image);
	// Swapchain-image targets track the submitted (output) size, which is
	// half for feathered groups; intermediates below track full extent.
	// Depth companions build lazily on first reversed use (a later
	// reversed frame rebuilds via the has_depth clause).
	if (it == targets.end() or it->second.extent.width != out_extent.width or it->second.extent.height != out_extent.height or
	    (has_reversed and not it->second.has_depth))
	{
		if (it != targets.end())
		{
			// The merge samples these array views cross-frame
			// (a prior async merge CB may still fly on inline-fallback
			// frames): never destroy a live entry out from under it.
			// Rare path (submit resize, first reversed frame) — the
			// idle is affordable here, unlike on the frame path.
			device.waitIdle();
			targets.erase(it);
		}
		// NOTE: the view below is a named local, not an &* temporary: the
		// framebuffer create-info stores the pointer, so the VkImageView
		// must outlive the vk::raii::Framebuffer constructor call.
		// (&* on a vk::raii handle takes the address of a returned
		// temporary: freed stack read as an image view, GPU fault, silent
		// fence hang. That exact bug froze the first mask frame.)
		// Multiview submit target: array view over both layers plus
		// one layers=2 framebuffer under blur_renderpass_mv. All mask
		// output (raster, V upscale, VR/volume merges, cutout punch)
		// is multiview: no per-eye framebuffers remain.
		vk::raii::ImageView mv_array_view(device,
		        vk::ImageViewCreateInfo{
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
		        });
		vk::ImageView mv_raw_array = *mv_array_view;
		vk::raii::Framebuffer mv_framebuffer(device,
		        vk::FramebufferCreateInfo{
		                .renderPass = *blur_renderpass_mv,
		                .attachmentCount = 1,
		                .pAttachments = &mv_raw_array,
		                .width = out_extent.width,
		                .height = out_extent.height,
		                .layers = 2,
		        });
		// Depth-tested raster companion for tier-0 direct raster: one
		// layers=2 color+depth FB under depth_renderpass_mv, built lazily
		// on first reversed use (has_depth); positive-only targets stay
		// lean.
		bool has_depth = false;
		vk::raii::Framebuffer depth_framebuffer_mv{nullptr};
		if (has_reversed)
		{
			vk::ImageView depth_array_view = output_depth_array_view(out_extent);
			vk::ImageView mv_depth_atts[2] = {mv_raw_array, depth_array_view};
			depth_framebuffer_mv = vk::raii::Framebuffer(device,
			        vk::FramebufferCreateInfo{
			                .renderPass = *depth_renderpass_mv,
			                .attachmentCount = 2,
			                .pAttachments = mv_depth_atts,
			                .width = out_extent.width,
			                .height = out_extent.height,
			                .layers = 2,
			        });
			has_depth = true;
		}
		it = targets.emplace(image, frame_targets(out_extent, has_depth, std::move(mv_array_view), std::move(mv_framebuffer))).first;
		if (has_reversed)
			it->second.depth_framebuffer_mv = std::move(depth_framebuffer_mv);
	}

	vk::ClearValue clear{};
	clear.color.float32.fill(0);
	// Shared empty descriptor offsets (all fullscreen passes bind a single
	// set with no dynamic offsets).
	std::array<uint32_t, 0> no_offsets{};

	// Silhouette raster shared by every path: one draw per (mesh,
	// instance) broadcasts to both layers. Missing meshes (removed
	// object racing a queued draw) skip defensively. NDC is
	// resolution-independent, so the same draws serve full- and
	// tier-res targets with only the viewport changing.
	// Raster target bundle: color-only and depth (color+depth) MV
	// framebuffers for one raster site. The caller engages depth
	// whenever a reversed draw exists.
	struct raster_site
	{
		vk::Framebuffer color_fb;
		vk::Framebuffer depth_fb;
		vk::Pipeline pipe;
		vk::Pipeline depth_pipe;
		vk::Pipeline rev_pipe;
		vk::RenderPass rp;
		vk::RenderPass depth_rp;
	};
	// Multiview silhouette raster: one draw broadcasts to both layers via
	// mask_mv.vert (gl_ViewIndex selects the MVP). Same polarity passes as
	// the depth path, same clears; one begin and one pipe bind per pass.
	auto raster_silhouette_mv = [&](const raster_site & site, vk::Extent2D e, bool use_depth) {
		std::array<vk::ClearValue, 2> clears{};
		clears[0].color.float32.fill(0);
		clears[1].depthStencil = vk::ClearDepthStencilValue{0.f, 0};
		vk::RenderPassBeginInfo begin_info{
		        .renderPass = use_depth ? site.depth_rp : site.rp,
		        .framebuffer = use_depth ? site.depth_fb : site.color_fb,
		        .renderArea = {.offset = {0, 0}, .extent = e},
		        .clearValueCount = use_depth ? 2u : 1u,
		        .pClearValues = clears.data(),
		};
		cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
		set_full_viewport(e);
		for (int pass = 0; pass < (use_depth ? 2 : 1); ++pass)
		{
			bool rev_pass = use_depth && pass == 1;
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, rev_pass ? site.rev_pipe : (use_depth ? site.depth_pipe : site.pipe));
			for (const auto & d: draws)
			{
				if (d.reverse != rev_pass)
					continue;
				auto mit = meshes.find(d.mesh);
				if (mit == meshes.end() or mit->second.index_count == 0)
					continue;
				cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
				cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
				float v = d.reverse ? 1.f - d.opacity : d.opacity;
				cmd.pushConstants<silhouette_mv_push>(*silhouette_mv_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
				                                  silhouette_mv_push{.mvp = {{d.mvp[0], d.mvp[1]}}, .opacity = v});
				cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
				++n_draws;
			}
		}
		cmd.endRenderPass();
	};


	// A volume-only group has no silhouette draws: it still runs the
	// tier path (cleared seed, then the volume merge below sets it).
	// Same for a VR-only cut (positives always present in practice, but
	// the gate mirrors the volume one for safety).
	if (not rasterize or (not has_positive and not use_volume))
	{
		section_clock raster_clk(&ms_raster);
		// Bypass: clear-only pass, transparent mask, full stack hot.
		// Also covers reversed-only calls (a provable no-op: MIN against
		// clear 0 writes 0); the cutout punch below still runs if armed.
		// Single multiview clear.
		vk::RenderPassBeginInfo begin_info{
		        .renderPass = *renderpass_mv,
		        .framebuffer = *it->second.framebuffer_mv,
		        .renderArea = {.offset = {0, 0}, .extent = out_extent},
		        .clearValueCount = 1,
		        .pClearValues = &clear,
		};
		cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
		cmd.endRenderPass();
	}
	else
	{
		// No meshes: nothing to raster, but cutouts still punch (the
		// debug-only group has no meshes by design). Tier branches below
		// run over empty draws (clear passes), then the punch stamps.
		if (meshes.empty() and cutouts.empty())
		{
			fill_stats();
			return;
		}

		for (auto & [key, mesh]: meshes)
			flush_upload(cmd, mesh);
		ensure_targets(extent, tier_for_feather(feather_px) != 0);

		// Tier from feather-px (see header): 0 hard edge, 1 half-res
		// blur, 2/4/8 downsampled. Every feathered group submits a
		// half-res image (V writes half pixels, the compositor
		// bilinear-upscales to full on submit), so the blur works in
		// 1/k-size texels (k = 2 tier-1, 4 tier-2/4, 8 tier-8) and the
		// screen-space band is 2k*sqrt(2)*spread px. Divisor 12*k keeps
		// the band continuous across tiers (within float rounding) with
		// tap density flat everywhere. Values past 128 clamp (documented).
		// Selected before raster: tier 0 goes straight into the swapchain
		// image, tier 1 via the half-res A/B intermediates, tiered via a
		// level-2 (tier 2) or level-1 (tier 4/8) seed.
		float f = feather_px > 128.f ? 128.f : feather_px;
		int tier = tier_for_feather(feather_px);
		float spread = 1.0f;
		if (tier == 1)
		{
			// k = 2: band 4*sqrt(2)*spread px.
			spread = f / 24.f;
		}
		else if (tier == 2)
		{
			// k = 4: band 8*sqrt(2)*spread px, same slope as tier 1.
			spread = f / 48.f;
		}
		else if (tier == 4)
			// Same quarter-res blur level as tier 2 (blur_level 2 either
			// way), hence the same divisor.
			spread = f / 48.f;
		else if (tier == 8)
		{
			// k = 8: band 16*sqrt(2)*spread px, same slope again.
			spread = f / 96.f;
		}

		// Half-res working extent for the A/B intermediates (the mask
		// extent is 64-quantized upstream, hence evenly divisible).
		vk::Extent2D half_extent = level_extent(extent, 1);

		// Group content box (full working pixels): NDC bounds of every
		// draw, both eyes, grown by a conservative support margin. The
		// fullscreen image passes (erode/blur/downsample) scissor to it;
		// scissor clips writes only, reads still see full neighborhoods,
		// and the box covers total downstream support by construction.
		// Raster and the V upscale stay fullscreen (coverage-defining and
		// submit-defining respectively: zero risk there). Any
		// behind-camera corner falls back to fullscreen for the group.
		struct content_box
		{
			float minx = 1e30f, miny = 1e30f, maxx = -1e30f, maxy = -1e30f;
			bool behind = false;
			bool empty = true;
		};
		content_box eye_box[2];
		auto accumulate_box = [&](const instance_draw & d) {
			auto mit = meshes.find(d.mesh);
			if (mit == meshes.end())
				return;
			const glm::vec3 & mn = mit->second.aabb_min;
			const glm::vec3 & mx = mit->second.aabb_max;
			for (int eye = 0; eye < 2; ++eye)
			{
				for (int corner = 0; corner < 8; ++corner)
				{
					glm::vec4 q((corner & 1) ? mx.x : mn.x, (corner & 2) ? mx.y : mn.y,
					            (corner & 4) ? mx.z : mn.z, 1);
					glm::vec4 c = d.mvp[eye] * q;
					if (c.w <= 0)
					{
						eye_box[eye].behind = true;
						continue;
					}
					float nx = c.x / c.w, ny = c.y / c.w;
					eye_box[eye].minx = std::min(eye_box[eye].minx, nx);
					eye_box[eye].miny = std::min(eye_box[eye].miny, ny);
					eye_box[eye].maxx = std::max(eye_box[eye].maxx, nx);
					eye_box[eye].maxy = std::max(eye_box[eye].maxy, ny);
					eye_box[eye].empty = false;
				}
			}
		};
		// Content box only feeds pass_scissor, which the tier-0 direct
		// raster never calls: skip the 16N projections + N finds when
		// tier==0 with no volume/cut draws. A future tier-0 consumer of
		// pass_scissor must lift this gate (unconsumed, eye_box stays
		// empty and pass_scissor falls back to fullscreen — fail-safe).
		if (tier != 0 or use_volume or use_cut)
		{
			for (const auto & d: draws)
				accumulate_box(d);
			// Depth volumes feed the same erode/blur chain, so the scissor
			// box must cover them too (viewer-inside volumes usually trip
			// the behind-camera fallback to fullscreen, which is correct).
			if (use_volume)
				for (const auto & d: volume->draws)
					accumulate_box(d);
			// Exact occluders feed the same chain: cover them too
// (cut-only groups have no volume draws but still merge).
			if (volume && !volume->exact_draws.empty())
				for (const auto & d: volume->exact_draws)
					accumulate_box(d);
		}
		// Support margin (full working pixels, conservative): the feather
		// bleed itself plus blur/erode/downsample support at any tier plus
		// rounding. Over-wide just costs area; under-wide would clip.
		float box_margin = feather_px + 8.f * spread + 8.f;
		// Scissor rect for a pass extent: the both-eyes union mapped down
		// from full working pixels, padded in pass texels. Degenerate or
		// behind-camera groups fall back to fullscreen.
		auto pass_scissor = [&](vk::Extent2D pe) {
			float uminx = 1e30f, uminy = 1e30f, umaxx = -1e30f, umaxy = -1e30f;
			for (int eye = 0; eye < 2; ++eye)
			{
				if (eye_box[eye].behind or eye_box[eye].empty)
					return vk::Rect2D{{0, 0}, pe};
				uminx = std::min(uminx, eye_box[eye].minx);
				uminy = std::min(uminy, eye_box[eye].miny);
				umaxx = std::max(umaxx, eye_box[eye].maxx);
				umaxy = std::max(umaxy, eye_box[eye].maxy);
			}
			float sx = (float)pe.width / (float)extent.width;
			float sy = (float)pe.height / (float)extent.height;
			// Mapped in two steps for clarity: full-px box, then scale.
			float fx0 = (uminx * 0.5f + 0.5f) * (float)extent.width - box_margin;
			float fy0 = (uminy * 0.5f + 0.5f) * (float)extent.height - box_margin;
			float fx1 = (umaxx * 0.5f + 0.5f) * (float)extent.width + box_margin;
			float fy1 = (umaxy * 0.5f + 0.5f) * (float)extent.height + box_margin;
			int32_t px0 = std::max<int32_t>(0, (int32_t)(fx0 * sx) - 2);
			int32_t py0 = std::max<int32_t>(0, (int32_t)(fy0 * sy) - 2);
			int32_t px1 = std::min<int32_t>((int32_t)pe.width, (int32_t)(fx1 * sx) + 2);
			int32_t py1 = std::min<int32_t>((int32_t)pe.height, (int32_t)(fy1 * sy) + 2);
			if (px1 <= px0 or py1 <= py0)
				return vk::Rect2D{{0, 0}, pe};
			box_frac = std::min(box_frac, (float)(px1 - px0) * (float)(py1 - py0) / ((float)pe.width * (float)pe.height));
			return vk::Rect2D{{px0, py0}, {(uint32_t)(px1 - px0), (uint32_t)(py1 - py0)}};
		};

		// Raster happens inside each path's own eye loop below (tier 1
		// into half-res A, tiered as a level seed, tier 0 direct into the
		// swapchain image).
		tier_out = tier;

		// Virtual-depth conversion, once per record: SBS meters->D at
		// 320px into volume_virt (shared by the soft compare).
		// Descriptor set update once per record: the conversion binds the
		// decoded view on first use (same-CB draws would otherwise all
		// read the last update).
		bool virt_desc_updated = false;
		bool low_virt_done = false;
		auto convert_virt_canvas = [&](blur_target & virt, vk::raii::Framebuffer & virt_fb, vk::Extent2D ve, bool & done) {
			if (done or not volume or not volume->virt_valid)
				return;
			done = true;
			section_clock virt_clk(&ms_volume_virt);
			if (not virt_desc_updated)
			{
				virt_desc_updated = true;
				vk::DescriptorImageInfo virt_info{
				        .sampler = volume->virt_sampler,
				        .imageView = volume->virt_view,
				        // Decoded depth arrives GENERAL (same barrier
				        // discipline as color in the stream scene).
				        .imageLayout = vk::ImageLayout::eGeneral,
				};
				std::array<vk::CopyDescriptorSet, 0> no_virt_copies{};
				std::array<vk::WriteDescriptorSet, 1> virt_writes = {{
				        {
				                .dstSet = *volume_virt_set,
				                .dstBinding = 0,
				                .descriptorCount = 1,
				                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
				                .pImageInfo = &virt_info,
				        },
				}};
				device.updateDescriptorSets(virt_writes, no_virt_copies);
			}
			vk::ClearValue vfar{};
			const float far1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
			std::memcpy(vfar.color.float32, far1, sizeof(far1));
			vk::RenderPassBeginInfo begin_virt{
			        .renderPass = *volume_virt_renderpass,
			        .framebuffer = *virt_fb,
			        .renderArea = {.offset = {0, 0}, .extent = ve},
			        .clearValueCount = 1,
			        .pClearValues = &vfar,
			};
			cmd.beginRenderPass(begin_virt, vk::SubpassContents::eInline);
			set_full_viewport(ve);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *volume_virt_pipeline);
			std::array<vk::DescriptorSet, 1> virt_sets{*volume_virt_set};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *volume_virt_layout, 0, virt_sets, no_offsets);
			cmd.pushConstants<volume_virt_push>(*volume_virt_layout, vk::ShaderStageFlagBits::eFragment, 0,
			                                    volume_virt_push{.near_z = volume->near_z, .far_z = volume->far_z});
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
			make_readable(*virt.image);
		};

		// Exact-depth MIN raster shared by the cut (cut res, per-pixel) and
		// the sensor compare (LINEAR-downsampled when cut res > working).
		// Rasterized once per group iteration at the first needed extent.
		auto raster_exact = [&](vk::Extent2D re) {
			if (exact_done && exact_done_ext.width == re.width && exact_done_ext.height == re.height)
			        return;
			section_clock exact_clk(&ms_cut);
			for (const auto & d: volume->exact_draws)
			{
				auto mit = meshes.find(d.mesh);
				if (mit != meshes.end())
				        flush_upload(cmd, mit->second);
			}
			vk::ClearValue exact_clear{};
			const float far1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
			std::memcpy(exact_clear.color.float32, far1, sizeof(far1));
			vk::RenderPassBeginInfo begin_exact{
			        .renderPass = *volume_raster_renderpass_mv,
			        .framebuffer = *exact_depth.raster_fb_mv,
			        .renderArea = {.offset = {0, 0}, .extent = re},
			        .clearValueCount = 1,
			        .pClearValues = &exact_clear,
			};
			cmd.beginRenderPass(begin_exact, vk::SubpassContents::eInline);
			set_full_viewport(re);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *exact_raster_pipeline_mv);
			for (const auto & d: volume->exact_draws)
			{
				auto mit = meshes.find(d.mesh);
				if (mit == meshes.end() or mit->second.index_count == 0)
				        continue;
				cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
				cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
				cmd.pushConstants<volume_mv_push>(*volume_raster_layout_mv, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
				                  volume_mv_push{.mvp = {{d.mvp[0], d.mvp[1]}}, .near_z = volume->near_z, .far_z = volume->far_z});
				cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
				++n_draws;
			}
			cmd.endRenderPass();
			make_readable(*exact_depth.image);
			exact_done = true;
			exact_done_ext = re;
		};
		// VR cut (cut res): MIN-cuts the seed where virtual is nearer than
		// the rasterized known geometry (passthrough + hidden + hands),
		// so VR occludes known geometry per layer pixel with no sensor
		// involvement. Runs ahead of the soft MAX (cut-before-add).
		auto run_cut = [&](vk::Image seed_img, vk::Framebuffer seed_fb, vk::Extent2D seed_ext, bool seed_is_rgb8) {
			if (!use_cut)
			        return;
			raster_exact(cutE);
			{
				section_clock compare_clk(&ms_cut);
				vk::DescriptorImageInfo cut_virt_info{
				        .sampler = *sampler_nearest,
				        .imageView = *volume_virt.view_array,
				        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
				};
				vk::DescriptorImageInfo cut_exact_info{
				        .sampler = *sampler_nearest,
				        .imageView = *exact_depth.view_array,
				        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
				};
				std::array<vk::CopyDescriptorSet, 0> no_cut_copies{};
				std::array<vk::WriteDescriptorSet, 2> cut_writes = {{
				        {
				                .dstSet = *cut_set,
				                .dstBinding = 0,
				                .descriptorCount = 1,
				                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
				                .pImageInfo = &cut_virt_info,
				        },
				        {
				                .dstSet = *cut_set,
				                .dstBinding = 1,
				                .descriptorCount = 1,
				                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
				                .pImageInfo = &cut_exact_info,
				        },
				}};
				device.updateDescriptorSets(cut_writes, no_cut_copies);
				vk::RenderPassBeginInfo begin_cut_cmp{
				        .renderPass = *compare_renderpass_mv,
				        .framebuffer = *cut_bin.blur_fb_mv,
				        .renderArea = {.offset = {0, 0}, .extent = cutE},
				        .clearValueCount = 1,
				        .pClearValues = &clear,
				};
				cmd.beginRenderPass(begin_cut_cmp, vk::SubpassContents::eInline);
				set_full_viewport(cutE);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *cut_pipeline_mv);
				std::array<vk::DescriptorSet, 1> cut_sets{*cut_set};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *cut_pipe_layout, 0, cut_sets, no_offsets);
				cmd.draw(3, 1, 0, 0);
				cmd.endRenderPass();
			}
			make_readable(*cut_bin.image);
			{
				section_clock merge_clk(&ms_cut);
				vk::ImageMemoryBarrier cut_barrier{
				        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
				        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
				        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
				        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
				        .image = seed_img,
				        .subresourceRange = {
				                .aspectMask = vk::ImageAspectFlagBits::eColor,
				                .levelCount = 1,
				                .layerCount = 2,
				        },
				};
				cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                    {},
				                    {},
				                    {},
				                    cut_barrier);
				vk::Rect2D merge_box = pass_scissor(seed_ext);
				vk::RenderPassBeginInfo begin_cut_min{
				        .renderPass = seed_is_rgb8 ? *volume_max_rgb8_renderpass : *volume_max_r8_renderpass,
				        .framebuffer = seed_fb,
				        .renderArea = merge_box,
				        .clearValueCount = 0,
				};
				cmd.beginRenderPass(begin_cut_min, vk::SubpassContents::eInline);
				set_full_viewport(seed_ext);
				cmd.setScissor(0, merge_box);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, seed_is_rgb8 ? *cut_min_rgb8_pipeline : *cut_min_r8_pipeline);
				update_source(*cut_bin.view_array, 3, true);
				std::array<vk::DescriptorSet, 1> min_sets{*descriptor_sets[3]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, min_sets, no_offsets);
				blur_push cut_push{{1.f / cutE.width, 1.f / cutE.height}, {0, 1}, 0, 0};
				cmd.pushConstants<blur_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, cut_push);
				cmd.draw(3, 1, 0, 0);
				cmd.endRenderPass();
			}
		};

		// Depth-volume layer: furthest-D raster at 320px, strict
		// sensor<render> flatten, upscale+MAX into the seed. The seed
		// already holds silhouettes + reversed MIN cuts (or is clear),
		// so the merge LOADs and MAXes over them: the volume overlays
		// everything, then erode/blur feather it like silhouettes.
		auto run_volume = [&](vk::Image seed_img, vk::Framebuffer seed_fb, vk::Extent2D seed_ext, bool seed_is_rgb8) {
			ensure_volume_targets(workE, cutE);
			vk::Extent2D ve = workE;
			// GPU brackets for the volume stages (TopOfPipe discipline,
			// same caveat as the mask spans: a late producer inflates the
			// following span). Null pool = no stamping (zero GPU cost).
			// Reset per call: with several volume groups the pool holds
			// the last group only (matches the CPU last-wins reporting).
			bool volume_ts = volume->timestamp_pool != vk::QueryPool{};
			auto volume_stamp = [&](uint32_t q) {
				if (volume_ts)
					cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, volume->timestamp_pool, q);
				};
			if (volume_ts)
				cmd.resetQueryPool(volume->timestamp_pool, 0, 5);
			volume_stamp(0);
			{
				section_clock raster_clk(&ms_volume_raster);
				// Tier-0 has no flush above (direct submit raster), so
				// flush volume meshes here; elsewhere this is a no-op.
				for (const auto & d: volume->draws)
				{
					auto mit = meshes.find(d.mesh);
					if (mit != meshes.end())
						flush_upload(cmd, mit->second);
				}
				vk::ClearValue vclear{};
				vclear.color.float32.fill(0);
				// Single multiview raster.
				vk::RenderPassBeginInfo begin_vol{
				        .renderPass = *volume_raster_renderpass_mv,
				        .framebuffer = *volume_depth.raster_fb_mv,
				        .renderArea = {.offset = {0, 0}, .extent = ve},
				        .clearValueCount = 1,
				        .pClearValues = &vclear,
				};
				cmd.beginRenderPass(begin_vol, vk::SubpassContents::eInline);
				set_full_viewport(ve);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *volume_raster_pipeline_mv);
				for (const auto & d: volume->draws)
				{
					auto mit = meshes.find(d.mesh);
					if (mit == meshes.end() or mit->second.index_count == 0)
						continue;
					cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
					cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
					cmd.pushConstants<volume_mv_push>(*volume_raster_layout_mv, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
					                                  volume_mv_push{.mvp = {{d.mvp[0], d.mvp[1]}}, .near_z = volume->near_z, .far_z = volume->far_z});
					cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
					++n_draws;
				}
				cmd.endRenderPass();
			}
			volume_stamp(1);
			make_readable(*volume_depth.image);
			// Exact depth comes from the shared raster (cut res when paired
// so the VR cut sees it, working res otherwise for the sensor
// tie-break; LINEAR-downsampled by the compare when larger).
if (!volume->exact_draws.empty())
raster_exact(paired_virt ? cutE : workE);
			convert_virt_canvas(volume_virt, volume_virt_fb_mv, cutE, low_virt_done);
			volume_stamp(2);
			{
				section_clock compare_clk(&ms_volume_compare);
				// Homography + B2 corr + bias push fill (144B named vec4s).
				// Pushed just before the draw below.
				volume_homography hom{};
				for (int i = 0; i < 6; ++i)
					hom.col[i] = volume->remap_cols[i];
				for (int i = 0; i < 2; ++i)
					hom.corr[i] = volume->remap_corr[i];
				hom.bias_near_far = glm::vec4(volume->bias, volume->near_z, volume->far_z, 0.f);
				vk::DescriptorImageInfo sensor_info{
					        .sampler = *volume_sampler_linear,
					        .imageView = volume->sensor_view,
					        // The runtime hands the acquired image ready to
					        // sample (acquire is the sync point).
					        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
					};
				vk::DescriptorImageInfo volume_info{
					        .sampler = *volume_sampler_linear,
					        .imageView = *volume_depth.view_array,
					        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
					};
				// Virtual D canvas (already readable when paired; the
				// binding below rebinds volume_info when unpaired).
				vk::DescriptorImageInfo virt_d_info{
					        .sampler = *volume_sampler_linear,
					        .imageView = *volume_virt.view_array,
					        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
					};
				vk::DescriptorImageInfo exact_d_info{
					        .sampler = *volume_sampler_linear,
					        .imageView = *exact_depth.view_array,
					        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
					};
				std::array<vk::CopyDescriptorSet, 0> no_compare_copies{};
				std::array<vk::WriteDescriptorSet, 4> compare_writes = {{
					{
					        .dstSet = *compare_set,
					        .dstBinding = 0,
					        .descriptorCount = 1,
					        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
					        .pImageInfo = &sensor_info,
					},
					{
					        .dstSet = *compare_set,
					        .dstBinding = 1,
					        .descriptorCount = 1,
					        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
					        .pImageInfo = &volume_info,
					},
					{
					        // Paired: virtual D canvas. Unpaired: the
					        // volume canvas again (min(v,v) = sensor-only).
					        .dstSet = *compare_set,
					        .dstBinding = 2,
					        .descriptorCount = 1,
					        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
					        .pImageInfo = volume->virt_valid ? &virt_d_info : &volume_info,
					},
					{
					        // Exact nearest-D (passthrough + hidden). Empty
					        // exact set rebinds the volume canvas (no-op min).
					        .dstSet = *compare_set,
					        .dstBinding = 3,
					        .descriptorCount = 1,
					        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
					        .pImageInfo = volume->exact_draws.empty() ? &volume_info : &exact_d_info,
					},
				}};
				device.updateDescriptorSets(compare_writes, no_compare_copies);
				vk::RenderPassBeginInfo begin_cmp{
					        .renderPass = *compare_renderpass_mv,
					        .framebuffer = *volume_bin.blur_fb_mv,
					        .renderArea = {.offset = {0, 0}, .extent = ve},
					        .clearValueCount = 1,
					        .pClearValues = &clear,
					};
				cmd.beginRenderPass(begin_cmp, vk::SubpassContents::eInline);
				set_full_viewport(ve);
				// Footprint box only: the output footprint is the volume raster
				// footprint (the remap only moves the sensor read), and CLEAR
				// zeroed the canvas so unwritten texels read back defined 0
				// instead of tile garbage.
				cmd.setScissor(0, pass_scissor(ve));
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *compare_pipeline_mv);
				std::array<vk::DescriptorSet, 1> cmp_sets{*compare_set};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *compare_pipe_layout, 0, cmp_sets, no_offsets);
				cmd.pushConstants<volume_homography>(*compare_pipe_layout, vk::ShaderStageFlagBits::eFragment, 0, hom);
				cmd.draw(3, 1, 0, 0);
				cmd.endRenderPass();
			}
			volume_stamp(3);
			make_readable(*volume_bin.image);
			{
				section_clock merge_clk(&ms_volume_merge);
			// No presmooth: the binary compare output upscales straight
				// into the seed; the shared blur chain below does all feathering.
				// Box support already covered by the silhouette/erode margins.

LOAD-after-write ordering, same as the cutout punch
				// barrier below: without it a tile GPU may MAX against
				// stale seed tiles.
				vk::ImageMemoryBarrier merge_barrier{
					        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
					        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
					        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
					        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
					        .image = seed_img,
					        .subresourceRange = {
					                .aspectMask = vk::ImageAspectFlagBits::eColor,
					                .levelCount = 1,
					                .layerCount = 2,
					        },
				};
				cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                    {},
				                    {},
				                    {},
				                    merge_barrier);
				// Box-sized render area: LOAD/STORE skip tiles outside it, so
				// merge bandwidth scales with object area instead of seed area.
				// Viewport stays full (surviving pixels keep correct UVs) and
				// degenerate boxes fall back to fullscreen, never broken.
				vk::Rect2D merge_box = pass_scissor(seed_ext);
				volume_box_frac = (float)merge_box.extent.width * (float)merge_box.extent.height /
				                  ((float)seed_ext.width * (float)seed_ext.height);
				vk::RenderPassBeginInfo begin_max{
					        .renderPass = seed_is_rgb8 ? *volume_max_rgb8_renderpass : *volume_max_r8_renderpass,
					        .framebuffer = seed_fb,
					        .renderArea = merge_box,
					        .clearValueCount = 0,
					};
				cmd.beginRenderPass(begin_max, vk::SubpassContents::eInline);
				set_full_viewport(seed_ext);
				cmd.setScissor(0, merge_box);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, seed_is_rgb8 ? *volume_max_rgb8_pipeline : *volume_max_r8_pipeline);
				// Spread-0 plain upscale under MAX blend (source texel =
				// working res); the shared blur chain feathers the edge.
				update_source(*volume_bin.view_array, 1);
				std::array<vk::DescriptorSet, 1> max_sets{*descriptor_sets[1]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, max_sets, no_offsets);
				blur_push max_push{{1.f / ve.width, 1.f / ve.height}, {0, 1}, 0, 0};
				cmd.pushConstants<blur_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, max_push);
				cmd.draw(3, 1, 0, 0);
				cmd.endRenderPass();
				volume_stamp(4);
			}
		};

		if (tier == 0)
		{
			section_clock raster_clk(&ms_raster);
			// Hard edge: rasterize silhouettes straight into the swapchain
			// image. Exact, one multiview raster pass, no blur passes.
			// (Tier-0 groups submit full, so out matches working here.)
			raster_silhouette_mv({*it->second.framebuffer_mv, *it->second.depth_framebuffer_mv, *pipeline_mv, *depth_pipeline_mv, *reverse_pipeline_mv, *renderpass_mv, *depth_renderpass_mv}, out_extent, has_reversed);
			// VR cut (MIN) then soft reveal (MAX add) after silhouettes +
			// reversed cuts. Cut-before-add lets a sensor-inside reveal
			// win where the real world is nearer than VR.
			if (use_cut)
				run_cut(image, *it->second.framebuffer_mv, out_extent, true);
			if (use_volume)
				run_volume(image, *it->second.framebuffer_mv, out_extent, true);
		}
		else if (tier == 1)
		{
		// Stages 1+2+3: raster into half-res A via one multiview pass,
		// then ONE multiview chain covers both eyes —
		// erode H/V, blur H, V upscale. Same pushes and clears; the _mv
		// fragment twins read each view's layer via gl_ViewIndex, so no
		// per-eye state exists past the raster. One descriptor set per
		// pass, each written once per frame.
		blur_push base{
		        .texel = {1.f / half_extent.width, 1.f / half_extent.height},
		        .spread = spread,
		};
		// Internal-only feather (see header): separable box-min erode
		// at half res (the blur input here), so the band below lands
		// inside the silhouette for any feather width.
		int erode = erode_radius(spread, 2.f);
		{
			section_clock raster_clk(&ms_raster);
			raster_silhouette_mv({*target_a.raster_fb_mv, *target_a.raster_depth_fb_mv, *raster_r8_pipeline_mv, *raster_r8_depth_pipeline_mv, *raster_r8_reverse_pipeline_mv, *raster_r8_renderpass_mv, *raster_r8_depth_renderpass_mv}, half_extent, has_reversed);
		}

		if (use_cut)
			run_cut(*target_a.image, *target_a.blur_fb_mv, half_extent, false);
		if (use_volume)
			run_volume(*target_a.image, *target_a.blur_fb_mv, half_extent, false);
		make_readable(*target_a.image);
		// Merged triple (one renderpass, subpasses 0/1/2): erode H into
		// B, erode V back into A, blur H into B — tile-resident, ordered
		// by subpass dependencies (no explicit barriers inside). Pushes,
		// sets and sampler discipline identical to the standalone
		// passes (nearest for erode taps, GENERAL descriptors inside).
		vk::Framebuffer triple_fb = triple_fb_for(*target_a.image, *target_b.image, *target_a.view_array, *target_b.view_array, half_extent);
		vk::ClearValue triple_clears[2]{};
		vk::RenderPassBeginInfo begin_triple{
		        .renderPass = *triple_rp,
		        .framebuffer = triple_fb,
		        .renderArea = {.offset = {0, 0}, .extent = half_extent},
		        .clearValueCount = 2,
		        .pClearValues = triple_clears,
		};
		cmd.beginRenderPass(begin_triple, vk::SubpassContents::eInline);
		set_full_viewport(half_extent);
		cmd.setScissor(0, pass_scissor(half_extent));
		{
			section_clock erode_clk(&ms_erode);
			update_source(*target_a.view_array, 7, true, vk::ImageLayout::eGeneral);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_erode_0);
			std::array<vk::DescriptorSet, 1> sets_e0{*descriptor_sets[7]};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_e0, no_offsets);
			erode_push epush_h{{1.f / half_extent.width, 1.f / half_extent.height}, {1, 0}, (float)erode, 0};
			cmd.pushConstants<erode_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, epush_h);
			cmd.draw(3, 1, 0, 0);
			cmd.nextSubpass(vk::SubpassContents::eInline);
			update_source(*target_b.view_array, 8, true, vk::ImageLayout::eGeneral);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_erode_1);
			std::array<vk::DescriptorSet, 1> sets_e1{*descriptor_sets[8]};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_e1, no_offsets);
			erode_push epush_v{{1.f / half_extent.width, 1.f / half_extent.height}, {0, 1}, (float)erode, 0};
			cmd.pushConstants<erode_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, epush_v);
			cmd.draw(3, 1, 0, 0);
			cmd.nextSubpass(vk::SubpassContents::eInline);
		}
		{
			section_clock blur_clk(&ms_blur);
			update_source(*target_a.view_array, 0, false, vk::ImageLayout::eGeneral);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_blur_h);
			std::array<vk::DescriptorSet, 1> sets_h{*descriptor_sets[0]};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_h, no_offsets);
			blur_push push = base;
			push.dir[0] = 1;
			push.dir[1] = 0;
			cmd.pushConstants<blur_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, push);
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
		}
		make_readable(*target_b.image, vk::ImageLayout::eGeneral);
		{
			section_clock blur_v_clk(&ms_blur);
			update_source(*target_b.view_array, 2);
		vk::RenderPassBeginInfo begin_v{
		        .renderPass = *blur_renderpass_mv,
		        .framebuffer = *it->second.framebuffer_mv,
		        .renderArea = {.offset = {0, 0}, .extent = out_extent},
		        .clearValueCount = 1,
		        .pClearValues = &clear,
		};
		cmd.beginRenderPass(begin_v, vk::SubpassContents::eInline);
		set_full_viewport(out_extent);
		cmd.setScissor(0, pass_scissor(out_extent));
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *blur_pipeline_mv);
		std::array<vk::DescriptorSet, 1> sets_v{*descriptor_sets[2]};
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_v, no_offsets);
		blur_push push_v = base;
		push_v.dir[0] = 0;
		push_v.dir[1] = 1;
		cmd.pushConstants<blur_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, push_v);
		cmd.draw(3, 1, 0, 0);
		cmd.endRenderPass();
		}
		} // tier == 1
		else
		{
			// Tiered path: seed-rasterize below full res (multiview), one
			// multiview downsample/erode/blur chain to the blur level,
			// blur at 1/k, upscale submit.
			// Tier 2 seeds at quarter (level 2); higher tiers seed at half,
			// so the H working level is the deeper of the seed level and
			// the tier's natural level. down_targets[N].raster_fb_mv serves
			// the seed raster: identical attachment spec, which is all that
			// render-pass compatibility compares (house idiom, cf. A/B).
			int seed_level = (tier == 2) ? 2 : 1;
			vk::Extent2D de_seed = level_extent(extent, seed_level);
			int levels = 0;
			for (int k = tier; k >= 2; k >>= 1)
				++levels;
			int blur_level = std::max(levels, seed_level);
			blur_target & eblur = eblur_targets[blur_level - 1];
			vk::Extent2D te = level_extent(extent, blur_level);
			blur_push base{
			        .texel = {1.f / te.width, 1.f / te.height},
			        .spread = spread,
			};
			// Internal-only feather (see header): separable box-min
			// erode at the blur level (te texels), so the band below
			// lands inside the silhouette for any feather width.
			int erode = erode_radius(spread, (float)(1 << blur_level));
			// Seed raster is multiview; the downsample/erode/blur chain
			// below runs once per level for both eyes (multiview).
			{
				section_clock raster_clk(&ms_raster);
				raster_silhouette_mv({*down_targets[seed_level - 1].raster_fb_mv, *down_targets[seed_level - 1].raster_depth_fb_mv, *raster_r8_pipeline_mv, *raster_r8_depth_pipeline_mv, *raster_r8_reverse_pipeline_mv, *raster_r8_renderpass_mv, *raster_r8_depth_renderpass_mv}, de_seed, has_reversed);
			}
			make_readable(*down_targets[seed_level - 1].image);
			section_clock blur_clk(&ms_blur);
			// Levels seed_level+1..N from the selected tier; sizes shared
			// with ensure via level_extent() (exact halving required for
			// the box map).
			vk::ImageView down_src = *down_targets[seed_level - 1].view_array;
			vk::Extent2D down_src_extent = de_seed;
			int h_src_level = seed_level;
			for (int l = seed_level + 1; l <= levels; ++l)
			{
				blur_target & dst = down_targets[l - 1];
				vk::Extent2D de = level_extent(extent, l);
				update_source(down_src, l + 2);
				make_writable(*dst.image);
				vk::RenderPassBeginInfo begin_down{
				        .renderPass = *blur_r8_renderpass_mv,
				        .framebuffer = *dst.blur_fb_mv,
				        .renderArea = {.offset = {0, 0}, .extent = de},
				        .clearValueCount = 1,
				        .pClearValues = &clear,
				};
				cmd.beginRenderPass(begin_down, vk::SubpassContents::eInline);
				set_full_viewport(de);
				cmd.setScissor(0, pass_scissor(de));
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *downsample_pipeline_mv);
				std::array<vk::DescriptorSet, 1> sets_d{*descriptor_sets[l + 2]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_d, no_offsets);
				blur_push push{};
				push.texel[0] = 1.f / down_src_extent.width;
				push.texel[1] = 1.f / down_src_extent.height;
				cmd.pushConstants<blur_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, push);
				cmd.draw(3, 1, 0, 0);
				cmd.endRenderPass();
				make_readable(*dst.image);
				down_src = *dst.view_array;
				down_src_extent = de;
				h_src_level = l;
			}
			blur_target & h_src = down_targets[h_src_level - 1];

			if (use_cut)
				run_cut(*h_src.image, *h_src.blur_fb_mv, te, false);
			if (use_volume)
				run_volume(*h_src.image, *h_src.blur_fb_mv, te, false);
			make_readable(*h_src.image);
			// Merged triple (one renderpass, subpasses 0/1/2): erode H
			// into eblur, erode V back into h, blur H into eblur —
			// tile-resident, ordered by subpass dependencies. Same
			// pushes/sets/samplers as the standalone passes (nearest
			// for erode taps, GENERAL descriptors inside).
			vk::Framebuffer triple_fb = triple_fb_for(*h_src.image, *eblur.image, *h_src.view_array, *eblur.view_array, te);
			vk::ClearValue triple_clears[2]{};
			vk::RenderPassBeginInfo begin_triple{
			        .renderPass = *triple_rp,
			        .framebuffer = triple_fb,
			        .renderArea = {.offset = {0, 0}, .extent = te},
			        .clearValueCount = 2,
			        .pClearValues = triple_clears,
			};
			cmd.beginRenderPass(begin_triple, vk::SubpassContents::eInline);
			set_full_viewport(te);
			cmd.setScissor(0, pass_scissor(te));
			{
				section_clock erode_clk(&ms_erode);
				update_source(*h_src.view_array, 7, true, vk::ImageLayout::eGeneral);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_erode_0);
				std::array<vk::DescriptorSet, 1> sets_e0{*descriptor_sets[7]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_e0, no_offsets);
				erode_push epush_h{{1.f / te.width, 1.f / te.height}, {1, 0}, (float)erode, 0};
				cmd.pushConstants<erode_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, epush_h);
				cmd.draw(3, 1, 0, 0);
				cmd.nextSubpass(vk::SubpassContents::eInline);
				update_source(*eblur.view_array, 8, true, vk::ImageLayout::eGeneral);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_erode_1);
				std::array<vk::DescriptorSet, 1> sets_e1{*descriptor_sets[8]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_e1, no_offsets);
				erode_push epush_v{{1.f / te.width, 1.f / te.height}, {0, 1}, (float)erode, 0};
				cmd.pushConstants<erode_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, epush_v);
				cmd.draw(3, 1, 0, 0);
				cmd.nextSubpass(vk::SubpassContents::eInline);
			}
			{
				update_source(*h_src.view_array, 0, false, vk::ImageLayout::eGeneral);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_blur_h);
				std::array<vk::DescriptorSet, 1> sets_h{*descriptor_sets[0]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_h, no_offsets);
				blur_push push = base;
				push.dir[0] = 1;
				push.dir[1] = 0;
				cmd.pushConstants<blur_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, push);
				cmd.draw(3, 1, 0, 0);
				cmd.endRenderPass();
			}
			make_readable(*eblur.image, vk::ImageLayout::eGeneral);
			update_source(*eblur.view_array, 2);
			vk::RenderPassBeginInfo begin_v{
			        .renderPass = *blur_renderpass_mv,
			        .framebuffer = *it->second.framebuffer_mv,
			        .renderArea = {.offset = {0, 0}, .extent = out_extent},
			        .clearValueCount = 1,
			        .pClearValues = &clear,
			};
			cmd.beginRenderPass(begin_v, vk::SubpassContents::eInline);
			set_full_viewport(out_extent);
			cmd.setScissor(0, pass_scissor(out_extent));
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *blur_pipeline_mv);
			std::array<vk::DescriptorSet, 1> sets_v{*descriptor_sets[2]};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_v, no_offsets);
			blur_push push_v = base;
			push_v.dir[0] = 0;
			push_v.dir[1] = 1;
			cmd.pushConstants<blur_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, push_v);
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
		} // tiered
	}

	// Marker window cutouts: crisp post-blur punch. Re-begins the finished
	// swapchain framebuffer (multiview LOAD: framebuffer-compatible with
	// the CLEAR passes above) and stamps each quad at FULL alpha (mask 1
	// = reality window): holes show passthrough, not game. No blur pass
	// touches them, so edges stay pixel-exact. One draw per quad
	// broadcasts to both layers via the silhouette MV pipe (opacity 1:
	// blend-off overwrite, identical to the retired per-eye punch).
	// Debug only; rasterize=false (bypass) skips them with the silhouettes.
	if (not cutouts.empty() and rasterize)
	{
		// Execution barrier: the V upscale (or tier-0 raster) above wrote
		// the swapchain image as COLOR_ATTACHMENT; the LOAD cutout pass
		// below reads it. Without this a tile GPU may LOAD stale tiles.
		vk::ImageMemoryBarrier cut_barrier{
		        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
		        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .image = image,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .levelCount = 1,
		                .layerCount = 2,
		        },
		};
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                    {},
		                    {},
		                    {},
		                    cut_barrier);
		// One vertex buffer for every quad, uploaded once: record() only
		// records, so per-quad memcpys before submit would leave every
		// draw reading the last quad. Grown on demand (debug path only).
		vk::DeviceSize cut_need = sizeof(glm::vec3) * 6 * cutouts.size();
		if (not cutout_verts or cutout_verts.size() < cut_need)
		{
			cutout_verts = buffer_allocation{
			        device,
			        vk::BufferCreateInfo{
			                .size = cut_need,
			                .usage = vk::BufferUsageFlagBits::eVertexBuffer,
			        },
			        VmaAllocationCreateInfo{
			                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			                .usage = VMA_MEMORY_USAGE_AUTO,
			        },
			        "feather_mask cutout",
			};
		}
		for (size_t qi = 0; qi < cutouts.size(); ++qi)
			std::memcpy((char *)cutout_verts.map() + qi * sizeof(glm::vec3) * 6, cutouts[qi].data(), sizeof(glm::vec3) * 6);
		// House idiom (cf. gpu_buffer): unmap flushes host writes for GPU read.
		cutout_verts.unmap();
		// Both-eyes union of one quad's pixel bounds (same NDC mapping
		// the viewport itself uses, so alignment is exact by
		// construction): fullscreen LOAD/STORE traffic for a small debug
		// rect would cost a full image round-trip per group. Any corner
		// behind the camera falls back to fullscreen for that quad.
		// Bounds track the submitted (output) size, like the viewport
		// below: NDC maps consistently at any size. False only when the
		// quad is degenerate in both eyes (nothing to punch).
		auto quad_union = [&](const std::array<glm::vec3, 6> & quad, vk::Rect2D & out) {
			bool got = false;
			int32_t ax0 = 0, ay0 = 0, ax1 = 0, ay1 = 0;
			for (int eye = 0; eye < 2; ++eye)
			{
				float minx = (float)out_extent.width, miny = (float)out_extent.height;
				float maxx = 0, maxy = 0;
				bool behind = false;
				for (const auto & v: quad)
				{
					glm::vec4 c = cutout_mvp[eye] * glm::vec4(v, 1);
					if (c.w <= 0)
					{
						behind = true;
						break;
					}
					float px = (c.x / c.w * 0.5f + 0.5f) * (float)out_extent.width;
					float py = (c.y / c.w * 0.5f + 0.5f) * (float)out_extent.height;
					minx = std::min(minx, px);
					miny = std::min(miny, py);
					maxx = std::max(maxx, px);
					maxy = std::max(maxy, py);
				}
				if (behind)
				{
					out = vk::Rect2D{{0, 0}, out_extent};
					return true;
				}
				int32_t x0 = std::clamp<int32_t>((int32_t)std::floor(minx) - 1, 0, (int32_t)out_extent.width);
				int32_t y0 = std::clamp<int32_t>((int32_t)std::floor(miny) - 1, 0, (int32_t)out_extent.height);
				int32_t x1 = std::clamp<int32_t>((int32_t)std::ceil(maxx) + 1, 0, (int32_t)out_extent.width);
				int32_t y1 = std::clamp<int32_t>((int32_t)std::ceil(maxy) + 1, 0, (int32_t)out_extent.height);
				if (x1 <= x0 or y1 <= y0)
					continue;
				if (not got)
				{
					ax0 = x0;
					ay0 = y0;
					ax1 = x1;
					ay1 = y1;
					got = true;
				}
				else
				{
					ax0 = std::min(ax0, x0);
					ay0 = std::min(ay0, y0);
					ax1 = std::max(ax1, x1);
					ay1 = std::max(ay1, y1);
				}
			}
			if (not got)
				return false;
			out = vk::Rect2D{{ax0, ay0}, {(uint32_t)(ax1 - ax0), (uint32_t)(ay1 - ay0)}};
			return true;
		};
		// renderArea must cover every draw below: union of all quad
		// boxes (fullscreen when any quad falls back or all degenerate —
		// over-wide just costs area, never correctness).
		vk::Rect2D cut_area{{0, 0}, out_extent};
		{
			bool got = false;
			int32_t ax0 = 0, ay0 = 0, ax1 = 0, ay1 = 0;
			for (const auto & quad: cutouts)
			{
				vk::Rect2D box;
				if (not quad_union(quad, box))
					continue;
				if (box.extent.width == out_extent.width and box.extent.height == out_extent.height)
				{
					got = false;
					break;
				}
				if (not got)
				{
					ax0 = box.offset.x;
					ay0 = box.offset.y;
					ax1 = box.offset.x + (int32_t)box.extent.width;
					ay1 = box.offset.y + (int32_t)box.extent.height;
					got = true;
				}
				else
				{
					ax0 = std::min(ax0, box.offset.x);
					ay0 = std::min(ay0, box.offset.y);
					ax1 = std::max(ax1, box.offset.x + (int32_t)box.extent.width);
					ay1 = std::max(ay1, box.offset.y + (int32_t)box.extent.height);
				}
			}
			if (got)
				cut_area = vk::Rect2D{{ax0, ay0}, {(uint32_t)(ax1 - ax0), (uint32_t)(ay1 - ay0)}};
		}
		vk::RenderPassBeginInfo begin_cut{
		        .renderPass = *volume_max_rgb8_renderpass,
		        .framebuffer = *it->second.framebuffer_mv,
		        .renderArea = cut_area,
		        .clearValueCount = 0,
		};
		cmd.beginRenderPass(begin_cut, vk::SubpassContents::eInline);
		set_full_viewport(out_extent);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_mv);
		cmd.bindVertexBuffers(0, (vk::Buffer)cutout_verts, (vk::DeviceSize)0);
		cmd.pushConstants<silhouette_mv_push>(*silhouette_mv_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                                      silhouette_mv_push{.mvp = {{cutout_mvp[0], cutout_mvp[1]}}, .opacity = 1});
		section_clock punch_clk(&ms_punch);
		for (size_t qi = 0; qi < cutouts.size(); ++qi)
		{
			vk::Rect2D area;
			if (not quad_union(cutouts[qi], area))
				continue;
			cmd.setScissor(0, area);
			cmd.draw(6, 1, (uint32_t)(qi * 6), 0);
		}
		cmd.endRenderPass();
	}

	// All mask passes recorded (barrier below is submit bookkeeping, not
	// mask work): stamp the diagnostics before leaving.
	fill_stats();

	vk::ImageMemoryBarrier barrier{
	        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
	        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
	        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
	        .image = image,
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
	                    barrier);
}
