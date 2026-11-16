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
	// One-shot forensic for the v3 depth chain (H2): it MIN-blends and
	// linearly samples R16F (intermediate/dilTemp/sensorW/rev_depth)
	// and linearly samples the runtime D16 sensor in the warp. If this
	// driver exposes R16F without BLEND or LINEAR_FILTER (or D16
	// without LINEAR_FILTER), those passes produce undefined values —
	// bilateral layer corruption. Logged once per renderer
	// construction; no behavior change (fallbacks, if needed, are a
	// separate change once run.log gives the verdict).
	{
		auto feat = [&](vk::Format f) { return physical_device.getFormatProperties(f).optimalTilingFeatures; };
		auto has = [](vk::FormatFeatureFlags fs, vk::FormatFeatureFlagBits b) { return (fs & b) == b; };
		using F = vk::FormatFeatureFlagBits;
		vk::FormatFeatureFlags r16 = feat(vk::Format::eR16Sfloat);
		vk::FormatFeatureFlags r8 = feat(vk::Format::eR8Unorm);
		vk::FormatFeatureFlags d16 = feat(vk::Format::eD16Unorm);
		spdlog::info("Depth layer formats (optimal tiling): R16F attach={} blend={} sampled={} linear={}; R8 attach={} blend={} sampled={} linear={}; D16 sampled={} linear={}",
		             has(r16, F::eColorAttachment), has(r16, F::eColorAttachmentBlend), has(r16, F::eSampledImage),
		             has(r16, F::eSampledImageFilterLinear), has(r8, F::eColorAttachment), has(r8, F::eColorAttachmentBlend),
		             has(r8, F::eSampledImage), has(r8, F::eSampledImageFilterLinear), has(d16, F::eSampledImage),
		             has(d16, F::eSampledImageFilterLinear));
		if (not has(r16, F::eColorAttachmentBlend) or not has(r16, F::eSampledImageFilterLinear))
			spdlog::error("Depth layer formats: R16F lacks {} — MIN-blend/linear taps on the common layer are UNDEFINED on this driver",
			              not has(r16, F::eColorAttachmentBlend) ? "BLEND" : "LINEAR_FILTER");
		if (not has(d16, F::eSampledImageFilterLinear))
			spdlog::error("Depth layer formats: D16 lacks LINEAR_FILTER — warp sensor taps are UNDEFINED on this driver");
	}
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
	        .descriptorCount = 11,
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .maxSets = 11,
	        .poolSizeCount = 1,
	        .pPoolSizes = &pool_size,
	};
	descriptor_pool = vk::raii::DescriptorPool(device, pool_info);
	for (int i = 0; i < 11; ++i)
	{
		vk::DescriptorSetAllocateInfo set_alloc_info{
		        .descriptorPool = *descriptor_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*descriptor_layout,
		};
		descriptor_sets.emplace_back(std::move(device.allocateDescriptorSets(set_alloc_info)[0]));
	}

	// Common depth layer (3d-depth-passthrough, tier-1 only): hidden +
	// bbox geometry raster into the merge-owned intermediate (R16F
	// linear meters, MIN over the large-finite FAR clear), sensor warp
	// (D domain, hands MINed) tested against the min-dilated layer
	// natively (bias on the layer only), winning sensor depths MINed
	// back into the raw intermediate, compare verdict into the tier seed
	// ahead of erode/blur. Raster encodes linear meters as R16F (MIN
	// blend, no depth test); compare flattens to binary R8.
	static_assert(sizeof(volume_mv_push) == 136);
	static_assert(sizeof(volume_mv_push) <= 256);
	static_assert(sizeof(geom_depth_push) == 128);
	static_assert(sizeof(layer_add_push) == 8);
	static_assert(sizeof(warp_push) == 96);
	static_assert(sizeof(compare_push) == 16);
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
		vk::PipelineRasterizationStateCreateInfo geom_raster_base{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eBack,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo volume_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		// MIN accumulation (order-independent nearest, float-safe: src/dst
		// factors are ignored by MIN/MAX).
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
		vk::DynamicState volume_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo volume_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = volume_dynamics,
		};
		vk::AttachmentDescription volume_raster_attachment{
		        .format = linear_depth_format,
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
		// Exact-depth raster: nearest state for the hand draws (shared pipe):
		// cull-nothing + MIN over far. Shared attachment/subpass shapes
		// below; only the clear value (1.0) rides the begin info, not
		// the pass.
		vk::PipelineRasterizationStateCreateInfo exact_rasterization = geom_raster_base;
		exact_rasterization.cullMode = vk::CullModeFlagBits::eNone;
		// Multiview hands raster: same R16F CLEAR/STORE/OPTIMAL attachment plus
		// viewMask=both, volume_mv.vert (MVP pair) + matching fragment
		// twin (push blocks must agree byte-for-byte — mask_mv precedent).
		// D-encodes eye depth for the MIN over the warped sensor canvas.
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
		// Common-layer geometry raster: linear meters (geom_depth_mv) MIN
		// over the borrowed intermediate, which the merge base wrote
		// earlier in the same command buffer — so LOAD (initial OPTIMAL,
		// no discard), never CLEAR. Three cull variants share one layout:
		// BACK (hidden), FRONT (bbox far wall on the closed unit box),
		// NONE (rev cuts, legacy coverage from either side).
		if (physical_device.getProperties().limits.maxPushConstantsSize < sizeof(geom_depth_push))
			throw std::runtime_error("feather geom depth multiview needs 128B push constants");
		{
			vk::PushConstantRange geom_push_range{
			        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			        .offset = 0,
			        .size = sizeof(geom_depth_push),
			};
			vk::PipelineLayoutCreateInfo geom_layout_info{
			        .pushConstantRangeCount = 1,
			        .pPushConstantRanges = &geom_push_range,
			};
			geom_raster_layout_mv = vk::raii::PipelineLayout(device, geom_layout_info);
		}
		uint32_t geom_mv_mask = 0b11, geom_mv_corr = 0b11;
		vk::RenderPassMultiviewCreateInfo geom_mv_info{
		        .subpassCount = 1,
		        .pViewMasks = &geom_mv_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &geom_mv_corr,
		};
		vk::AttachmentDescription geom_load_attachment{
		        .format = linear_depth_format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eLoad,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference geom_load_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription geom_load_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &geom_load_ref,
		};
		geom_load_renderpass_mv = vk::raii::RenderPass(device, vk::RenderPassCreateInfo{
		        .pNext = &geom_mv_info,
		        .attachmentCount = 1,
		        .pAttachments = &geom_load_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &geom_load_subpass,
		});
		auto geom_vert = load_shader(device, "geom_depth_mv.vert");
		auto geom_frag = load_shader(device, "geom_depth_mv.frag");
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
		auto make_geom_pipe = [&](vk::CullModeFlags cull) {
			vk::PipelineRasterizationStateCreateInfo rs = geom_raster_base;
			rs.cullMode = cull;
			vk::GraphicsPipelineCreateInfo info{
			        .stageCount = 2,
			        .pStages = geom_stages,
			        .pVertexInputState = &volume_vertex_input,
			        .pInputAssemblyState = &volume_input_assembly,
			        .pViewportState = &volume_viewport_state,
			        .pRasterizationState = &rs,
			        .pMultisampleState = &volume_multisample,
			        .pColorBlendState = &exact_blend,
			        .pDynamicState = &volume_dynamic,
			        .layout = *geom_raster_layout_mv,
			        .renderPass = *geom_load_renderpass_mv,
			};
			return vk::raii::Pipeline(device, *pipeline_cache(device), info);
		};
		geom_back_pipeline_mv = make_geom_pipe(vk::CullModeFlagBits::eBack);
		geom_front_pipeline_mv = make_geom_pipe(vk::CullModeFlagBits::eFront);
		geom_none_pipeline_mv = make_geom_pipe(vk::CullModeFlagBits::eNone);
		// Layer add: fullscreen MIN-blend LOAD back into the raw
		// intermediate (linearized sensor). Own 1-binding layout + 8B
		// push, under the same LOAD renderpass. No vertex input: the
		// fullscreen triangle comes from gl_VertexIndex.
		{
			auto add_frag = load_shader(device, "layer_add_mv.frag");
			vk::PipelineShaderStageCreateInfo add_stages[2] = {
			        {
			                .stage = vk::ShaderStageFlagBits::eVertex,
			                .module = **blur_vert,
			                .pName = "main",
			        },
			        {
			                .stage = vk::ShaderStageFlagBits::eFragment,
			                .module = **add_frag,
			                .pName = "main",
			        },
			};
			vk::DescriptorSetLayoutBinding add_binding{
			        .binding = 0,
			        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			        .descriptorCount = 1,
			        .stageFlags = vk::ShaderStageFlagBits::eFragment,
			};
			vk::DescriptorSetLayoutCreateInfo add_set_info{
			        .bindingCount = 1,
			        .pBindings = &add_binding,
			};
			add_set_layout = vk::raii::DescriptorSetLayout(device, add_set_info);
			vk::PushConstantRange add_push_range{
			        .stageFlags = vk::ShaderStageFlagBits::eFragment,
			        .offset = 0,
			        .size = sizeof(layer_add_push),
			};
			vk::PipelineLayoutCreateInfo add_pipe_layout_info{
			        .setLayoutCount = 1,
			        .pSetLayouts = &*add_set_layout,
			        .pushConstantRangeCount = 1,
			        .pPushConstantRanges = &add_push_range,
			};
			add_pipe_layout = vk::raii::PipelineLayout(device, add_pipe_layout_info);
			vk::DescriptorPoolSize add_pool_size{
			        .type = vk::DescriptorType::eCombinedImageSampler,
			        .descriptorCount = 1,
			};
			vk::DescriptorPoolCreateInfo add_pool_info{
			        .maxSets = 1,
			        .poolSizeCount = 1,
			        .pPoolSizes = &add_pool_size,
			};
			add_pool = vk::raii::DescriptorPool(device, add_pool_info);
			vk::DescriptorSetAllocateInfo add_alloc_info{
			        .descriptorPool = *add_pool,
			        .descriptorSetCount = 1,
			        .pSetLayouts = &*add_set_layout,
			};
			add_set = std::move(device.allocateDescriptorSets(add_alloc_info)[0]);
			vk::PipelineVertexInputStateCreateInfo add_vertex_input{};
			vk::PipelineInputAssemblyStateCreateInfo add_input_assembly{
			        .topology = vk::PrimitiveTopology::eTriangleList,
			};
			vk::PipelineViewportStateCreateInfo add_viewport_state{
			        .viewportCount = 1,
			        .scissorCount = 1,
			};
			vk::PipelineRasterizationStateCreateInfo add_rasterization{
			        .polygonMode = vk::PolygonMode::eFill,
			        .cullMode = vk::CullModeFlagBits::eNone,
			        .frontFace = vk::FrontFace::eCounterClockwise,
			        .lineWidth = 1,
			};
			vk::PipelineMultisampleStateCreateInfo add_multisample{
			        .rasterizationSamples = vk::SampleCountFlagBits::e1,
			};
			vk::PipelineColorBlendAttachmentState add_blend_attachment{
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
			vk::PipelineColorBlendStateCreateInfo add_blend{
			        .attachmentCount = 1,
			        .pAttachments = &add_blend_attachment,
			};
			vk::DynamicState add_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
			vk::PipelineDynamicStateCreateInfo add_dynamic{
			        .dynamicStateCount = 2,
			        .pDynamicStates = add_dynamics,
			};
			vk::GraphicsPipelineCreateInfo add_pipeline_info{
			        .stageCount = 2,
			        .pStages = add_stages,
			        .pVertexInputState = &add_vertex_input,
			        .pInputAssemblyState = &add_input_assembly,
			        .pViewportState = &add_viewport_state,
			        .pRasterizationState = &add_rasterization,
			        .pMultisampleState = &add_multisample,
			        .pColorBlendState = &add_blend,
			        .pDynamicState = &add_dynamic,
			        .layout = *add_pipe_layout,
			        .renderPass = *geom_load_renderpass_mv,
			};
			add_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), add_pipeline_info);
		}
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
	// Sensor compare: fullscreen _mv binary flatten (warped sensor D vs
	// min-dilated R16F layer, same UV). Renders into
	// volume_bin.blur_fb_mv under the shared R8 MV pass
	// (identical attachment shape); only the fragment shader + set
	// layout (2 samplers) are new. The layer-only bias lives in the
	// push here, not in a retired clean (sensor exact-cut, no margin).
	{
		auto compare_frag = load_shader(device, "layer_compare_mv.frag");
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
		vk::DescriptorSetLayoutBinding compare_bindings[2] = {
		        {
		                // Warped sensor (hands MINed over it, D domain).
		                .binding = 0,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		        {
		                // Min-dilated common layer (R16F linear meters,
		                // sampled NEAREST at record time: exact min
		                // values, no invented intermediates).
		                .binding = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		};
		vk::DescriptorSetLayoutCreateInfo compare_set_info{
		        .bindingCount = 2,
		        .pBindings = compare_bindings,
		};
		// 16B sensor-compare push (near/far/bias): far below the
		// 128-byte Vulkan minimum, so no limit throw is needed.
		compare_set_layout = vk::raii::DescriptorSetLayout(device, compare_set_info);
		vk::PushConstantRange compare_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(compare_push),
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
		        .descriptorCount = 2,
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
	}
	// Sensor warp: fullscreen _mv rotation warp of the runtime sensor
	// (sensor extent) into sensorW (blend off; the skinned hand draws MIN-blend
	// after in the same pass via exact_raster_pipeline_mv, which is
	// renderpass-compatible). Renders under
	// volume_raster_renderpass_mv (same R16F CLEAR/STORE/OPTIMAL MV
	// shape, viewMask=both).
	{
		auto warp_frag = load_shader(device, "sensor_warp_mv.frag");
		vk::PipelineShaderStageCreateInfo warp_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **blur_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **warp_frag,
		                .pName = "main",
		        },
		};
		vk::DescriptorSetLayoutBinding warp_binding{
		        .binding = 0,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 1,
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		};
		vk::DescriptorSetLayoutCreateInfo warp_set_info{
		        .bindingCount = 1,
		        .pBindings = &warp_binding,
		};
		warp_set_layout = vk::raii::DescriptorSetLayout(device, warp_set_info);
		vk::PushConstantRange warp_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(warp_push),
		};
		vk::PipelineLayoutCreateInfo warp_pipe_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*warp_set_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &warp_push_range,
		};
		warp_pipe_layout = vk::raii::PipelineLayout(device, warp_pipe_layout_info);
		vk::DescriptorPoolSize warp_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 1,
		};
		vk::DescriptorPoolCreateInfo warp_pool_info{
		        .maxSets = 1,
		        .poolSizeCount = 1,
		        .pPoolSizes = &warp_pool_size,
		};
		warp_pool = vk::raii::DescriptorPool(device, warp_pool_info);
		vk::DescriptorSetAllocateInfo warp_alloc_info{
		        .descriptorPool = *warp_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*warp_set_layout,
		};
		warp_set = std::move(device.allocateDescriptorSets(warp_alloc_info)[0]);
		vk::PipelineVertexInputStateCreateInfo warp_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo warp_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo warp_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo warp_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo warp_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState warp_blend_attachment{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo warp_blend{
		        .attachmentCount = 1,
		        .pAttachments = &warp_blend_attachment,
		};
		vk::DynamicState warp_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo warp_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = warp_dynamics,
		};
		vk::GraphicsPipelineCreateInfo warp_pipeline_info{
		        .stageCount = 2,
		        .pStages = warp_stages,
		        .pVertexInputState = &warp_vertex_input,
		        .pInputAssemblyState = &warp_input_assembly,
		        .pViewportState = &warp_viewport_state,
		        .pRasterizationState = &warp_rasterization,
		        .pMultisampleState = &warp_multisample,
		        .pColorBlendState = &warp_blend,
		        .pDynamicState = &warp_dynamic,
		        .layout = *warp_pipe_layout,
		        .renderPass = *volume_raster_renderpass_mv,
		};
		warp_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), warp_pipeline_info);
	}
	// Alpha build: fullscreen _mv sensor-verdict field (2 samplers:
	// verdict R8 + rev linear R16F, no push). Renders into target_a.blur_fb_mv
	// (the tier-1 seed, R8 MV) under compare_renderpass_mv (same R8 MV
	// CLEAR shape; loadOp is not part of render-pass compatibility).
	{
		auto alpha_frag = load_shader(device, "alpha_build_mv.frag");
		vk::PipelineShaderStageCreateInfo alpha_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **blur_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **alpha_frag,
		                .pName = "main",
		        },
		};
		vk::DescriptorSetLayoutBinding alpha_bindings[2] = {
		        {
		                // Sensor-verdict canvas (R8 at W). Binary reveal.
		                .binding = 0,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		        {
		                // Reversed-depth scratch (R16F linear). No
		                // reversed draws rasterizes to far (no-op cut).
		                .binding = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		};
		vk::DescriptorSetLayoutCreateInfo alpha_set_info{
		        .bindingCount = 2,
		        .pBindings = alpha_bindings,
		};
		alpha_set_layout = vk::raii::DescriptorSetLayout(device, alpha_set_info);
		vk::PipelineLayoutCreateInfo alpha_pipe_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*alpha_set_layout,
		};
		alpha_pipe_layout = vk::raii::PipelineLayout(device, alpha_pipe_layout_info);
		vk::DescriptorPoolSize alpha_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 2,
		};
		vk::DescriptorPoolCreateInfo alpha_pool_info{
		        .maxSets = 1,
		        .poolSizeCount = 1,
		        .pPoolSizes = &alpha_pool_size,
		};
		alpha_pool = vk::raii::DescriptorPool(device, alpha_pool_info);
		vk::DescriptorSetAllocateInfo alpha_alloc_info{
		        .descriptorPool = *alpha_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*alpha_set_layout,
		};
		alpha_set = std::move(device.allocateDescriptorSets(alpha_alloc_info)[0]);
		vk::PipelineVertexInputStateCreateInfo alpha_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo alpha_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo alpha_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo alpha_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo alpha_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState alpha_blend_attachment{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo alpha_blend{
		        .attachmentCount = 1,
		        .pAttachments = &alpha_blend_attachment,
		};
		vk::DynamicState alpha_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo alpha_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = alpha_dynamics,
		};
		vk::GraphicsPipelineCreateInfo alpha_pipeline_info{
		        .stageCount = 2,
		        .pStages = alpha_stages,
		        .pVertexInputState = &alpha_vertex_input,
		        .pInputAssemblyState = &alpha_input_assembly,
		        .pViewportState = &alpha_viewport_state,
		        .pRasterizationState = &alpha_rasterization,
		        .pMultisampleState = &alpha_multisample,
		        .pColorBlendState = &alpha_blend,
		        .pDynamicState = &alpha_dynamic,
		        .layout = *alpha_pipe_layout,
		        .renderPass = *compare_renderpass_mv,
		};
		alpha_pipeline_mv = vk::raii::Pipeline(device, *pipeline_cache(device), alpha_pipeline_info);
	}
	// Standalone min-dilate H/V (intermediate -> dilTemp, adjustable radius in
	// half-res texels): same erode fragment as the triple, own pipeline (the triple
	// pipes bake their subpass index) under volume_raster_renderpass_mv
	// (R16F CLEAR MV; fullscreen coverage makes CLEAR deterministic).
	// The erode fragment is R16F-agnostic (MIN on .r). Standalone passes
	// (not the triple): nearest sampler, READ_ONLY
	// sources, fullscreen (every texel the fullscreen triangle writes
	// is defined).
	{
		auto dilate_frag = load_shader(device, "erode1d_mv.frag");
		vk::PipelineShaderStageCreateInfo dilate_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **blur_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **dilate_frag,
		                .pName = "main",
		        },
		};
		vk::PipelineVertexInputStateCreateInfo dilate_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo dilate_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo dilate_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo dilate_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo dilate_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState dilate_blend_attachment{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo dilate_blend{
		        .attachmentCount = 1,
		        .pAttachments = &dilate_blend_attachment,
		};
		vk::DynamicState dilate_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dilate_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = dilate_dynamics,
		};
		vk::GraphicsPipelineCreateInfo dilate_pipe_info{
		        .stageCount = 2,
		        .pStages = dilate_stages,
		        .pVertexInputState = &dilate_vertex_input,
		        .pInputAssemblyState = &dilate_input_assembly,
		        .pViewportState = &dilate_viewport_state,
		        .pRasterizationState = &dilate_rasterization,
		        .pMultisampleState = &dilate_multisample,
		        .pColorBlendState = &dilate_blend,
		        .pDynamicState = &dilate_dynamic,
		        .layout = *blur_layout,
		        .renderPass = *volume_raster_renderpass_mv,
		};
		// Subpass index is baked into the triple pipes, not this one:
		// the standalone erode fragment runs identically at subpass 0
		// for both the H and V passes, so one pipeline serves both.
		dilate_pipeline = vk::raii::Pipeline(device, *pipeline_cache(device), dilate_pipe_info);
	}
	// Homography rides push constants (named vec4 members, see header);
	// no buffers, views, copies or barriers per frame.
	// Linear sampler for the warp + add + compare sensor taps (warped
	// sensor D), same filtering as the blur passes: the bilinear
	// ramp is the outline pre-smoothing (see header), not a correctness
	// hazard.
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
	// Cutout punch: RGBA8 LOAD twin of the submit pass (re-begins the
	// finished swapchain framebuffer; loadOp is not part of render-pass
	// compatibility). The old verdict-MAX-merge pipes are retired with
	// MAX-into-seed (the compare CLEAR+writes the seed path directly).
	{
		uint32_t punch_mv_mask = 0b11, punch_mv_corr = 0b11;
		vk::RenderPassMultiviewCreateInfo punch_mv_rp_info{
		        .subpassCount = 1,
		        .pViewMasks = &punch_mv_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &punch_mv_corr,
		};
		vk::AttachmentDescription punch_attachment{
		        .format = format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eLoad,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference punch_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription punch_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &punch_ref,
		};
		vk::RenderPassCreateInfo punch_rp_info{
		        .pNext = &punch_mv_rp_info,
		        .attachmentCount = 1,
		        .pAttachments = &punch_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &punch_subpass,
		};
		punch_load_renderpass = vk::raii::RenderPass(device, punch_rp_info);
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

	// Replace path only (map change): make sure no in-flight frame
	// still reads the old buffers before replacing them. New keys have
	// no live buffers (in-flight draws miss the key and skip), so first
	// sight pays no waitIdle hitch mid-session. Rare either way;
	// render-thread only, no threading change.
	if (meshes.find(key) != meshes.end())
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
	mesh.vertex_count = (uint32_t)soup.vertices.size();
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
	if (vertex_bytes == 0)
		return;
	// Topology is fixed (same count as the uploaded soup): the index
	// buffer is never rewritten here. A count change is a caller bug —
	// fall back to set_mesh. Staging was sized for vertex+index at
	// upload, so vertex-only always fits unless the count changed.
	// The vertex_count check catches a smaller-but-fitting size that
	// would otherwise silently mismatch drawIndexed's stale index_count.
	if (mesh.vertex_count != (uint32_t)verts.size() or mesh.staging_size < vertex_bytes)
	{
		static bool warned = false;
		if (not warned)
		{
			warned = true;
			spdlog::warn("update_mesh_positions: size changed, use set_mesh");
		}
		return;
	}
	void * mapped = mesh.staging_memory.mapMemory(0, vertex_bytes);
	std::memcpy(mapped, verts.data(), vertex_bytes);
	mesh.staging_memory.unmapMemory();
	// index_bytes = 0: flush_upload skips the index copy/barrier and
	// reuses the live index buffer (no per-frame index re-upload).
	mesh.pending_upload = {vertex_bytes, 0};
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

void feather_mask_renderer::ensure_depth_targets(vk::Extent2D working, vk::Extent2D sensor)
{
	if (volume_targets_ready and depth_working_extent.width == working.width and
	    depth_working_extent.height == working.height and depth_sensor_extent.width == sensor.width and
	    depth_sensor_extent.height == sensor.height)
		return;

	// Rebuilt on extent change (video resize, depth-size change):
	// waitIdle retires in-flight frames using the old views, mirroring
	// the scene swapchain recreation discipline above.
	device.waitIdle();
	vk::Extent2D ve = working;
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
	auto make_linear_target = [&](blur_target & tgt, vk::Extent2D e, bool with_blur_fb) {
		std::tie(tgt.image, tgt.memory) = make_target_image(linear_depth_format, e);
		tgt.view_array = vk::raii::ImageView(device,
		        vk::ImageViewCreateInfo{
		                .image = *tgt.image,
		                .viewType = vk::ImageViewType::e2DArray,
		                .format = linear_depth_format,
		                .subresourceRange = {
		                        .aspectMask = vk::ImageAspectFlagBits::eColor,
		                        .baseMipLevel = 0,
		                        .levelCount = 1,
		                        .baseArrayLayer = 0,
		                        .layerCount = 2,
		                },
		        });
		{
			vk::ImageView raw_mv = *tgt.view_array;
			tgt.raster_fb_mv = vk::raii::Framebuffer(device,
			        vk::FramebufferCreateInfo{
			                .renderPass = *volume_raster_renderpass_mv,
			                .attachmentCount = 1,
			                .pAttachments = &raw_mv,
			                .width = e.width,
			                .height = e.height,
			                .layers = 2,
			        });
			if (with_blur_fb)
				tgt.blur_fb_mv = vk::raii::Framebuffer(device,
				        vk::FramebufferCreateInfo{
				                .renderPass = *volume_raster_renderpass_mv,
			                .attachmentCount = 1,
			                .pAttachments = &raw_mv,
			                .width = e.width,
			                .height = e.height,
			                .layers = 2,
			        });
		}
	};

	// Display-space targets (working res W = half of full):
	// warped-sensor canvas (R16F, D domain): rotation-warped runtime
	// sensor with skinned hand depth MINed over it in the same CLEAR
	// pass. Raster FB under volume_raster_renderpass_mv (hands pipe is
	// renderpass-compatible). Warp taps the runtime view directly (no
	// clean pass anymore); the image still stores D 0..1 (float now).
	make_linear_target(sensorW, ve, false);
	// Min-dilated layer (R16F linear meters at W): H renders into
	// blur_fb_mv, V back into raster_fb_mv (same dual-FB idiom as the
	// retired occluder dilate).
	make_linear_target(dilTemp, ve, true);

	// Binary verdict canvas (R8): sensor compare renders MV into
	// blur_fb_mv under the shared CLEAR R8 pass; the alpha build samples
	// the array view with NEAREST.
	std::tie(volume_bin.image, volume_bin.memory) = make_target_image(blur_format, ve);
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
	                .renderPass = *compare_renderpass_mv,
	                .attachmentCount = 1,
	                .pAttachments = &raw_bin_array,
	                .width = ve.width,
	                .height = ve.height,
	                .layers = 2,
	        });

	// Reversed-depth scratch (R16F linear, working res): the group's own
	// reversed draws, nearest (far clear, MIN blend, cull-nothing),
	// sampled by the alpha build. Always re-rasterized (CLEAR covers the
	// no-draws case).
	make_linear_target(rev_depth, ve, false);

	// Closed unit-box mesh ([-1,1]^3, outward-CCW, non-indexed), baked
	// once: the bbox draws map it onto the volume bounds (scene MVPs).
	// Deleted-object soups are never trusted for the far-wall trick
	// (arbitrary glTF, not guaranteed closed) — hence this dedicated
	// closed box. Creation-only (host-visible VMA, no waitIdle needed).
	if (not bbox_verts)
	{
		// Per-face outward-CCW quads (verified normals: +X/-X/+Y/-Y/+Z/-Z).
		const float v[36][3] = {
		        {1, -1, 1}, {1, -1, -1}, {1, 1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}, // +X
		        {-1, -1, -1}, {-1, -1, 1}, {-1, 1, 1}, {-1, -1, -1}, {-1, 1, 1}, {-1, 1, -1}, // -X
		        {-1, 1, 1}, {1, 1, 1}, {1, 1, -1}, {-1, 1, 1}, {1, 1, -1}, {-1, 1, -1}, // +Y
		        {-1, -1, -1}, {1, -1, -1}, {1, -1, 1}, {-1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, // -Y
		        {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, -1, 1}, {1, 1, 1}, {-1, 1, 1}, // +Z
		        {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1}, // -Z
		};
		bbox_verts = buffer_allocation{
		        device,
		        vk::BufferCreateInfo{
		                .size = sizeof(v),
		                .usage = vk::BufferUsageFlagBits::eVertexBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "feather_mask bbox",
		};
		std::memcpy(bbox_verts.map(), v, sizeof(v));
		bbox_verts.unmap();
	}

	depth_working_extent = working;
	depth_sensor_extent = sensor;
	volume_targets_ready = true;
	spdlog::info("Feather mask depth targets: work {}x{} (R16F linear layer chain)", ve.width, ve.height);
}

void feather_mask_renderer::flush_upload(vk::raii::CommandBuffer & cmd, mesh_buffers & mesh)
{
	if (not mesh.pending_upload)
		return;
	auto [vertex_bytes, index_bytes] = *mesh.pending_upload;
	mesh.pending_upload.reset();

	// Either span may be zero (update_mesh_positions rewrites vertices
	// only): skip zero-size copies — a zero copy is a validation error.
	std::vector<vk::BufferMemoryBarrier> barriers;
	barriers.reserve(2);
	if (vertex_bytes > 0)
	{
		vk::BufferCopy vertex_copy{
		        .srcOffset = 0,
		        .dstOffset = 0,
		        .size = vertex_bytes,
		};
		cmd.copyBuffer(*mesh.staging_buffer, *mesh.vertex_buffer, vertex_copy);
		barriers.push_back(vk::BufferMemoryBarrier{
		        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
		        .dstAccessMask = vk::AccessFlagBits::eVertexAttributeRead,
		        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .buffer = *mesh.vertex_buffer,
		        .offset = 0,
		        .size = vertex_bytes,
		});
	}
	if (index_bytes > 0)
	{
		vk::BufferCopy index_copy{
		        .srcOffset = vertex_bytes,
		        .dstOffset = 0,
		        .size = index_bytes,
		};
		cmd.copyBuffer(*mesh.staging_buffer, *mesh.index_buffer, index_copy);
		barriers.push_back(vk::BufferMemoryBarrier{
		        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
		        .dstAccessMask = vk::AccessFlagBits::eIndexRead,
		        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .buffer = *mesh.index_buffer,
		        .offset = 0,
		        .size = index_bytes,
		});
	}
	if (barriers.empty())
		return;
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
	double ms_warp = 0, ms_geom = 0, ms_alpha = 0, ms_dilate = 0;
	double ms_compare = 0, ms_add = 0;
	// Depth geometry engages when the group carries the volume bbox with
	// a borrowed layer target (tier-1 only, enforced upstream: W = half
	// of full == the tier-1 seed size). Runs even if the sensor is dead
	// (hidden + bbox still gate the global verdict). The sensor section
	// additionally needs a valid sensor view + extent.
	bool geometry_ok = volume && !volume->bbox_draws.empty() && volume->layer_image != vk::Image{} &&
	                   volume->layer_view != vk::ImageView{} &&
	                   volume->layer_extent.width > 0 && volume->layer_extent.height > 0 &&
	                   tier_for_feather(feather_px) == 1; // belt-and-braces: upstream only sends tier-1
	bool sensor_ok = geometry_ok && volume->sensor_view != vk::ImageView{} && volume->sensor_extent.width > 0 &&
	                 volume->sensor_extent.height > 0;
	// Display working res W: level 1 of full (half), floored like the
	// tier intermediates. Identical to the tier-1 submit size by
	// construction, so seed and display chain always agree. Must match
	// the merge-owned layer extent (loud one-shot warn on mismatch: the
	// verdict would compare across sizes, still legal UV-wise, but the
	// seed agreement below would break).
	vk::Extent2D workE = level_extent(extent, 1);
	vk::Extent2D sensorE = sensor_ok ? volume->sensor_extent : vk::Extent2D{0, 0};
	if (geometry_ok)
	{
		if (volume->layer_extent.width != workE.width or volume->layer_extent.height != workE.height)
		{
			static bool warned = false;
			if (not warned)
			{
				warned = true;
				spdlog::error("Depth layer extent {}x{} != working {}x{}: verdict/seed disagree, geometry off",
				              volume->layer_extent.width, volume->layer_extent.height, workE.width, workE.height);
			}
			geometry_ok = false;
			sensor_ok = false;
		}
	}
	if (geometry_ok)
	        ensure_depth_targets(workE, sensorE);
	float box_frac = 1;
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
			cpu_stats->warp_ms = ms_warp;
			cpu_stats->geom_ms = ms_geom;
			cpu_stats->alpha_ms = ms_alpha;
			cpu_stats->dilate_ms = ms_dilate;
			cpu_stats->compare_ms = ms_compare;
			cpu_stats->add_ms = ms_add;
			cpu_stats->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total0).count();
			cpu_stats->tier = tier_out;
			cpu_stats->draws = n_draws;
			cpu_stats->box_frac = box_frac;
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

	// Merged-triple framebuffer for this renderer's (src, tmp, extent)
	// triple (fixed per group: tier is fixed per feather). Creation-only
	// on first record — the pair never changes mid-stream (extent
	// changes rebuild the images themselves, hence new handles), and the
	// extent is keyed explicitly so a same-handle size change still
	// rebuilds. Both layers, triple renderpass.
	auto triple_fb_for = [&](vk::Image src_img, vk::Image tmp_img, vk::ImageView src_view, vk::ImageView tmp_view,
	                          vk::Extent2D e) -> vk::Framebuffer {
		if (triple_src != src_img or triple_tmp != tmp_img or triple_extent.width != e.width or
		    triple_extent.height != e.height)
		{
			triple_src = src_img;
			triple_tmp = tmp_img;
			triple_extent = e;
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


	// A geometry-only group has no silhouette draws: it still runs the
	// tier path (cleared seed, then the depth chain below sets it).
	// Same for a reversed-only cut (positives always present in practice,
	// but the gate mirrors the geometry one for safety).
	if (not rasterize or (not has_positive and not geometry_ok))
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
		// No meshes and no depth geometry: nothing to raster, but cutouts
		// still punch (the debug-only group has no meshes by design).
		// Tier branches below run over empty draws (clear passes), then
		// the punch stamps.
		if (meshes.empty() and cutouts.empty() and not geometry_ok)
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
		// tier==0 with no volume draws. A future tier-0 consumer of
		// pass_scissor must lift this gate (unconsumed, eye_box stays
		// empty and pass_scissor falls back to fullscreen — fail-safe).
		if (tier != 0 or geometry_ok)
		{
			for (const auto & d: draws)
				accumulate_box(d);
			// Depth chain canvases feed the same erode/blur chain, so
			// the scissor box must cover them too (viewer-inside
			// volumes usually trip the behind-camera fallback to
			// fullscreen, which is correct).
			if (geometry_ok)
			{
				for (const auto & d: volume->hidden_draws)
					accumulate_box(d);
				for (const auto & d: volume->hand_draws)
					accumulate_box(d);
				// bbox draws carry MVP pairs only (no mesh key): project
				// the 8 unit-box corners directly.
				for (const auto & mvp: volume->bbox_draws)
				{
					for (int eye = 0; eye < 2; ++eye)
					{
						for (int corner = 0; corner < 8; ++corner)
						{
							glm::vec4 q((corner & 1) ? 1.f : -1.f, (corner & 2) ? 1.f : -1.f,
							            (corner & 4) ? 1.f : -1.f, 1);
							glm::vec4 c = mvp[eye] * q;
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
				}
			}
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
			// Float-domain clamp first: NDC from a near-behind-w corner
			// or a huge margin can push fx*/fy* far outside int32 range,
			// and an out-of-range float->int cast is UB (x86 resolves to
			// INT_MIN => wrongful skip => under-reveal, fail-unsafe).
			// Clamped floats cast exactly, like field_union above
			// (floor the mins / ceil the maxes to stay conservative).
			float qx0 = std::min(std::max(fx0 * sx - 2.f, 0.f), (float)pe.width);
			float qy0 = std::min(std::max(fy0 * sy - 2.f, 0.f), (float)pe.height);
			float qx1 = std::min(std::max(fx1 * sx + 2.f, 0.f), (float)pe.width);
			float qy1 = std::min(std::max(fy1 * sy + 2.f, 0.f), (float)pe.height);
			int32_t px0 = (int32_t)std::floor(qx0);
			int32_t py0 = (int32_t)std::floor(qy0);
			int32_t px1 = (int32_t)std::ceil(qx1);
			int32_t py1 = (int32_t)std::ceil(qy1);
			if (px1 <= px0 or py1 <= py0)
				return vk::Rect2D{{0, 0}, pe};
			box_frac = std::min(box_frac, (float)(px1 - px0) * (float)(py1 - py0) / ((float)pe.width * (float)pe.height));
			return vk::Rect2D{{px0, py0}, {(uint32_t)(px1 - px0), (uint32_t)(py1 - py0)}};
		};

		// Raster happens inside each path's own eye loop below (tier 1
		// into half-res A, tiered as a level seed, tier 0 direct into the
		// swapchain image).
		tier_out = tier;

		// Common depth layer (tier-1 depth group): stage-0 windows were
		// MINed into the borrowed intermediate by the merge record_base
		// earlier in this command buffer; the sensor stage MINs hidden +
		// bbox, then revealed sensor depths. Geometry runs even when the
		// sensor is dead (hidden + bbox still gate the global verdict);
		// the warp/compare/add section additionally needs sensor_ok.
		// GPU stamps on the shared pool: this record writes slots 2..5
		// (0/1 base and 8/6/7/9 merge live on the same pool; write order
		// is 0,1,...5 then 8,6,7,9 — the scene maps spans accordingly).
		// Null pool = no stamping (zero GPU cost). The pool is reset by
		// the scene once per frame (before record_base), never here (a
		// reset would wipe slots 0/1).
		bool volume_ts = volume && volume->timestamp_pool != vk::QueryPool{};
		auto volume_stamp = [&](uint32_t q) {
			if (volume_ts && volume)
				cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, volume->timestamp_pool, q);
		};
		bool warp_desc_updated = false;
		bool add_desc_updated = false;
		// Re-bake the borrowed-layer framebuffer when the merge rebuilt
		// (handle change) or W changed. Creation-only, no idle: the
		// rebuild waitIdles upstream and handle+extent keying means a
		// stale entry can never alias a live image.
		if (geometry_ok and (layer_fb_image != volume->layer_image or layer_fb_extent.width != workE.width or
		                     layer_fb_extent.height != workE.height))
		{
			vk::ImageView layer_view = volume->layer_view;
			layer_fb = vk::raii::Framebuffer(device,
			        vk::FramebufferCreateInfo{
			                .renderPass = *geom_load_renderpass_mv,
			                .attachmentCount = 1,
			                .pAttachments = &layer_view,
			                .width = workE.width,
			                .height = workE.height,
			                .layers = 2,
			        });
			layer_fb_image = volume->layer_image;
			layer_fb_extent = workE;
		}
		// Geometry section (runs when geometry_ok even if the sensor is
		// dead): hidden (CULL_BACK nearest linear) + bbox (CULL_FRONT
		// far-wall linear on the closed unit box) MIN into the borrowed
		// intermediate under LOAD. The merge base wrote earlier in this
		// CB, so a production barrier separates them (same queue).
		auto run_geometry = [&]() {
			section_clock geom_clk(&ms_geom);
			for (const auto & d: volume->hidden_draws)
			{
				auto mit = meshes.find(d.mesh);
				if (mit != meshes.end())
					flush_upload(cmd, mit->second);
			}
			// Base color write out, geometry read+write in (MIN blend
			// reads dst). Layout stays OPTIMAL (LOAD renderpass).
			transfer_barrier(volume->layer_image,
			                 vk::ImageLayout::eShaderReadOnlyOptimal,
			                 vk::ImageLayout::eColorAttachmentOptimal,
			                 vk::AccessFlagBits::eShaderRead,
			                 vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
			                 vk::PipelineStageFlagBits::eFragmentShader,
			                 vk::PipelineStageFlagBits::eColorAttachmentOutput);
			vk::RenderPassBeginInfo begin_geom{
			        .renderPass = *geom_load_renderpass_mv,
			        .framebuffer = *layer_fb,
			        .renderArea = {.offset = {0, 0}, .extent = workE},
			};
			cmd.beginRenderPass(begin_geom, vk::SubpassContents::eInline);
			set_full_viewport(workE);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *geom_back_pipeline_mv);
			for (const auto & d: volume->hidden_draws)
			{
				if (d.reverse)
					continue; // reversed cuts stay per-group (rev raster + alpha below)
				auto mit = meshes.find(d.mesh);
				if (mit == meshes.end() or mit->second.index_count == 0)
					continue;
				cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
				cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
				cmd.pushConstants<geom_depth_push>(*geom_raster_layout_mv, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
				                                  geom_depth_push{.mvp = {{d.mvp[0], d.mvp[1]}}});
				cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
				++n_draws;
			}
			// Bbox far wall: the scene MVPs map the closed unit box
			// onto each volume bounds; CULL_FRONT keeps the far wall
			// (dedicated closed box — object soups are arbitrary glTF,
			// never trusted for this trick).
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *geom_front_pipeline_mv);
			cmd.bindVertexBuffers(0, (vk::Buffer)bbox_verts, (vk::DeviceSize)0);
			for (const auto & mvp: volume->bbox_draws)
			{
				cmd.pushConstants<geom_depth_push>(*geom_raster_layout_mv, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
				                                  geom_depth_push{.mvp = {{mvp[0], mvp[1]}}});
				cmd.draw(36, 1, 0, 0);
				++n_draws;
			}
			cmd.endRenderPass();
			make_readable(volume->layer_image);
			volume_stamp(2);
		};
		// Standalone min-dilate H+V at W (adjustable radius in half-res
		// texels): intermediate -> dilTemp. Fullscreen (every texel
		// written, so CLEAR is deterministic); nearest sampling like the
		// erode passes. The erode fragment is R16F-agnostic (MIN on .r).
		auto run_layer_dilate = [&]() {
			section_clock dil_clk(&ms_dilate);
			float dil_r = std::clamp(volume->dilate_texels, 0.f, 8.f);
			update_source(volume->layer_view, 9, true);
			vk::ClearValue far{};
			const float farv[4] = {layer_far_clear, layer_far_clear, layer_far_clear, layer_far_clear};
			std::memcpy(far.color.float32, farv, sizeof(farv));
			vk::RenderPassBeginInfo begin_dh{
			        .renderPass = *volume_raster_renderpass_mv,
			        .framebuffer = *dilTemp.blur_fb_mv,
			        .renderArea = {.offset = {0, 0}, .extent = workE},
			        .clearValueCount = 1,
			        .pClearValues = &far,
			};
			cmd.beginRenderPass(begin_dh, vk::SubpassContents::eInline);
			set_full_viewport(workE);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *dilate_pipeline);
			std::array<vk::DescriptorSet, 1> sets_dh{*descriptor_sets[9]};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_dh, no_offsets);
			erode_push dh{{1.f / workE.width, 1.f / workE.height}, {1, 0}, dil_r, 0};
			cmd.pushConstants<erode_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, dh);
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
			make_readable(*dilTemp.image);
			transfer_barrier(*dilTemp.image,
			                 vk::ImageLayout::eShaderReadOnlyOptimal,
			                 vk::ImageLayout::eColorAttachmentOptimal,
			                 vk::AccessFlagBits::eShaderRead,
			                 vk::AccessFlagBits::eColorAttachmentWrite,
			                 vk::PipelineStageFlagBits::eFragmentShader,
			                 vk::PipelineStageFlagBits::eColorAttachmentOutput);
			update_source(*dilTemp.view_array, 10, true);
			vk::RenderPassBeginInfo begin_dv{
			        .renderPass = *volume_raster_renderpass_mv,
			        .framebuffer = *dilTemp.raster_fb_mv,
			        .renderArea = {.offset = {0, 0}, .extent = workE},
			        .clearValueCount = 1,
			        .pClearValues = &far,
			};
			cmd.beginRenderPass(begin_dv, vk::SubpassContents::eInline);
			set_full_viewport(workE);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *dilate_pipeline);
			std::array<vk::DescriptorSet, 1> sets_dv{*descriptor_sets[10]};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_dv, no_offsets);
			erode_push dv{{1.f / workE.width, 1.f / workE.height}, {0, 1}, dil_r, 0};
			cmd.pushConstants<erode_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, dv);
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
			make_readable(*dilTemp.image);
			volume_stamp(3);
		};
		// Sensor warp + hands MIN in one CLEAR pass at W (far clear bakes
		// off-canvas warp to far: fail closed). Hands use the D-domain
		// exact (MIN, cull-nothing) pipe: renderpass-compatible, no new
		// pipe. The warp taps the runtime sensor view directly (no
		// clean pass anymore); the canvas stores D 0..1 (float now).
		// Skipped without a sensor (compare CLEARs neutral, add is a
		// no-op then). Warp cost lands in the compare GPU span (3->4).
		auto run_warp_hands = [&]() {
			if (!sensor_ok)
				return;
			section_clock warp_clk(&ms_warp);
			for (const auto & d: volume->hand_draws)
			{
				auto mit = meshes.find(d.mesh);
				if (mit != meshes.end())
					flush_upload(cmd, mit->second);
			}
			if (!warp_desc_updated)
			{
				warp_desc_updated = true;
				vk::DescriptorImageInfo sensor_info{
				        .sampler = *volume_sampler_linear,
				        .imageView = volume->sensor_view,
				        // The runtime hands the acquired image ready to
				        // sample (acquire is the sync point).
				        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
				};
				std::array<vk::CopyDescriptorSet, 0> no_warp_copies{};
				std::array<vk::WriteDescriptorSet, 1> warp_writes = {{
				        {
				                .dstSet = *warp_set,
				                .dstBinding = 0,
				                .descriptorCount = 1,
				                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
				                .pImageInfo = &sensor_info,
				        },
				}};
				device.updateDescriptorSets(warp_writes, no_warp_copies);
			}
			vk::ClearValue far1{};
			const float far1v[4] = {1.0f, 1.0f, 1.0f, 1.0f};
			std::memcpy(far1.color.float32, far1v, sizeof(far1v));
			vk::RenderPassBeginInfo begin_warp{
			        .renderPass = *volume_raster_renderpass_mv,
			        .framebuffer = *sensorW.raster_fb_mv,
			        .renderArea = {.offset = {0, 0}, .extent = workE},
			        .clearValueCount = 1,
			        .pClearValues = &far1,
			};
			cmd.beginRenderPass(begin_warp, vk::SubpassContents::eInline);
			set_full_viewport(workE);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *warp_pipeline_mv);
			std::array<vk::DescriptorSet, 1> warp_sets{*warp_set};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *warp_pipe_layout, 0, warp_sets, no_offsets);
			warp_push wp{};
			for (int i = 0; i < 6; ++i)
				wp.col[i] = volume->warp_cols[i];
			cmd.pushConstants<warp_push>(*warp_pipe_layout, vk::ShaderStageFlagBits::eFragment, 0, wp);
			cmd.draw(3, 1, 0, 0);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *exact_raster_pipeline_mv);
			for (const auto & d: volume->hand_draws)
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
			make_readable(*sensorW.image);
		};
		// Sensor compare (warped D vs dilated layer, bias on the layer):
		// CLEAR+writes the binary verdict canvas (no MAX anywhere —
		// the old verdict-MAX-merge is retired). Without a sensor the
		// canvas CLEARs to 0 (neutral: the seed stays empty, fail
		// closed) without sampling anything. Feeds the depth-group seed
		// only — never the global path.
		auto run_compare = [&]() {
			vk::RenderPassBeginInfo begin_cmp{
			        .renderPass = *compare_renderpass_mv,
			        .framebuffer = *volume_bin.blur_fb_mv,
			        .renderArea = {.offset = {0, 0}, .extent = workE},
			        .clearValueCount = 1,
			        .pClearValues = &clear,
			};
			if (!sensor_ok)
			{
				cmd.beginRenderPass(begin_cmp, vk::SubpassContents::eInline);
				cmd.endRenderPass();
				make_readable(*volume_bin.image);
				volume_stamp(4);
				return;
			}
			section_clock compare_clk(&ms_compare);
			vk::DescriptorImageInfo sensor_info{
			        .sampler = *volume_sampler_linear,
			        .imageView = *sensorW.view_array,
			        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			};
			vk::DescriptorImageInfo layer_info{
			        .sampler = *sampler_nearest,
			        .imageView = volume->layer_view,
			        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			};
			std::array<vk::CopyDescriptorSet, 0> no_compare_copies{};
			std::array<vk::WriteDescriptorSet, 2> compare_writes = {{
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
			                .pImageInfo = &layer_info,
			        },
			}};
			device.updateDescriptorSets(compare_writes, no_compare_copies);
			cmd.beginRenderPass(begin_cmp, vk::SubpassContents::eInline);
			set_full_viewport(workE);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *compare_pipeline_mv);
			std::array<vk::DescriptorSet, 1> cmp_sets{*compare_set};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *compare_pipe_layout, 0, cmp_sets, no_offsets);
			compare_push cp{};
			cp.near_far_bias = glm::vec4(volume->near_z, volume->far_z, volume->bias, 0.f);
			cmd.pushConstants<compare_push>(*compare_pipe_layout, vk::ShaderStageFlagBits::eFragment, 0, cp);
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
			make_readable(*volume_bin.image);
			volume_stamp(4);
		};
		// Layer add: linearize the warped sensor sample, MIN-blend LOAD
		// back into the raw intermediate. UNGATED by design: where the
		// compare test failed sens >= layer, MIN is a no-op there. The
		// global verdict then sees sensor-revealed depths too.
		auto run_add = [&]() {
			if (!sensor_ok)
			{
				volume_stamp(5);
				return;
			}
			section_clock add_clk(&ms_add);
			if (!add_desc_updated)
			{
				add_desc_updated = true;
				vk::DescriptorImageInfo add_info{
				        .sampler = *volume_sampler_linear,
				        .imageView = *sensorW.view_array,
				        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
				};
				std::array<vk::CopyDescriptorSet, 0> no_add_copies{};
				std::array<vk::WriteDescriptorSet, 1> add_writes = {{
				        {
				                .dstSet = *add_set,
				                .dstBinding = 0,
				                .descriptorCount = 1,
				                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
				                .pImageInfo = &add_info,
				        },
				}};
				device.updateDescriptorSets(add_writes, no_add_copies);
			}
			transfer_barrier(volume->layer_image,
			                 vk::ImageLayout::eShaderReadOnlyOptimal,
			                 vk::ImageLayout::eColorAttachmentOptimal,
			                 vk::AccessFlagBits::eShaderRead,
			                 vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
			                 vk::PipelineStageFlagBits::eFragmentShader,
			                 vk::PipelineStageFlagBits::eColorAttachmentOutput);
			vk::RenderPassBeginInfo begin_add{
			        .renderPass = *geom_load_renderpass_mv,
			        .framebuffer = *layer_fb,
			        .renderArea = {.offset = {0, 0}, .extent = workE},
			};
			cmd.beginRenderPass(begin_add, vk::SubpassContents::eInline);
			set_full_viewport(workE);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *add_pipeline_mv);
			std::array<vk::DescriptorSet, 1> add_sets{*add_set};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *add_pipe_layout, 0, add_sets, no_offsets);
			cmd.pushConstants<layer_add_push>(*add_pipe_layout, vk::ShaderStageFlagBits::eFragment, 0,
			                                  layer_add_push{.near_z = volume->near_z, .far_z = volume->far_z});
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
			make_readable(volume->layer_image);
			volume_stamp(5);
		};
		// Display-space reversed raster (rev linear at W, nearest): the
		// group's own reversed draws (per-group cuts stay per-group),
		// cull-nothing like the retired revD. Feeds the alpha cut.
		// Always re-rasterized (CLEAR covers the no-draws case).
		auto run_rev = [&]() {
			section_clock rev_clk(&ms_alpha);
			vk::ClearValue far{};
			const float farv[4] = {layer_far_clear, layer_far_clear, layer_far_clear, layer_far_clear};
			std::memcpy(far.color.float32, farv, sizeof(farv));
			vk::RenderPassBeginInfo begin_rev{
			        .renderPass = *volume_raster_renderpass_mv,
			        .framebuffer = *rev_depth.raster_fb_mv,
			        .renderArea = {.offset = {0, 0}, .extent = workE},
			        .clearValueCount = 1,
			        .pClearValues = &far,
			};
			cmd.beginRenderPass(begin_rev, vk::SubpassContents::eInline);
			set_full_viewport(workE);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *geom_none_pipeline_mv);
			for (const auto & d: draws)
			{
				if (!d.reverse)
					continue;
				auto mit = meshes.find(d.mesh);
				if (mit == meshes.end() or mit->second.index_count == 0)
					continue;
				cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
				cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
				cmd.pushConstants<geom_depth_push>(*geom_raster_layout_mv, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
				                                  geom_depth_push{.mvp = {{d.mvp[0], d.mvp[1]}}});
				cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
				++n_draws;
			}
			cmd.endRenderPass();
			make_readable(*rev_depth.image);
		};
		// Alpha build: sensor verdict minus per-group rev cut into the
		// tier seed (target_a, CLEAR). Stays OPTIMAL for the erode/blur
		// triple below (same state as the legacy raster path leaves).
		auto build_alpha = [&]() {
			section_clock alpha_clk(&ms_alpha);
			vk::DescriptorImageInfo win_info{
			        .sampler = *sampler_nearest,
			        .imageView = *volume_bin.view_array,
			        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			};
			vk::DescriptorImageInfo rev_info{
			        .sampler = *sampler_nearest,
			        .imageView = *rev_depth.view_array,
			        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			};
			std::array<vk::CopyDescriptorSet, 0> no_alpha_copies{};
			std::array<vk::WriteDescriptorSet, 2> alpha_writes = {{
			        {
			                .dstSet = *alpha_set,
			                .dstBinding = 0,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			                .pImageInfo = &win_info,
			        },
			        {
			                .dstSet = *alpha_set,
			                .dstBinding = 1,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			                .pImageInfo = &rev_info,
			        },
			}};
			device.updateDescriptorSets(alpha_writes, no_alpha_copies);
			vk::ClearValue azero{};
			azero.color.float32.fill(0);
			vk::RenderPassBeginInfo begin_alpha{
			        .renderPass = *compare_renderpass_mv,
			        .framebuffer = *target_a.blur_fb_mv,
			        .renderArea = {.offset = {0, 0}, .extent = workE},
			        .clearValueCount = 1,
			        .pClearValues = &azero,
			};
			cmd.beginRenderPass(begin_alpha, vk::SubpassContents::eInline);
			set_full_viewport(workE);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *alpha_pipeline_mv);
			std::array<vk::DescriptorSet, 1> alpha_sets{*alpha_set};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *alpha_pipe_layout, 0, alpha_sets, no_offsets);
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
		};
		// Full depth chain for the tier-1 depth group. Stamps: geom (2),
		// dilate (3), compare (4, warp folds in), add (5). Rev + alpha
		// are CPU-timed only (no GPU slots left on the shared pool).
		auto run_depth = [&]() {
			run_geometry();
			run_layer_dilate();
			run_warp_hands();
			run_compare();
			run_add();
			run_rev();
			build_alpha();
			make_readable(*target_a.image);
		};

		if (tier == 0)
		{
			section_clock raster_clk(&ms_raster);
			// Hard edge: rasterize silhouettes straight into the swapchain
			// image. Exact, one multiview raster pass, no blur passes.
			// (Tier-0 groups submit full, so out matches working here.)
			// Depth never runs here (tier-1-only by construction):
			// legacy silhouette raster only.
			raster_silhouette_mv({*it->second.framebuffer_mv, *it->second.depth_framebuffer_mv, *pipeline_mv, *depth_pipeline_mv, *reverse_pipeline_mv, *renderpass_mv, *depth_renderpass_mv}, out_extent, has_reversed);
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
		// Depth path replaces the legacy silhouette raster: the sensor
		// verdict minus the per-group rev cut is the seed the shared
		// erode/blur chain feathers. Legacy path runs otherwise.
		if (geometry_ok)
			run_depth();
		else
		{
			section_clock raster_clk(&ms_raster);
			raster_silhouette_mv({*target_a.raster_fb_mv, *target_a.raster_depth_fb_mv, *raster_r8_pipeline_mv, *raster_r8_depth_pipeline_mv, *raster_r8_reverse_pipeline_mv, *raster_r8_renderpass_mv, *raster_r8_depth_renderpass_mv}, half_extent, has_reversed);
			make_readable(*target_a.image);
		}
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

			// Depth never runs in tiered groups (tier-1-only by
			// construction): legacy seed raster above stands as-is.
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
		        .renderPass = *punch_load_renderpass,
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
