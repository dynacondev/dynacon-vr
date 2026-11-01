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
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <spdlog/spdlog.h>
#include <stdexcept>

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

// Silhouette raster push block. Mirrors mask.vert/frag.glsl PushConstants:
// mvp plus the first-acquisition fade opacity. The blur chain is linear in
// alpha, so one raster-alpha scale fades interior and feather band alike.
struct raster_push
{
	glm::mat4 mvp;
	float opacity = 1;
};

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

// Rotation-reprojection UBO mirror (fragment only): per-eye homography
// as 6 vec4 columns. std140 vec4 arrays are 16-byte strided on both
// sides, so the C++ mirror packs identically (96 bytes, asserted below
// next to the upload).

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
	auto vert = load_shader(device, "mask.vert");
	auto frag = load_shader(device, "mask.frag");

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

	vk::PushConstantRange push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = sizeof(raster_push),
	};
	vk::PipelineLayoutCreateInfo layout_info{
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &push_range,
	};
	pipeline_layout = vk::raii::PipelineLayout(device, layout_info);

	vk::AttachmentDescription attachment{
	        .format = format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eGeneral,
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
	vk::RenderPassCreateInfo renderpass_info{
	        .attachmentCount = 1,
	        .pAttachments = &attachment,
	        .subpassCount = 1,
	        .pSubpasses = &subpass,
	};
	renderpass = vk::raii::RenderPass(device, renderpass_info);

	// R8 variant of the raster pass for the single-channel intermediates
	// (same CLEAR/STORE shape, R8 attachment).
	vk::AttachmentDescription raster_r8_attachment{
	        .format = blur_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        // Intermediates only (never submitted): optimal layout keeps
	        // Adreno UBWC compression live. Submitted images stay GENERAL.
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference raster_r8_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription raster_r8_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &raster_r8_ref,
	};
	vk::RenderPassCreateInfo raster_r8_rp_info{
	        .attachmentCount = 1,
	        .pAttachments = &raster_r8_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &raster_r8_subpass,
	};
	raster_r8_renderpass = vk::raii::RenderPass(device, raster_r8_rp_info);

	// Cutout pass: same shape as the raster pass but LOAD, so the
	// finished (blurred) swapchain image can be re-begun and stamped.
	// Framebuffer-compatible with both passes above by construction.
	vk::AttachmentDescription cutout_attachment{
	        .format = format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eLoad,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eGeneral,
	        .finalLayout = vk::ImageLayout::eGeneral,
	};
	vk::AttachmentReference cutout_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription cutout_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &cutout_ref,
	};
	vk::RenderPassCreateInfo cutout_rp_info{
	        .attachmentCount = 1,
	        .pAttachments = &cutout_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &cutout_subpass,
	};
	cutout_renderpass = vk::raii::RenderPass(device, cutout_rp_info);

	vk::GraphicsPipelineCreateInfo pipeline_info{
	        .stageCount = 2,
	        .pStages = stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *pipeline_layout,
	        .renderPass = *renderpass,
	};
	pipeline = vk::raii::Pipeline(device, *pipeline_cache(device), pipeline_info);

	// R8 raster pipeline: same shaders (mask.frag replicates opacity to
	// all channels) and layout, R8 render pass.
	vk::GraphicsPipelineCreateInfo raster_r8_pipeline_info{
	        .stageCount = 2,
	        .pStages = stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *pipeline_layout,
	        .renderPass = *raster_r8_renderpass,
	};
	raster_r8_pipeline = vk::raii::Pipeline(device, *pipeline_cache(device), raster_r8_pipeline_info);

	// Depth-tested raster variants for reversed draws: a D16 depth
	// attachment joins the color attachment (same color formats as above,
	// so two renderpasses). Depth is test-only (never sampled) and always
	// cleared: no layout history is ever assumed.
	auto make_depth_renderpass = [&](vk::Format color_format, vk::ImageLayout color_final = vk::ImageLayout::eGeneral) {
		vk::AttachmentDescription attachments[2] = {
		        {
		                .format = color_format,
		                .samples = vk::SampleCountFlagBits::e1,
		                .loadOp = vk::AttachmentLoadOp::eClear,
		                .storeOp = vk::AttachmentStoreOp::eStore,
		                .initialLayout = vk::ImageLayout::eUndefined,
		                .finalLayout = color_final,
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
		vk::AttachmentReference color_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference depth_ref{
		        .attachment = 1,
		        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::SubpassDescription depth_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &color_ref,
		        .pDepthStencilAttachment = &depth_ref,
		};
		vk::RenderPassCreateInfo depth_rp_info{
		        .attachmentCount = 2,
		        .pAttachments = attachments,
		        .subpassCount = 1,
		        .pSubpasses = &depth_subpass,
		};
		return vk::raii::RenderPass(device, depth_rp_info);
	};
	depth_renderpass = make_depth_renderpass(format);
	raster_r8_depth_renderpass = make_depth_renderpass(blur_format, vk::ImageLayout::eColorAttachmentOptimal);

	// Same vertex input/assembly/viewport/rasterization/multisample/
	// dynamic state as the plain raster pipes (reused locals); only blend
	// and depth-stencil state vary. Positives keep overwrite + depth write;
	// reversed draws MIN-blend (1 - fade) with a read-only depth test, so a
	// negative cuts only where it is nearer than the positives covering
	// that pixel (MIN/MAX blend ops are format-independent for UNORM).
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
	auto make_raster_variant = [&](vk::raii::RenderPass & rp, vk::PipelineColorBlendStateCreateInfo & bl,
	                               vk::PipelineDepthStencilStateCreateInfo & ds) {
		vk::GraphicsPipelineCreateInfo info{
		        .stageCount = 2,
		        .pStages = stages,
		        .pVertexInputState = &vertex_input,
		        .pInputAssemblyState = &input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pDepthStencilState = &ds,
		        .pColorBlendState = &bl,
		        .pDynamicState = &dynamic,
		        .layout = *pipeline_layout,
		        .renderPass = *rp,
		};
		return vk::raii::Pipeline(device, *pipeline_cache(device), info);
	};
	depth_pipeline = make_raster_variant(depth_renderpass, blend, depth_write);
	reverse_pipeline = make_raster_variant(depth_renderpass, min_blend, depth_test);
	raster_r8_depth_pipeline = make_raster_variant(raster_r8_depth_renderpass, blend, depth_write);
	raster_r8_reverse_pipeline = make_raster_variant(raster_r8_depth_renderpass, min_blend, depth_test);

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
	        .finalLayout = vk::ImageLayout::eGeneral,
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
	        .descriptorCount = 14,
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .maxSets = 14,
	        .poolSizeCount = 1,
	        .pPoolSizes = &pool_size,
	};
	descriptor_pool = vk::raii::DescriptorPool(device, pool_info);
	for (int i = 0; i < 14; ++i)
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
	static_assert(sizeof(volume_push) == 72);
	static_assert(sizeof(volume_homography) == 128);
	{
		auto volume_vert = load_shader(device, "volume.vert");
		auto volume_frag = load_shader(device, "volume_raster.frag");
		vk::PipelineShaderStageCreateInfo volume_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **volume_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **volume_frag,
		                .pName = "main",
		        },
		};
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
		vk::PushConstantRange volume_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(volume_push),
		};
		vk::PipelineLayoutCreateInfo volume_layout_info{
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &volume_push_range,
		};
		volume_raster_layout = vk::raii::PipelineLayout(device, volume_layout_info);
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
		vk::RenderPassCreateInfo volume_raster_rp_info{
		        .attachmentCount = 1,
		        .pAttachments = &volume_raster_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &volume_raster_subpass,
		};
		volume_raster_renderpass = vk::raii::RenderPass(device, volume_raster_rp_info);
		vk::GraphicsPipelineCreateInfo volume_pipeline_info{
		        .stageCount = 2,
		        .pStages = volume_stages,
		        .pVertexInputState = &volume_vertex_input,
		        .pInputAssemblyState = &volume_input_assembly,
		        .pViewportState = &volume_viewport_state,
		        .pRasterizationState = &volume_rasterization,
		        .pMultisampleState = &volume_multisample,
		        .pColorBlendState = &volume_blend,
		        .pDynamicState = &volume_dynamic,
		        .layout = *volume_raster_layout,
		        .renderPass = *volume_raster_renderpass,
		};
		volume_raster_pipeline = vk::raii::Pipeline(device, *pipeline_cache(device), volume_pipeline_info);
	}
	// Compare: fullscreen _mv binary flatten (sensor D16 + volume R16).
	// Renders into volume_bin.blur_fb_mv under the shared R8 MV pass
	// (identical attachment shape); only the fragment shader + set
	// layout (2 samplers) are new.
	{
		auto compare_frag = load_shader(device, "volume_compare_mv.frag");
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
		vk::DescriptorSetLayoutBinding compare_bindings[3] = {
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
		};
		vk::DescriptorSetLayoutCreateInfo compare_set_info{
		        .bindingCount = 3,
		        .pBindings = compare_bindings,
		};
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
		        .descriptorCount = 4,
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
		// Single compare pipeline: the shader has no variants (no spec
		// constants, no debug modes) — one remap path, selected by
		// construction.
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
	// Virtual depth SBS -> sensor-D canvas: fullscreen _mv conversion
	// (meters->D with the volume near/far), so the compare stays in D
	// space with untouched push constants. Renders into
	// volume_virt.blur_fb_mv... under its own CLEAR R16 MV pass (the
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
	// GENERAL).
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
		        .initialLayout = vk::ImageLayout::eGeneral,
		        .finalLayout = vk::ImageLayout::eGeneral,
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
	for (int layer = 0; layer < 2; ++layer)
	{
		vk::ImageViewCreateInfo view_info{
		        .image = *entry.image,
		        .viewType = vk::ImageViewType::e2D,
		        .format = depth_format,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eDepth,
		                .baseMipLevel = 0,
		                .levelCount = 1,
		                .baseArrayLayer = (uint32_t)layer,
		                .layerCount = 1,
		        },
		};
		entry.views.emplace_back(device, view_info);
	}
	return entry;
}

vk::ImageView feather_mask_renderer::output_depth_view(vk::Extent2D extent, int eye)
{
	auto key = std::make_pair(extent.width, extent.height);
	auto it = output_depth.find(key);
	if (it == output_depth.end())
	{
		spdlog::info("Fiducial mask depth target: {}x{}", extent.width, extent.height);
		it = output_depth.emplace(key, make_depth_entry(extent)).first;
	}
	return *it->second.views[eye];
}

void feather_mask_renderer::ensure_targets(vk::Extent2D extent)
{
	if (targets_extent.width == extent.width and targets_extent.height == extent.height and
	    targets_extent.width != 0)
		return;

	// Rare path (extent change): in-flight frames may still read the old
	// images, mirroring setup_reprojection_swapchain.
	device.waitIdle();
	targets_extent = extent;
	// A/B intermediates at half mask resolution (the mask extent is
	// 64-quantized upstream, hence evenly divisible); tier-1 raster and
	// H run small, V writes the half-res submitted image (the compositor
	// upscales to full).
	vk::Extent2D half_targets = level_extent(extent, 1);
	for (auto * target: {&target_a, &target_b})
	{
		// Views/framebuffers must die before the image they reference
		// (the image assignment below destroys the old image).
		target->views.clear();
		target->raster_fbs.clear();
		target->blur_fbs.clear();
		target->view_array.clear();
		target->blur_fb_mv.clear();
		// Depth companions likewise (before the depth image assignment
		// below destroys the old one). Blur-only target B never
		// rasterizes, so it keeps no depth resources.
		target->raster_depth_fbs.clear();
		target->depth_views.clear();
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
		for (int layer = 0; layer < 2; ++layer)
		{
			vk::ImageViewCreateInfo view_info{
			        .image = *target->image,
			        .viewType = vk::ImageViewType::e2D,
			        .format = blur_format,
			        .subresourceRange = {
			                .aspectMask = vk::ImageAspectFlagBits::eColor,
			                .baseMipLevel = 0,
			                .levelCount = 1,
			                .baseArrayLayer = (uint32_t)layer,
			                .layerCount = 1,
			        },
			};
			target->views.emplace_back(device, view_info);
		}
		// Named copies: framebuffer create-info stores the pointer, so the
		// handles must outlive the constructor calls (no &* temporaries).
		vk::ImageView raw_views[2] = {*target->views[0], *target->views[1]};
		for (int layer = 0; layer < 2; ++layer)
		{
			vk::FramebufferCreateInfo fb_info{
			        .renderPass = *raster_r8_renderpass,
			        .attachmentCount = 1,
			        .pAttachments = &raw_views[layer],
			        .width = half_targets.width,
			        .height = half_targets.height,
			        .layers = 1,
			};
			target->raster_fbs.emplace_back(device, fb_info);
		}
		// Multiview sampling + render target: array view over both
		// layers, one layers=2 framebuffer under the multiview R8 pass.
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
		target->blur_fb_mv = vk::raii::Framebuffer(device,
		        vk::FramebufferCreateInfo{
		                .renderPass = *blur_r8_renderpass_mv,
		                .attachmentCount = 1,
		                .pAttachments = &raw_array,
		                .width = half_targets.width,
		                .height = half_targets.height,
		                .layers = 2,
		        });
		if (needs_depth)
		{
			// Depth-tested seed-raster companions: D16 image + per-eye
			// views + one color+depth framebuffer per eye wrapping the
			// color views above (R8+depth pass).
			depth_entry depth = make_depth_entry(half_targets);
			target->depth_image = std::move(depth.image);
			target->depth_memory = std::move(depth.memory);
			target->depth_views = std::move(depth.views);
			vk::ImageView raw_depth[2] = {*target->depth_views[0], *target->depth_views[1]};
			for (int layer = 0; layer < 2; ++layer)
			{
				vk::ImageView attachments[2] = {raw_views[layer], raw_depth[layer]};
				vk::FramebufferCreateInfo depth_fb_info{
				        .renderPass = *raster_r8_depth_renderpass,
				        .attachmentCount = 2,
				        .pAttachments = attachments,
				        .width = half_targets.width,
				        .height = half_targets.height,
				        .layers = 1,
				};
				target->raster_depth_fbs.emplace_back(device, depth_fb_info);
			}
		}
	}
	// Downsample/blur levels at half, quarter and eighth mask resolution
	// (exact halving: the mask extent is 64-quantized upstream). One set of
	// framebuffers each (blur render pass throughout); A/B above keep both.
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
			target->views.clear();
			target->blur_fbs.clear();
			target->view_array.clear();
			target->blur_fb_mv.clear();
			// Depth companions likewise. Seed raster runs on the down
			// levels (never on the eblur workspaces), so only they keep
			// depth resources.
			target->raster_depth_fbs.clear();
			target->depth_views.clear();
			bool needs_depth = (target == &down_targets[li]);
			target->image = vk::raii::Image(device, image_info);
			auto requirements = target->image.getMemoryRequirements();
			vk::MemoryAllocateInfo alloc_info{
			        .allocationSize = requirements.size,
			        .memoryTypeIndex = find_memory_type(physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
			};
			target->memory = vk::raii::DeviceMemory(device, alloc_info);
			target->image.bindMemory(*target->memory, 0);
			for (int layer = 0; layer < 2; ++layer)
			{
				vk::ImageViewCreateInfo view_info{
				        .image = *target->image,
				        .viewType = vk::ImageViewType::e2D,
				        .format = blur_format,
				        .subresourceRange = {
				                .aspectMask = vk::ImageAspectFlagBits::eColor,
				                .baseMipLevel = 0,
				                .levelCount = 1,
				                .baseArrayLayer = (uint32_t)layer,
				                .layerCount = 1,
				        },
				};
				target->views.emplace_back(device, view_info);
			}
			// Per-eye framebuffers serve the seed raster only (down
			// levels; eblur workspaces never rasterize), so they are
			// built directly against the raster pass now that the
			// single-view blur pass is gone.
			vk::ImageView raw_views[2] = {*target->views[0], *target->views[1]};
			if (needs_depth)
			{
				for (int layer = 0; layer < 2; ++layer)
				{
					vk::FramebufferCreateInfo fb_info{
					        .renderPass = *raster_r8_renderpass,
					        .attachmentCount = 1,
					        .pAttachments = &raw_views[layer],
					        .width = le.width,
					        .height = le.height,
					        .layers = 1,
					};
					target->blur_fbs.emplace_back(device, fb_info);
				}
			}
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
			target->blur_fb_mv = vk::raii::Framebuffer(device,
			        vk::FramebufferCreateInfo{
			                .renderPass = *blur_r8_renderpass_mv,
			                .attachmentCount = 1,
			                .pAttachments = &raw_array,
			                .width = le.width,
			                .height = le.height,
			                .layers = 2,
			        });
			if (needs_depth)
			{
				// Seed raster runs on down_targets[seed-1].blur_fbs (identical
				// attachment spec); the depth path pairs the same color views
				// with a D16 image under the R8+depth pass (see above).
				depth_entry depth = make_depth_entry(le);
				target->depth_image = std::move(depth.image);
				target->depth_memory = std::move(depth.memory);
				target->depth_views = std::move(depth.views);
				vk::ImageView raw_depth[2] = {*target->depth_views[0], *target->depth_views[1]};
				for (int layer = 0; layer < 2; ++layer)
				{
					vk::ImageView attachments[2] = {raw_views[layer], raw_depth[layer]};
					vk::FramebufferCreateInfo depth_fb_info{
					        .renderPass = *raster_r8_depth_renderpass,
					        .attachmentCount = 2,
					        .pAttachments = attachments,
					        .width = le.width,
					        .height = le.height,
					        .layers = 1,
					};
					target->raster_depth_fbs.emplace_back(device, depth_fb_info);
				}
			}
		}
	}
	spdlog::info("Fiducial mask blur targets: {}x{} (A/B at half, levels below)", extent.width, extent.height);
}

void feather_mask_renderer::ensure_volume_targets()
{
	if (volume_targets_ready)
		return;

	// One-time path (fixed 320px size): in-flight frames cannot exist
	// before the first volume record, but waitIdle is cheap insurance
	// on this cold path, mirroring ensure_targets.
	device.waitIdle();
	vk::Extent2D ve{volume_size, volume_size};
	auto make_target_image = [&](vk::Format fmt) {
		vk::ImageCreateInfo image_info{
		        .imageType = vk::ImageType::e2D,
		        .format = fmt,
		        .extent = {ve.width, ve.height, 1},
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

	// Furthest-D canvas (R16, sensor-compatible): per-eye raster views +
	// array view for the compare sampler.
	std::tie(volume_depth.image, volume_depth.memory) = make_target_image(volume_depth_format);
	for (int layer = 0; layer < 2; ++layer)
	{
		vk::ImageViewCreateInfo view_info{
		        .image = *volume_depth.image,
		        .viewType = vk::ImageViewType::e2D,
		        .format = volume_depth_format,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .baseMipLevel = 0,
		                .levelCount = 1,
		                .baseArrayLayer = (uint32_t)layer,
		                .layerCount = 1,
		        },
		};
		volume_depth.views.emplace_back(device, view_info);
	}
	vk::ImageView raw_depth_views[2] = {*volume_depth.views[0], *volume_depth.views[1]};
	for (int layer = 0; layer < 2; ++layer)
	{
		vk::FramebufferCreateInfo fb_info{
		        .renderPass = *volume_raster_renderpass,
		        .attachmentCount = 1,
		        .pAttachments = &raw_depth_views[layer],
		        .width = ve.width,
		        .height = ve.height,
		        .layers = 1,
		};
		volume_depth.raster_fbs.emplace_back(device, fb_info);
	}
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

	// Virtual-depth D canvas (R16, same 320px 2-layer shape): SBS
	// meters->D conversion output, cleared to far. Own MV framebuffer
	// under volume_virt_renderpass (CLEAR R16 MV).
	std::tie(volume_virt.image, volume_virt.memory) = make_target_image(volume_depth_format);
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
	                .width = ve.width,
	                .height = ve.height,
	                .layers = 2,
	        });

	// Binary canvas (R8): compare renders MV into blur_fb_mv under the
	// shared R8 pass; merge samples the array view with LINEAR.
	std::tie(volume_bin.image, volume_bin.memory) = make_target_image(blur_format);
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

	// Presmooth workspace (R8, same shape as volume_bin): the 2x2
	// coverage output is blurred H+V here, then V writes back into
	// volume_bin, so the merge below upscales an already-smooth field.
	// Framebuffer under the same blur pass (format-identical: the only
	// compatibility that matters).
	std::tie(volume_smooth.image, volume_smooth.memory) = make_target_image(blur_format);
	volume_smooth.view_array = vk::raii::ImageView(device,
	        vk::ImageViewCreateInfo{
	                .image = *volume_smooth.image,
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
	vk::ImageView raw_smooth_array = *volume_smooth.view_array;
	volume_smooth.blur_fb_mv = vk::raii::Framebuffer(device,
	        vk::FramebufferCreateInfo{
	                .renderPass = *blur_r8_renderpass_mv,
	                .attachmentCount = 1,
	                .pAttachments = &raw_smooth_array,
	                .width = ve.width,
	                .height = ve.height,
	                .layers = 2,
	        });

	volume_targets_ready = true;
	spdlog::info("Fiducial mask depth-volume targets: {}x{}", ve.width, ve.height);
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

	std::array<vk::BufferMemoryBarrier, 2> barriers = {
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
	// Depth volume engages exactly when the group carries depth draws
	// with a valid sensor view. The raster is a plain MAX accumulation
	// (no MIN cuts: the merge below MAXes over the reversed result).
	bool use_volume = volume && !volume->draws.empty() && volume->sensor_view != vk::ImageView{};
	float box_frac = 1;
	size_t n_draws = 0;
	int tier_out = -1;
	bool has_reversed = std::ranges::any_of(draws, [](const instance_draw & d) { return d.reverse; });
	bool has_positive = std::ranges::any_of(draws, [](const instance_draw & d) { return !d.reverse; });
	// Reversed-only raster is a provable no-op (MIN against clear 0 writes
	// 0), so clear-only and let the cutout punch below run if armed. Depth
	// engages exactly when a reversed draw exists; positive-only calls
	// keep the legacy passes bit-identically.
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

	// Same-queue read-after-write between passes. Intermediates live
	// in OPTIMAL layouts (UBWC on); submitted images stay GENERAL and
	// never pass through here.
	// Full-image range: the multiview chain reads both layers together,
	// so per-eye narrowing no longer applies (one barrier replaces two).
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
	// pass's final layout (ATTACH from standalone passes, GENERAL
	// from the merged triple); new is always READ_ONLY.
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
	// covers) or the GENERAL submit upscale.
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
			targets.erase(it);
		// NOTE: views below are named locals, not &* temporaries: the
		// framebuffer create-info stores the pointer, so the VkImageView
		// must outlive the vk::raii::Framebuffer constructor call.
		// (&* on a vk::raii handle takes the address of a returned
		// temporary: freed stack read as an image view, GPU fault, silent
		// fence hang. That exact bug froze the first mask frame.)
		std::array<vk::raii::ImageView, 2> views{{
		        [&] {
			        vk::ImageViewCreateInfo view_info{
			                .image = image,
			                .viewType = vk::ImageViewType::e2D,
			                .format = format,
			                .subresourceRange = {
			                        .aspectMask = vk::ImageAspectFlagBits::eColor,
			                        .baseMipLevel = 0,
			                        .levelCount = 1,
			                        .baseArrayLayer = 0,
			                        .layerCount = 1,
			                },
			        };
			        return vk::raii::ImageView(device, view_info);
		        }(),
		        [&] {
			        vk::ImageViewCreateInfo view_info{
			                .image = image,
			                .viewType = vk::ImageViewType::e2D,
			                .format = format,
			                .subresourceRange = {
			                        .aspectMask = vk::ImageAspectFlagBits::eColor,
			                        .baseMipLevel = 0,
			                        .levelCount = 1,
			                        .baseArrayLayer = 1,
			                        .layerCount = 1,
			                },
			        };
			        return vk::raii::ImageView(device, view_info);
		        }(),
		}};
		vk::ImageView raw_view_0 = *views[0];
		vk::ImageView raw_view_1 = *views[1];
		std::array<vk::raii::Framebuffer, 2> framebuffers{{
		        [&] {
			        vk::FramebufferCreateInfo fb_info{
			                .renderPass = *renderpass,
			                .attachmentCount = 1,
			                .pAttachments = &raw_view_0,
			                .width = out_extent.width,
			                .height = out_extent.height,
			                .layers = 1,
			        };
			        return vk::raii::Framebuffer(device, fb_info);
		        }(),
		        [&] {
			        vk::FramebufferCreateInfo fb_info{
			                .renderPass = *renderpass,
			                .attachmentCount = 1,
			                .pAttachments = &raw_view_1,
			                .width = out_extent.width,
			                .height = out_extent.height,
			                .layers = 1,
			        };
			        return vk::raii::Framebuffer(device, fb_info);
		        }(),
		}};
		// Multiview V-upscale target: array view over both layers plus
		// one layers=2 framebuffer under blur_renderpass_mv. Per-eye
		// framebuffers above stay (tier-0 raster, cutout punch).
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
		// Depth-tested raster companions for tier-0 direct raster (color +
		// depth under depth_renderpass): same color views as above plus the
		// cached submitted-size depth views (swapchain-independent, shared
		// across groups submitting at this size). Positive-only targets
		// stay lean (null framebuffers, has_depth false).
		std::array<vk::raii::Framebuffer, 2> depth_framebuffers{{nullptr, nullptr}};
		bool has_depth = false;
		if (has_reversed)
		{
			vk::ImageView raw_depth[2] = {output_depth_view(out_extent, 0), output_depth_view(out_extent, 1)};
			vk::ImageView raw_color[2] = {raw_view_0, raw_view_1};
			for (int layer = 0; layer < 2; ++layer)
			{
				vk::ImageView attachments[2] = {raw_color[layer], raw_depth[layer]};
				vk::FramebufferCreateInfo fb_info{
				        .renderPass = *depth_renderpass,
				        .attachmentCount = 2,
				        .pAttachments = attachments,
				        .width = out_extent.width,
				        .height = out_extent.height,
				        .layers = 1,
				};
				depth_framebuffers[layer] = vk::raii::Framebuffer(device, fb_info);
			}
			has_depth = true;
		}
		it = targets.emplace(image, frame_targets(out_extent, std::move(views), std::move(framebuffers), std::move(depth_framebuffers), has_depth, std::move(mv_array_view), std::move(mv_framebuffer))).first;
	}

	vk::ClearValue clear{};
	clear.color.float32.fill(0);
	// Shared empty descriptor offsets (all fullscreen passes bind a single
	// set with no dynamic offsets).
	std::array<uint32_t, 0> no_offsets{};

	// Silhouette raster shared by every path: one draw per (mesh,
	// instance) into one eye's framebuffer at the given extent. Missing
	// meshes (removed object racing a queued draw) skip defensively. NDC
	// is resolution-independent, so the same draws serve full- and
	// tier-res targets with only the viewport changing. Per-eye callers
	// complete each eye's full chain before the next eye starts (eye-outer
	// order), keeping that eye's tiles hot.
	// Raster target bundle: legacy (color-only) and depth (color+depth)
	// objects for one raster site. The caller engages depth whenever a
	// reversed draw exists; positive-only calls stay on the legacy triple
	// bit-identically (same passes, pipes and clears as before).
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
	auto raster_silhouette = [&](const raster_site & site, vk::Extent2D e, int eye, bool use_depth) {
		std::array<vk::ClearValue, 2> clears{};
		clears[0].color.float32.fill(0);
		// 0 = far plane under the reversed-Z projection (near = 1).
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
		// Two polarity passes in depth mode (positives depth-write first,
		// reversed MIN-cuts second: at most one pipe rebind, no allocation),
		// one legacy pass otherwise. Legacy mode never sees reversed draws
		// (the caller engages depth whenever one exists); skipped
		// defensively so a polarity can never masquerade as a fill.
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
				// Reversed draws push the hole strength (1 - fade): fresh
				// placements subtract nothing (no reality blob where no
				// window exists), fully faded cut fully. The blur chain is
				// linear, so the ramp feathers like any fade.
				float v = d.reverse ? 1.f - d.opacity : d.opacity;
				cmd.pushConstants<raster_push>(*pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
				                               raster_push{.mvp = d.mvp[eye], .opacity = v});
				cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
				++n_draws;
			}
		}
		cmd.endRenderPass();
	};


	// A volume-only group has no silhouette draws: it still runs the
	// tier path (cleared seed, then the volume merge below sets it).
	if (not rasterize or (not has_positive and not use_volume))
	{
		section_clock raster_clk(&ms_raster);
		// Bypass: clear-only passes, transparent mask, full stack hot.
		// Also covers reversed-only calls (a provable no-op: MIN against
		// clear 0 writes 0); the cutout punch below still runs if armed.
		for (int eye = 0; eye < 2; ++eye)
		{
			vk::RenderPassBeginInfo begin_info{
			        .renderPass = *renderpass,
			        .framebuffer = *it->second.framebuffers[eye],
			        .renderArea = {.offset = {0, 0}, .extent = out_extent},
			        .clearValueCount = 1,
			        .pClearValues = &clear,
			};
			cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
			cmd.endRenderPass();
		}
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
		ensure_targets(extent);

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
		for (const auto & d: draws)
			accumulate_box(d);
		// Depth volumes feed the same erode/blur chain, so the scissor
		// box must cover them too (viewer-inside volumes usually trip
		// the behind-camera fallback to fullscreen, which is correct).
		if (use_volume)
			for (const auto & d: volume->draws)
				accumulate_box(d);
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

		// Depth-volume layer: furthest-D raster at 320px, strict
		// sensor<render> flatten, upscale+MAX into the seed. The seed
		// already holds silhouettes + reversed MIN cuts (or is clear),
		// so the merge LOADs and MAXes over them: the volume overlays
		// everything, then erode/blur feather it like silhouettes.
		auto run_volume = [&](vk::Image seed_img, vk::Framebuffer seed_fb, vk::Extent2D seed_ext, bool seed_is_rgb8) {
			ensure_volume_targets();
			vk::Extent2D ve{volume_size, volume_size};
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
				for (int eye = 0; eye < 2; ++eye)
				{
					vk::RenderPassBeginInfo begin_vol{
					        .renderPass = *volume_raster_renderpass,
					        .framebuffer = *volume_depth.raster_fbs[eye],
					        .renderArea = {.offset = {0, 0}, .extent = ve},
					        .clearValueCount = 1,
					        .pClearValues = &vclear,
					};
					cmd.beginRenderPass(begin_vol, vk::SubpassContents::eInline);
					set_full_viewport(ve);
					cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *volume_raster_pipeline);
					for (const auto & d: volume->draws)
					{
						auto mit = meshes.find(d.mesh);
						if (mit == meshes.end() or mit->second.index_count == 0)
							continue;
						cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
						cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
						cmd.pushConstants<volume_push>(*volume_raster_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
						                               volume_push{.mvp = d.mvp[eye], .near_z = volume->near_z, .far_z = volume->far_z});
						cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
						++n_draws;
					}
					cmd.endRenderPass();
				}
			}
			volume_stamp(1);
			make_readable(*volume_depth.image);
			// Virtual depth SBS -> D canvas (only when paired): meters->D
			// fullscreen _mv at 320px with the volume near/far. Unpaired
			// frames skip the pass (stale canvas ignored: compare rebinds
			// the volume canvas, so the legacy test is bit-identical).
			if (volume->virt_valid)
			{
				section_clock virt_clk(&ms_volume_virt);
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
				vk::ClearValue vfar{};
				// memcpy: ClearColorValue union members are C arrays here,
				// not assignable.
				const float far1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
				std::memcpy(vfar.color.float32, far1, sizeof(far1));
				vk::RenderPassBeginInfo begin_virt{
				        .renderPass = *volume_virt_renderpass,
				        .framebuffer = *volume_virt_fb_mv,
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
				make_readable(*volume_virt.image);
			}
			volume_stamp(2);
			{
				section_clock compare_clk(&ms_volume_compare);
				// Homography + B2 corr push fill (128B named vec4s). Pushed just
				// before the draw below.
				volume_homography hom{};
				for (int i = 0; i < 6; ++i)
					hom.col[i] = volume->remap_cols[i];
				for (int i = 0; i < 2; ++i)
					hom.corr[i] = volume->remap_corr[i];
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
				std::array<vk::CopyDescriptorSet, 0> no_compare_copies{};
				std::array<vk::WriteDescriptorSet, 3> compare_writes = {{
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
					        // volume canvas again (min(v,v) = legacy).
					        .dstSet = *compare_set,
					        .dstBinding = 2,
					        .descriptorCount = 1,
					        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
					        .pImageInfo = volume->virt_valid ? &virt_d_info : &volume_info,
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
				// Presmooth the 320px coverage field (wide kernel): H here,
				// V folded into the merge upscale below. Stairs smaller than
				// the blur radius vanish into a stable gradient instead of
				// shimmering through the upscale. CPU and GPU cost land in
				// the merge spans by construction.
				// H: bin -> smooth. Box-sized render area (like the merge
				// below): LOAD/STORE skip tiles outside the footprint, so
				// presmooth scales with object area. CLEAR zeroes outside
				// the box (same discipline as the compare pass), so edge
				// taps read defined 0 instead of tile garbage.
				vk::Rect2D smooth_box = pass_scissor(ve);
				update_source(*volume_bin.view_array, 0);
				make_writable(*volume_smooth.image);
				vk::RenderPassBeginInfo begin_smooth_h{
					        .renderPass = *blur_r8_renderpass_mv,
					        .framebuffer = *volume_smooth.blur_fb_mv,
					        .renderArea = smooth_box,
					        .clearValueCount = 1,
					        .pClearValues = &clear,
				};
				cmd.beginRenderPass(begin_smooth_h, vk::SubpassContents::eInline);
				set_full_viewport(ve);
				cmd.setScissor(0, smooth_box);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *blur_r8_pipeline_mv);
				std::array<vk::DescriptorSet, 1> smooth_h_sets{*descriptor_sets[0]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, smooth_h_sets, no_offsets);
				blur_push smooth_h{{1.f / ve.width, 1.f / ve.height}, {1, 0}, volume_smooth_spread, 0};
				cmd.pushConstants<blur_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, smooth_h);
				cmd.draw(3, 1, 0, 0);
				cmd.endRenderPass();
				make_readable(*volume_bin.image);
				// LOAD-after-write ordering, same as the cutout punch
				// barrier below: without it a tile GPU may MAX against
				// stale seed tiles.
				vk::ImageLayout seed_layout = seed_is_rgb8 ? vk::ImageLayout::eGeneral : vk::ImageLayout::eColorAttachmentOptimal;
				vk::ImageMemoryBarrier merge_barrier{
					        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
					        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
					        .oldLayout = seed_layout,
					        .newLayout = seed_layout,
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
				if (cpu_stats)
				        cpu_stats->volume_box_frac = (float)merge_box.extent.width * (float)merge_box.extent.height /
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
				// V-blur folded into the upscale: same taps as the deleted
				// standalone V pass (source texel 1/320, V direction), executed
				// while upscaling under the existing MAX blend. Separable with
				// the H pass above by linearity.
				update_source(*volume_smooth.view_array, 1);
				std::array<vk::DescriptorSet, 1> max_sets{*descriptor_sets[1]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, max_sets, no_offsets);
				blur_push max_push{{1.f / ve.width, 1.f / ve.height}, {0, 1}, volume_smooth_spread, 0};
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
			// image. Exact, one raster pass per eye, no blur passes.
			// (Tier-0 groups submit full, so out matches working here.)
			for (int eye = 0; eye < 2; ++eye)
				raster_silhouette({*it->second.framebuffers[eye], *it->second.depth_framebuffers[eye], *pipeline, *depth_pipeline, *reverse_pipeline, *renderpass, *depth_renderpass}, out_extent, eye, has_reversed);
			if (use_volume)
				run_volume(image, *it->second.framebuffer_mv, out_extent, true);
		}
		else if (tier == 1)
		{
		// Stages 1+2+3: raster into half-res A stays per-eye (geometry
		// is trivial), then ONE multiview chain covers both eyes —
		// erode H/V, blur H, V upscale. Same pushes and clears; the _mv
		// fragment twins read each view's layer via gl_ViewIndex, so no
		// per-eye state exists past the raster. One descriptor set per
		// pass (the eye-0 indices), each written once per frame.
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
			for (int eye = 0; eye < 2; ++eye)
				raster_silhouette({*target_a.raster_fbs[eye], *target_a.raster_depth_fbs[eye], *raster_r8_pipeline, *raster_r8_depth_pipeline, *raster_r8_reverse_pipeline, *raster_r8_renderpass, *raster_r8_depth_renderpass}, half_extent, eye, has_reversed);
		}
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
			update_source(*target_a.view_array, 10, true, vk::ImageLayout::eGeneral);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_erode_0);
			std::array<vk::DescriptorSet, 1> sets_e0{*descriptor_sets[10]};
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_e0, no_offsets);
			erode_push epush_h{{1.f / half_extent.width, 1.f / half_extent.height}, {1, 0}, (float)erode, 0};
			cmd.pushConstants<erode_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, epush_h);
			cmd.draw(3, 1, 0, 0);
			cmd.nextSubpass(vk::SubpassContents::eInline);
			update_source(*target_b.view_array, 12, true, vk::ImageLayout::eGeneral);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_erode_1);
			std::array<vk::DescriptorSet, 1> sets_e1{*descriptor_sets[12]};
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
			// Tiered path: seed-rasterize below full res (per eye), one
			// multiview downsample/erode/blur chain to the blur level,
			// blur at 1/k, upscale submit.
			// Tier 2 seeds at quarter (level 2); higher tiers seed at half,
			// so the H working level is the deeper of the seed level and
			// the tier's natural level. down_targets[N].blur_fbs serves the
			// seed raster: identical attachment spec, which is all that
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
			// Seed raster stays per-eye; the downsample/erode/blur chain
			// below runs once per level for both eyes (multiview).
			{
				section_clock raster_clk(&ms_raster);
				for (int eye = 0; eye < 2; ++eye)
					raster_silhouette({*down_targets[seed_level - 1].blur_fbs[eye], *down_targets[seed_level - 1].raster_depth_fbs[eye], *raster_r8_pipeline, *raster_r8_depth_pipeline, *raster_r8_reverse_pipeline, *raster_r8_renderpass, *raster_r8_depth_renderpass}, de_seed, eye, has_reversed);
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
				update_source(down_src, 4 + (l - 1) * 2);
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
				std::array<vk::DescriptorSet, 1> sets_d{*descriptor_sets[4 + (l - 1) * 2]};
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
				update_source(*h_src.view_array, 10, true, vk::ImageLayout::eGeneral);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_erode_0);
				std::array<vk::DescriptorSet, 1> sets_e0{*descriptor_sets[10]};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blur_layout, 0, sets_e0, no_offsets);
				erode_push epush_h{{1.f / te.width, 1.f / te.height}, {1, 0}, (float)erode, 0};
				cmd.pushConstants<erode_push>(*blur_layout, vk::ShaderStageFlagBits::eFragment, 0, epush_h);
				cmd.draw(3, 1, 0, 0);
				cmd.nextSubpass(vk::SubpassContents::eInline);
				update_source(*eblur.view_array, 12, true, vk::ImageLayout::eGeneral);
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *triple_erode_1);
				std::array<vk::DescriptorSet, 1> sets_e1{*descriptor_sets[12]};
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
	// swapchain framebuffers with the LOAD pass and stamps each quad at
	// FULL alpha (mask 1 = reality window): holes show passthrough, not
	// game. No blur pass touches them, so edges stay pixel-exact. Debug
	// only; rasterize=false (bypass) skips them with the silhouettes.
	if (not cutouts.empty() and rasterize)
	{
		// Execution barrier: the V upscale (or tier-0 raster) above wrote
		// the swapchain image as COLOR_ATTACHMENT; the LOAD cutout pass
		// below reads it. Without this a tile GPU may LOAD stale tiles.
		vk::ImageMemoryBarrier cut_barrier{
		        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
		        .oldLayout = vk::ImageLayout::eGeneral,
		        .newLayout = vk::ImageLayout::eGeneral,
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
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
		cmd.bindVertexBuffers(0, (vk::Buffer)cutout_verts, (vk::DeviceSize)0);
		section_clock punch_clk(&ms_punch);
		for (size_t qi = 0; qi < cutouts.size(); ++qi)
		{
			const auto & quad = cutouts[qi];
			for (int eye = 0; eye < 2; ++eye)
			{
				// Tighten to the quad's pixel bounds (same NDC mapping the
				// viewport itself uses, so alignment is exact by
				// construction): fullscreen LOAD/STORE traffic for a small
				// debug rect would cost a full image round-trip per group.
				// Any corner behind the camera falls back to fullscreen.
				// Bounds track the submitted (output) size, like the
				// viewport below: NDC maps consistently at any size.
				vk::Rect2D area{{0, 0}, out_extent};
				bool behind = false;
				float minx = (float)out_extent.width, miny = (float)out_extent.height;
				float maxx = 0, maxy = 0;
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
				if (not behind)
				{
					int32_t x0 = std::clamp<int32_t>((int32_t)std::floor(minx) - 1, 0, (int32_t)out_extent.width);
					int32_t y0 = std::clamp<int32_t>((int32_t)std::floor(miny) - 1, 0, (int32_t)out_extent.height);
					int32_t x1 = std::clamp<int32_t>((int32_t)std::ceil(maxx) + 1, 0, (int32_t)out_extent.width);
					int32_t y1 = std::clamp<int32_t>((int32_t)std::ceil(maxy) + 1, 0, (int32_t)out_extent.height);
					if (x1 <= x0 or y1 <= y0)
						continue;
					area = {{x0, y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
				}
				vk::RenderPassBeginInfo begin_cut{
				        .renderPass = *cutout_renderpass,
				        .framebuffer = *it->second.framebuffers[eye],
				        .renderArea = area,
				        .clearValueCount = 0,
				};
				cmd.beginRenderPass(begin_cut, vk::SubpassContents::eInline);
				set_full_viewport(out_extent);
				cmd.setScissor(0, area);
				cmd.pushConstants<raster_push>(*pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
				                               raster_push{.mvp = cutout_mvp[eye], .opacity = 1});
				cmd.draw(6, 1, (uint32_t)(qi * 6), 0);
				cmd.endRenderPass();
			}
		}
	}

	// All mask passes recorded (barrier below is submit bookkeeping, not
	// mask work): stamp the diagnostics before leaving.
	fill_stats();

	vk::ImageMemoryBarrier barrier{
	        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
	        .oldLayout = vk::ImageLayout::eGeneral,
	        .newLayout = vk::ImageLayout::eGeneral,
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
