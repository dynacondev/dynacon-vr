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

#include "render/hidden_dots.h"

#include "render/feather_mask.h"
#include "vk/shader.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <cstring>

#include <spdlog/spdlog.h>

namespace
{
// Mirrors hidden_dots_mv.vert/frag.glsl PushConstants: per-eye mvp
// (2x64) + color and pitch (16) + per-eye view params (2x16) = 176B.
// EXCEEDS the 128-byte Vulkan minimum: the multiview path needs both
// eyes. Adreno 740 (Quest 3) provides 256B; init_pipelines throws
// loudly below if some future device reports less (do NOT silently
// fall back — a 128B encoding does not exist for stereo MVPs; the
// alternative is a dynamic-UBO MVP delivery, i.e. a redesign).
struct push
{
	std::array<glm::mat4, 2> mvp;
	glm::vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
	std::array<glm::vec4, 2> view_params; // per eye: x: view W px, y: view H px, z: proj[1][1], w: dot radius (cells)
};
static_assert(sizeof(push) == 176);
static_assert(sizeof(push) <= 256);
} // namespace

hidden_dots_renderer::hidden_dots_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device) :
        device(&device),
        physical_device(&physical_device)
{
}

std::shared_ptr<vk::raii::PipelineCache> hidden_dots_renderer::pipeline_cache(vk::raii::Device & device)
{
	// Same process-wide object the feather mask renderer shares (one entry
	// per VkDevice): keeps a single cache alive across both renderers.
	return feather_mask_renderer::pipeline_cache(device);
}

void hidden_dots_renderer::warm_pipelines(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device, vk::Format format)
{
	// Throwaway renderer: its pipeline compile populates the shared
	// cache above, so the first real dots pay ~nothing.
	hidden_dots_renderer warm(device, physical_device);
	warm.init_pipelines(format);
}

bool hidden_dots_renderer::ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h)
{
	// Quarter of full (halved args): small/distant dots fade by coverage
	// instead of popping (no shimmer), while clear/store/shading drop
	// 16x vs full. The compositor upscales the submitted rect to the
	// full FOV; binary alpha carries a ~1-texel upscale ramp.
	uint32_t hw = ((uint32_t)w + 1) / 2;
	uint32_t hh = ((uint32_t)h + 1) / 2;
	if (ready() and format_ == format and extent_.width == hw and extent_.height == hh)
		return false;
	// Rare path (first frame, video resize): nothing outstanding (we
	// acquire + release within one frame, always paired below).
	device->waitIdle();
	// Free the descriptor sets while their pool is still alive:
	// init_pipelines below recreates the pool, which implicitly frees
	// (destroys the use of) any sets still held.
	descriptor_sets.clear();
	format_ = format;
	extent_ = {hw, hh};
	swapchain_ = xr::swapchain(inst, sess, *device, format, (int)hw, (int)hh, 1, 2);
	spdlog::info("Hidden dots swapchain: {}x{} ({} images)", hw, hh, swapchain_.image_count());
	init_pipelines(format);
	// Depth target for the prepass (sampled by the dots pass at the dot
	// centers, device-local).
	depth_ = image_allocation{
	        *device,
	        vk::ImageCreateInfo{
	                .imageType = vk::ImageType::e2D,
	                .format = vk::Format::eD16Unorm,
	                .extent = {hw, hh, 1},
	                .mipLevels = 1,
	                .arrayLayers = 2,
	                .samples = vk::SampleCountFlagBits::e1,
	                .tiling = vk::ImageTiling::eOptimal,
	                // Written by the prepass, sampled by the dots pass.
	                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
	                .sharingMode = vk::SharingMode::eExclusive,
	                .initialLayout = vk::ImageLayout::eUndefined,
	        },
	        VmaAllocationCreateInfo{
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "hidden_dots depth",
	};
	// Array views (both layers): the multiview passes render into and
	// sample them with the view selected by gl_ViewIndex.
	depth_view_array = vk::raii::ImageView(*device, vk::ImageViewCreateInfo{
	                                                       .image = (VkImage)depth_,
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
	// Proximity sphere field: same D16 quarter-res shape, separate image
	// (the mesh field clears far/0 for GREATER; this one clears near/1.0
	// for LESS — the fields must not mix).
	sphere_depth_ = image_allocation{
	        *device,
	        vk::ImageCreateInfo{
	                .imageType = vk::ImageType::e2D,
	                .format = vk::Format::eD16Unorm,
	                .extent = {hw, hh, 1},
	                .mipLevels = 1,
	                .arrayLayers = 2,
	                .samples = vk::SampleCountFlagBits::e1,
	                .tiling = vk::ImageTiling::eOptimal,
	                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
	                .sharingMode = vk::SharingMode::eExclusive,
	                .initialLayout = vk::ImageLayout::eUndefined,
	        },
	        VmaAllocationCreateInfo{
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "hidden_dots sphere depth",
	};
	sphere_view_array_ = vk::raii::ImageView(*device, vk::ImageViewCreateInfo{
	                                                        .image = (VkImage)sphere_depth_,
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
	// One set over the array views (multiview frag selects the layer by
	// gl_ViewIndex). Both images are renderer-owned, so this runs on
	// rebuild only, never per frame; the old set was freed after
	// waitIdle, ahead of the pool recreation in init_pipelines above.
	{
		vk::DescriptorSetAllocateInfo set_alloc{
		        .descriptorPool = *descriptor_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*descriptor_layout,
		};
		descriptor_sets.emplace_back(std::move(device->allocateDescriptorSets(set_alloc)[0]));
		vk::DescriptorImageInfo depth_infos[2] = {
		        vk::DescriptorImageInfo{
		                .sampler = *depth_sampler,
		                .imageView = *depth_view_array,
		                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		        },
		        vk::DescriptorImageInfo{
		                .sampler = *depth_sampler,
		                .imageView = *sphere_view_array_,
		                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		        },
		};
		vk::WriteDescriptorSet depth_writes[2] = {
		        vk::WriteDescriptorSet{
		                .dstSet = *descriptor_sets[0],
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &depth_infos[0],
		        },
		        vk::WriteDescriptorSet{
		                .dstSet = *descriptor_sets[0],
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &depth_infos[1],
		        },
		};
		device->updateDescriptorSets(depth_writes, {});
	}
	// One multiview target per swapchain image: array color view +
	// layers=2 framebuffers; one draw broadcasts to both eyes.
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
		        .renderPass = *color_renderpass,
		        .attachmentCount = 1,
		        .pAttachments = &color_view,
		        .width = extent_.width,
		        .height = extent_.height,
		        .layers = 2,
		};
		targets.back().framebuffer = vk::raii::Framebuffer(*device, fb_info);
		vk::ImageView depth_view = *depth_view_array;
		vk::FramebufferCreateInfo depth_fb_info{
		        .renderPass = *depth_renderpass,
		        .attachmentCount = 1,
		        .pAttachments = &depth_view,
		        .width = extent_.width,
		        .height = extent_.height,
		        .layers = 2,
		};
		targets.back().depth_framebuffer = vk::raii::Framebuffer(*device, depth_fb_info);
	}
	return true;
}

int hidden_dots_renderer::acquire()
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

void hidden_dots_renderer::release()
{
	swapchain_.release();
}

void hidden_dots_renderer::init_pipelines(vk::Format format)
{
	pipe_cache = pipeline_cache(*device);
	// Static billboard quad (two triangles, corners -1..1), uploaded once:
	// every dot instance expands into these 6 verts in the vertex shader.
	if (not quad_)
	{
		quad_ = buffer_allocation{
		        *device,
		        vk::BufferCreateInfo{
		                .size = sizeof(float) * 2 * 6,
		                .usage = vk::BufferUsageFlagBits::eVertexBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "hidden_dots quad",
		};
		static const float corners[12] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
		std::memcpy(quad_.map(), corners, sizeof(corners));
		quad_.unmap();
	}
	// Sampled-depth plumbing: binding 0 = mesh prepass output, binding 1
	// = proximity sphere field (both array views over one nearest sampler:
	// exact texel, no cross-edge blending, layer selected by gl_ViewIndex).
	// One set layout + pool for the single set written in ensure().
	// Recreated with everything else on rebuild (move-assign drops old).
	vk::DescriptorSetLayoutBinding depth_bindings[2] = {
	        vk::DescriptorSetLayoutBinding{
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	        vk::DescriptorSetLayoutBinding{
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	descriptor_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                 .bindingCount = 2,
	                                                                 .pBindings = depth_bindings,
	                                                         });
	vk::SamplerCreateInfo sampler_info{
	        .magFilter = vk::Filter::eNearest,
	        .minFilter = vk::Filter::eNearest,
	        .mipmapMode = vk::SamplerMipmapMode::eNearest,
	        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
	        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
	        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
	        .maxLod = 0,
	};
	depth_sampler = vk::raii::Sampler(*device, sampler_info);
	vk::DescriptorPoolSize pool_size{
	        .type = vk::DescriptorType::eCombinedImageSampler,
	        .descriptorCount = 4,
	};
	descriptor_pool = vk::raii::DescriptorPool(*device, vk::DescriptorPoolCreateInfo{
	                                                           .maxSets = 2,
	                                                           .poolSizeCount = 1,
	                                                           .pPoolSizes = &pool_size,
	                                                   });
	// Multiview pushes (176B) exceed the 128-byte Vulkan minimum:
	// fail loudly here — not silently on device — when unsupported.
	if (physical_device->getProperties().limits.maxPushConstantsSize < sizeof(push))
		throw std::runtime_error("hidden dots multiview needs 176B push constants");
	auto vert = load_shader(*device, "hidden_dots_mv.vert");
	auto frag = load_shader(*device, "hidden_dots_mv.frag");

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

	vk::VertexInputBindingDescription bindings[2] = {
	        {
	                .binding = 0, // static quad corners, per-vertex
	                .stride = sizeof(float) * 2,
	                .inputRate = vk::VertexInputRate::eVertex,
	        },
	        {
	                // baked dot center + face normal, per-instance
	                // (borrowed feather buffer, 2x vec3).
	                .binding = 1,
	                .stride = sizeof(float) * 6,
	                .inputRate = vk::VertexInputRate::eInstance,
	        },
	};
	vk::VertexInputAttributeDescription attributes[3] = {
	        {
	                .location = 0,
	                .binding = 0,
	                .format = vk::Format::eR32G32Sfloat,
	                .offset = 0,
	        },
	        {
	                .location = 1,
	                .binding = 1,
	                .format = vk::Format::eR32G32B32Sfloat,
	                .offset = 0,
	        },
	        {
	                .location = 2,
	                .binding = 1,
	                .format = vk::Format::eR32G32B32Sfloat,
	                .offset = sizeof(float) * 3,
	        },
	};
	vk::PipelineVertexInputStateCreateInfo vertex_input{
	        .vertexBindingDescriptionCount = 2,
	        .pVertexBindingDescriptions = bindings,
	        .vertexAttributeDescriptionCount = 3,
	        .pVertexAttributeDescriptions = attributes,
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
	        // No culling: billboards face the camera by construction;
	        // backfaces collapse in the vertex shader via the baked
	        // per-dot normal (no depth test, always on top).
	        .cullMode = vk::CullModeFlagBits::eNone,
	        .frontFace = vk::FrontFace::eCounterClockwise,
	        .lineWidth = 1,
	};

	vk::PipelineMultisampleStateCreateInfo multisample{
	        .rasterizationSamples = vk::SampleCountFlagBits::e1,
	};

	// No depth attachment in the dots pass: occlusion is a sampled
	// center-depth compare in the fragment shader (whole circles live or
	// die), not a per-pixel test (which would chop circles at edges).
	vk::PipelineDepthStencilStateCreateInfo depth_stencil{
	        .depthTestEnable = VK_FALSE,
	        .depthWriteEnable = VK_FALSE,
	};

	// Depth-fill twin: mesh triangles only, color writes off, depth write
	// on. Same vertex layout as the feather silhouette raster (plain
	// vec3 positions).
	vk::PipelineDepthStencilStateCreateInfo fill_depth_stencil{
	        .depthTestEnable = VK_TRUE,
	        .depthWriteEnable = VK_TRUE,
	        .depthCompareOp = vk::CompareOp::eGreater,
	};
	vk::VertexInputBindingDescription fill_binding{
	        .binding = 0,
	        .stride = sizeof(float) * 3,
	        .inputRate = vk::VertexInputRate::eVertex,
	};
	vk::VertexInputAttributeDescription fill_attribute{
	        .location = 0,
	        .binding = 0,
	        .format = vk::Format::eR32G32B32Sfloat,
	        .offset = 0,
	};
	vk::PipelineVertexInputStateCreateInfo fill_vertex_input{
	        .vertexBindingDescriptionCount = 1,
	        .pVertexBindingDescriptions = &fill_binding,
	        .vertexAttributeDescriptionCount = 1,
	        .pVertexAttributeDescriptions = &fill_attribute,
	};

	// Straight alpha blend over transparent; dots are opaque, non-dot
	// pixels discard before blending.
	vk::PipelineColorBlendAttachmentState blend_attachment{
	        .blendEnable = VK_TRUE,
	        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
	        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
	        .colorBlendOp = vk::BlendOp::eAdd,
	        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
	        .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
	        .alphaBlendOp = vk::BlendOp::eAdd,
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
	        .size = sizeof(push),
	};
	vk::PipelineLayoutCreateInfo layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*descriptor_layout,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &push_range,
	};
	pipeline_layout = vk::raii::PipelineLayout(*device, layout_info);

	// Color CLEAR on UNDEFINED every frame (no layout history ever assumed,
	// gizmo precedent). Final COLOR_ATTACHMENT_OPTIMAL (spec-compatible at
	// submit, keeps Adreno UBWC live; nothing app-side re-reads this image).
	vk::AttachmentDescription color_attachment{
	        .format = format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference color_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription color_subpass{
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
	vk::RenderPassCreateInfo color_renderpass_info{
	        .pNext = &mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &color_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &color_subpass,
	};
	color_renderpass = vk::raii::RenderPass(*device, color_renderpass_info);

	// Depth-only prepass: CLEAR every frame (initial UNDEFINED is the
	// transient-target idiom — contents discarded, cleared anyway),
	// STORE + shader-readable final so the dots pass samples it. The
	// subpass dependency carries the write→read hazard (no manual
	// barrier between the passes).
	vk::AttachmentDescription depth_attachment{
	        .format = vk::Format::eD16Unorm,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
	        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	vk::AttachmentReference depth_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
	};
	vk::SubpassDescription depth_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .pDepthStencilAttachment = &depth_ref,
	};
	vk::SubpassDependency depth_dep{
	        .srcSubpass = 0,
	        .dstSubpass = VK_SUBPASS_EXTERNAL,
	        .srcStageMask = vk::PipelineStageFlagBits::eLateFragmentTests,
	        .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
	        .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
	};
	vk::RenderPassCreateInfo depth_renderpass_info{
	        .pNext = &mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &depth_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &depth_subpass,
	        .dependencyCount = 1,
	        .pDependencies = &depth_dep,
	};
	depth_renderpass = vk::raii::RenderPass(*device, depth_renderpass_info);

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
	        .renderPass = *color_renderpass,
	};
	pipeline = vk::raii::Pipeline(*device, *pipe_cache, pipeline_info);

	auto fill_vert = load_shader(*device, "hidden_dotfill_mv.vert");
	auto fill_frag = load_shader(*device, "hidden_dotfill.frag");
	vk::PipelineShaderStageCreateInfo fill_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **fill_vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **fill_frag,
	                .pName = "main",
	        },
	};
	vk::GraphicsPipelineCreateInfo fill_pipeline_info{
	        .stageCount = 2,
	        .pStages = fill_stages,
	        .pVertexInputState = &fill_vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pDepthStencilState = &fill_depth_stencil,
	        // Depth-only subpass: no color blend state (null, not a
	        // zero-mask attachment).
	        .pColorBlendState = nullptr,
	        .pDynamicState = &dynamic,
	        .layout = *pipeline_layout,
	        .renderPass = *depth_renderpass,
	};
	fill_pipeline = vk::raii::Pipeline(*device, *pipe_cache, fill_pipeline_info);
}

bool hidden_dots_renderer::record(vk::raii::CommandBuffer & cmd, size_t image_index,
                                  const std::array<vk::Extent2D, 2> & extents,
                                  const std::vector<instance_draw> & draws)
{
	if (draws.empty() or image_index >= targets.size())
		return false;
	// Multiview broadcasts one draw to both layers: both eyes must share
	// the extent (caller passes {dext, dext}). A mismatch would silently
	// misalign one eye — nausea-grade. Loud here instead.
	assert(extents[0].width == extents[1].width and extents[0].height == extents[1].height);

	// Clamp helper, hoisted (stateless: extent passed per call — one
	// definition serves the prescan and both passes below).
	auto clamp_scissor = [&](const vk::Rect2D & r, const vk::Extent2D & ext, vk::Rect2D & out) {
		int32_t x0 = std::max(r.offset.x, 0);
		int32_t y0 = std::max(r.offset.y, 0);
		int32_t x1 = std::min(r.offset.x + (int32_t)r.extent.width, (int32_t)ext.width);
		int32_t y1 = std::min(r.offset.y + (int32_t)r.extent.height, (int32_t)ext.height);
		if (x1 <= x0 or y1 <= y0)
			return false;
		out = vk::Rect2D{.offset = {x0, y0}, .extent = {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
		return true;
	};

	// Multiview: one draw broadcasts to both layers (viewMask=both), so
	// the two per-eye passes collapse into two begins total. The vertex
	// shader executes once per view; per-draw data carries both eyes and
	// each invocation selects its own (exact, no cross-eye leakage).
	// Both eyes share the dots extent in practice (caller passes
	// {dext, dext}); the framebuffer is that size.
	auto & tgt = targets[image_index];
	cmd.setViewport(0, vk::Viewport{
	                        .x = 0,
	                        .y = 0,
	                        .width = (float)extents[0].width,
	                        .height = (float)extents[0].height,
	                        .minDepth = 0,
	                        .maxDepth = 1,
	                });
	// Both-eyes union of a draw.s valid per-eye boxes (empty = skip the
	// draw outright; a single behind-camera eye must not force
	// fullscreen for the other). Shading tracks object size per draw.
	auto draw_union = [&](const instance_draw & d, vk::Rect2D & out) {
		bool got = false;
		vk::Rect2D acc{};
		for (uint32_t eye = 0; eye < 2; ++eye)
		{
			vk::Rect2D b;
			if (d.index_count == 0 or d.dot_count == 0 or
			    not clamp_scissor(d.scissor[eye], extents[eye], b))
				continue;
			if (not got)
			{
				acc = b;
				got = true;
			}
			else
			{
				int32_t x0 = std::min(acc.offset.x, b.offset.x);
				int32_t y0 = std::min(acc.offset.y, b.offset.y);
				int32_t x1 = std::max(acc.offset.x + (int32_t)acc.extent.width,
				                      b.offset.x + (int32_t)b.extent.width);
				int32_t y1 = std::max(acc.offset.y + (int32_t)acc.extent.height,
				                      b.offset.y + (int32_t)b.extent.height);
				acc = vk::Rect2D{.offset = {x0, y0},
				                  .extent = {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
			}
		}
		if (got)
			out = acc;
		return got;
	};
	// Content union for the depth prepass renderArea (same invariant as
	// before: sampled only at dot centers inside the per-draw boxes).
	// The COLOR pass keeps its full renderArea (submitted full-rect).
	bool any = false;
	vk::Rect2D prepass_area{.offset = {0, 0}, .extent = extents[0]};
	bool prepass_box = false;
	for (const auto & d: draws)
	{
		vk::Rect2D u;
		if (not draw_union(d, u))
			continue;
		any = true;
		if (not prepass_box)
		{
			prepass_area = u;
			prepass_box = true;
		}
		else
		{
			int32_t x0 = std::min(prepass_area.offset.x, u.offset.x);
			int32_t y0 = std::min(prepass_area.offset.y, u.offset.y);
			int32_t x1 = std::max(prepass_area.offset.x + (int32_t)prepass_area.extent.width,
			                      u.offset.x + (int32_t)u.extent.width);
			int32_t y1 = std::max(prepass_area.offset.y + (int32_t)prepass_area.extent.height,
			                      u.offset.y + (int32_t)u.extent.height);
			prepass_area = vk::Rect2D{.offset = {x0, y0},
			                          .extent = {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
		}
	}
	// Pass 1: depth prepass (mesh triangles resolve nearest-surface per
	// pixel; reversed-Z: clear far 0, GREATER wins). One multiview begin
	// over the layers=2 framebuffer (all-degenerate keeps full renderArea:
	// clear-only images still cycle defined content).
	vk::ClearValue depth_clear{};
	depth_clear.depthStencil = {.depth = 0.f, .stencil = 0};
	vk::RenderPassBeginInfo depth_begin{
	        .renderPass = *depth_renderpass,
	        .framebuffer = *tgt.depth_framebuffer,
	        .renderArea = {.offset = prepass_area.offset, .extent = prepass_area.extent},
	        .clearValueCount = 1,
	        .pClearValues = &depth_clear,
	};
	cmd.beginRenderPass(depth_begin, vk::SubpassContents::eInline);
	if (any)
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *fill_pipeline);
		for (const auto & d: draws)
		{
			// Unified skip with the dots loop below. Each loop pushes its
			// own constants per draw; nothing persists across passes.
			vk::Rect2D scissor;
			if (not draw_union(d, scissor))
				continue;
			cmd.setScissor(0, scissor);
			cmd.bindVertexBuffers(0, d.vertex_buffer, (vk::DeviceSize)0);
			cmd.bindIndexBuffer(d.index_buffer, (vk::DeviceSize)0, vk::IndexType::eUint32);
			push pc{
			        .mvp = {{d.mvp[0], d.mvp[1]}},
			        .color_pitch = glm::vec4(d.color, d.pitch),
			        .view_params = {{glm::vec4(d.view_params[0], dot_radius_cells),
			                          glm::vec4(d.view_params[1], dot_radius_cells)}},
			};
			cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
			cmd.drawIndexed(d.index_count, 1, 0, 0, 0);
		}
	}
	cmd.endRenderPass();
	// (Pass 1b retired: the proximity sphere field is rendered once by the
	// feather mask hidden record into the shared sphere image sampled below
	// — same command buffer, earlier. See record().)
	// Pass 2: dots sample the stored depths at their CENTERS (whole circles
	// live or die — no per-pixel chop). The depth renderpass.s subpass
	// dependency carries the write→read hazard, no manual barrier.
	vk::ClearValue color_clear{};
	color_clear.color.float32.fill(0);
	vk::RenderPassBeginInfo color_begin{
	        .renderPass = *color_renderpass,
	        .framebuffer = *tgt.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = extents[0]},
	        .clearValueCount = 1,
	        .pClearValues = &color_clear,
	};
	cmd.beginRenderPass(color_begin, vk::SubpassContents::eInline);
	if (any)
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, *descriptor_sets[0], {});
		cmd.bindVertexBuffers(0, (vk::Buffer)quad_, (vk::DeviceSize)0);
		for (const auto & d: draws)
		{
			// Same unified skip as the fill loop above.
			vk::Rect2D scissor;
			if (not draw_union(d, scissor))
				continue;
			cmd.setScissor(0, scissor);
			// Baked dot instances borrowed from the feather hidden renderer
			// (owns upload + eviction); bound fresh per draw. Per-draw push
			// (multi-draw frames must not share the last push).
			cmd.bindVertexBuffers(1, d.dot_buffer, (vk::DeviceSize)0);
			push pc{
			        .mvp = {{d.mvp[0], d.mvp[1]}},
			        .color_pitch = glm::vec4(d.color, d.pitch),
			        .view_params = {{glm::vec4(d.view_params[0], dot_radius_cells),
			                          glm::vec4(d.view_params[1], dot_radius_cells)}},
			};
			cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
			cmd.draw(6, d.dot_count, 0, 0);
		}
	}
	cmd.endRenderPass();
	return true;
}
