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

#include "render/depth_overlay.h"

#include "vk/shader.h"

#include <array>
#include <cstring>
#include <span>
#include <vector>

#include <spdlog/spdlog.h>

static_assert(sizeof(depth_overlay_renderer::viz_params) == 12);
static_assert(sizeof(depth_overlay_renderer::viz_push) == 16);
static_assert(sizeof(depth_overlay_renderer::splat_warp_params) == 272);
static_assert(sizeof(depth_overlay_renderer::splat_push) == 12);
namespace
{
// Opaque-black clear for empty viz frames (no data this frame): black
// at the slider opacity, so empty is distinguishable from the
// green->red ramp and from transparency.
vk::ClearValue black_clear(float opacity)
{
	vk::ClearValue clear{};
	clear.color.float32 = std::array<float, 4>{0, 0, 0, opacity};
	return clear;
}

// Order the viz read after the same-frame mask-phase writes (async
// path: separate command buffer, same queue, submission-ordered; inline
// path: same command buffer, already ordered — the barrier is a no-op
// there). Layout stays SHADER_READ_ONLY on both sides.
void viz_read_barrier(vk::raii::CommandBuffer & cmd, std::span<const vk::Image> images)
{
	std::vector<vk::ImageMemoryBarrier> barriers;
	barriers.reserve(images.size());
	for (vk::Image image: images)
	{
		if (image == vk::Image{})
			continue;
		barriers.push_back(vk::ImageMemoryBarrier{
		        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
		        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .image = image,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .levelCount = 1,
		                .layerCount = 2,
		        },
		});
	}
	if (barriers.empty())
		return;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    vk::PipelineStageFlagBits::eFragmentShader,
	                    {},
	                    {},
	                    {},
	                    barriers);
}

// Order the transient R16F write before the same-command-buffer pass-2
// tap (layout rides SHADER_READ_ONLY on both sides, viz_read pattern).
void splat_read_barrier(vk::raii::CommandBuffer & cmd, vk::Image image)
{
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    vk::PipelineStageFlagBits::eFragmentShader,
	                    {},
	                    {},
	                    {},
	                    vk::ImageMemoryBarrier{
	                            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	                            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
	                            .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	                            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                            .image = image,
	                            .subresourceRange = {
	                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                    .levelCount = 1,
	                                    .layerCount = 2,
	                            },
	                    });
}

} // namespace


depth_overlay_renderer::depth_overlay_renderer(vk::raii::Device & device) : device(&device) {}

void depth_overlay_renderer::ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h)
{
	if (ready() and format_ == format and (int)extent_.width == w and (int)extent_.height == h)
		return;
	// Rare path (first frame, video resize): nothing outstanding (we
	// acquire + release within one frame, always paired below).
	device->waitIdle();
	format_ = format;
	extent_ = {(uint32_t)w, (uint32_t)h};
	swapchain_ = xr::swapchain(inst, sess, *device, format, w, h, 1, 2);
	spdlog::info("Depth overlay swapchain: {}x{} ({} images)", w, h, swapchain_.image_count());
	init_pipelines(format);
	targets.clear();
	for (vk::Image image: swapchain_.images())
	{
		targets.emplace_back();
		for (uint32_t eye = 0; eye < 2; ++eye)
		{
			vk::ImageViewCreateInfo iv_info{
			        .image = image,
			        .viewType = vk::ImageViewType::e2DArray,
			        .format = format,
			        .subresourceRange = {
			                .aspectMask = vk::ImageAspectFlagBits::eColor,
			                .baseMipLevel = 0,
			                .levelCount = 1,
			                .baseArrayLayer = eye,
			                .layerCount = 1,
			        },
			};
			targets.back()[eye].view = vk::raii::ImageView(*device, iv_info);
			vk::FramebufferCreateInfo fb_info{
			        .renderPass = *renderpass,
			        .width = extent_.width,
			        .height = extent_.height,
			        .layers = 1,
			};
			fb_info.setAttachments(*targets.back()[eye].view);
			targets.back()[eye].framebuffer = vk::raii::Framebuffer(*device, fb_info);
		}
	}
	ensure_splat_targets();
}

void depth_overlay_renderer::ensure_splat_targets()
{
	// waitIdle'd by the caller (ensure: only runs on rebuild): retiring
	// FBs/views/images is safe. Rebuilt unconditionally here (cheap,
	// rare) so the FBs always reference the current splat renderpass,
	// even on a format change with unchanged size. Tear down in
	// dependency order (FBs reference views, views reference images).
	for (auto & tgt : splat_targets)
	{
		tgt.framebuffer.clear();
		tgt.view.clear();
		tgt.depth_view.clear();
	}
	splat_view_array.clear();
	splat_depth_image = image_allocation{};
	splat_image = image_allocation{};
	// Quarter-screen-linear transients (half the overlay per axis):
	// still supersampled vs the ~320-class sensor grid for sub-texel
	// reprojection, at a quarter of the overlay fill. Pass 2 upsamples
	// with the shared NEAREST sampler.
	splat_extent_ = {(extent_.width + 1) / 2, (extent_.height + 1) / 2};

	splat_image = image_allocation(
	        *device,
	        vk::ImageCreateInfo{
	                .imageType = vk::ImageType::e2D,
	                .format = vk::Format::eR16Sfloat,
	                .extent = {splat_extent_.width, splat_extent_.height, 1},
	                .mipLevels = 1,
	                .arrayLayers = 2,
	                .samples = vk::SampleCountFlagBits::e1,
	                .tiling = vk::ImageTiling::eOptimal,
	                .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
	                .sharingMode = vk::SharingMode::eExclusive,
	                .initialLayout = vk::ImageLayout::eUndefined,
	        },
	        VmaAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO},
	        "depth splat transient");
	splat_depth_image = image_allocation(
	        *device,
	        vk::ImageCreateInfo{
	                .imageType = vk::ImageType::e2D,
	                .format = vk::Format::eD16Unorm,
	                .extent = {splat_extent_.width, splat_extent_.height, 1},
	                .mipLevels = 1,
	                .arrayLayers = 2,
	                .samples = vk::SampleCountFlagBits::e1,
	                .tiling = vk::ImageTiling::eOptimal,
	                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
	                .sharingMode = vk::SharingMode::eExclusive,
	                .initialLayout = vk::ImageLayout::eUndefined,
	        },
	        VmaAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO},
	        "depth splat depth");
	splat_view_array = vk::raii::ImageView(*device, vk::ImageViewCreateInfo{
	                                                               .image = vk::Image(splat_image),
	                                                               .viewType = vk::ImageViewType::e2DArray,
	                                                               .format = vk::Format::eR16Sfloat,
	                                                               .subresourceRange = {
	                                                                       .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                                                       .baseMipLevel = 0,
	                                                                       .levelCount = 1,
	                                                                       .baseArrayLayer = 0,
	                                                                       .layerCount = 2,
	                                                               },
	                                                       });
	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		splat_targets[eye].view = vk::raii::ImageView(*device, vk::ImageViewCreateInfo{
		                                                                       .image = vk::Image(splat_image),
		                                                                       .viewType = vk::ImageViewType::e2DArray,
		                                                                       .format = vk::Format::eR16Sfloat,
		                                                                       .subresourceRange = {
		                                                                               .aspectMask = vk::ImageAspectFlagBits::eColor,
		                                                                               .baseMipLevel = 0,
		                                                                               .levelCount = 1,
		                                                                               .baseArrayLayer = eye,
		                                                                               .layerCount = 1,
		                                                                       },
		                                                               });
		splat_targets[eye].depth_view = vk::raii::ImageView(*device, vk::ImageViewCreateInfo{
		                                                                             .image = vk::Image(splat_depth_image),
		                                                                             .viewType = vk::ImageViewType::e2DArray,
		                                                                             .format = vk::Format::eD16Unorm,
		                                                                             .subresourceRange = {
		                                                                                     .aspectMask = vk::ImageAspectFlagBits::eDepth,
		                                                                                     .baseMipLevel = 0,
		                                                                                     .levelCount = 1,
		                                                                                     .baseArrayLayer = eye,
		                                                                                     .layerCount = 1,
		                                                                             },
		                                                                     });
		vk::ImageView fb_atts[2] = {*splat_targets[eye].view, *splat_targets[eye].depth_view};
		splat_targets[eye].framebuffer = vk::raii::Framebuffer(*device, vk::FramebufferCreateInfo{
		                                                                                .renderPass = *splat_renderpass,
		                                                                                .attachmentCount = 2,
		                                                                                .pAttachments = fb_atts,
		                                                                                .width = splat_extent_.width,
		                                                                                .height = splat_extent_.height,
		                                                                                .layers = 1,
		                                                                        });
	}
	spdlog::info("Depth splat targets: {}x{}", splat_extent_.width, splat_extent_.height);
}

int depth_overlay_renderer::acquire()
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

void depth_overlay_renderer::release()
{
	swapchain_.release();
}

vk::raii::Pipeline depth_overlay_renderer::make_pass2_pipeline(const char * frag_name, vk::PipelineLayout layout)
{
	auto vert = load_shader(*device, "depth_viz.vert");
	auto frag = load_shader(*device, frag_name);
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
	        .lineWidth = 1,
	};
	vk::PipelineMultisampleStateCreateInfo multisample{
	        .rasterizationSamples = vk::SampleCountFlagBits::e1,
	};
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
	        .layout = layout,
	        .renderPass = *renderpass,
	};
	return vk::raii::Pipeline(*device, nullptr, pipeline_info);
}

void depth_overlay_renderer::init_pipelines(vk::Format format)
{
	auto vert = load_shader(*device, "depth_viz.vert");
	auto frag = load_shader(*device, "depth_viz.frag");

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

	// Fullscreen triangle from gl_VertexIndex: no vertex input.
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
	        .lineWidth = 1,
	};

	vk::PipelineMultisampleStateCreateInfo multisample{
	        .rasterizationSamples = vk::SampleCountFlagBits::e1,
	};

	// Straight alpha blend over transparent (global opacity in the fragment push).
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

	vk::DescriptorSetLayoutBinding bindings[1] = {
	        {
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	vk::DescriptorSetLayoutCreateInfo set_layout_info{
	        .bindingCount = 1,
	        .pBindings = bindings,
	};
	// Free the old set before its layout and pool are retired: the set
	// references both, so destroying either first would strand it
	// (union_merge ensure() precedent).
	descriptor_set.clear();
	pipeline.clear();
	layer_pipeline.clear();
	splat_colorize_pipeline.clear();
	virt_plain_pipeline.clear();
	set_layout = vk::raii::DescriptorSetLayout(*device, set_layout_info);
	vk::DescriptorPoolSize pool_sizes[1] = {
	        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1},
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	        .maxSets = 1,
	        .poolSizeCount = 1,
	        .pPoolSizes = pool_sizes,
	};
	descriptor_pool = vk::raii::DescriptorPool(*device, pool_info);

	vk::DescriptorSetAllocateInfo alloc_info{
	        .descriptorPool = *descriptor_pool,
	        .descriptorSetCount = 1,
	        .pSetLayouts = &*set_layout,
	};
	auto sets = vk::raii::DescriptorSets(*device, alloc_info);
	descriptor_set = std::move(sets.front());

	// Depth is exact per-texel data (Godot samples NEAREST too):
	// filtering would smear silhouette edges into false ranges.
	vk::SamplerCreateInfo sampler_info{
	        .magFilter = vk::Filter::eNearest,
	        .minFilter = vk::Filter::eNearest,
	        .mipmapMode = vk::SamplerMipmapMode::eNearest,
	        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
	        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
	        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
	        .mipLodBias = 0,
	        .anisotropyEnable = VK_FALSE,
	        .compareEnable = VK_FALSE, // plain depth fetch, not a shadow compare
	        .minLod = 0,
	        .maxLod = 0,
	};
	sampler = vk::raii::Sampler(*device, sampler_info);

	// Separate splat-params UBO (272 B): the only viz UBO (pass-2
	// colorize params ride the fragment push block, not a buffer).
	if (not splat_ubo or splat_ubo.info().size < sizeof(splat_warp_params))
	{
		splat_ubo = buffer_allocation(
		        *device,
		        vk::BufferCreateInfo{
		                .size = sizeof(splat_warp_params),
		                .usage = vk::BufferUsageFlagBits::eUniformBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "depth overlay splat");
	}

	vk::PushConstantRange push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = sizeof(viz_push),
	};
	vk::PipelineLayoutCreateInfo layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*set_layout,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &push_range,
	};
	pipeline_layout = vk::raii::PipelineLayout(*device, layout_info);

	// CLEAR on UNDEFINED every frame (mirrors the gizmo renderer): no
	// layout history is ever assumed. Final GENERAL: universally
	// samplable at submit (mask precedent).
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
	renderpass = vk::raii::RenderPass(*device, renderpass_info);

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
	pipeline = vk::raii::Pipeline(*device, nullptr, pipeline_info);

	layer_pipeline = make_pass2_pipeline("layer_depth_viz.frag", *pipeline_layout);
	splat_colorize_pipeline = make_pass2_pipeline("splat_colorize.frag", *pipeline_layout);
	virt_plain_pipeline = make_pass2_pipeline("vr_depth_viz.frag", *pipeline_layout);

	// Two-image state shared by the verdict, raster-sensor, splat-compare
	// and plain vr_raster pipelines (same 2-image shape; a pipeline bakes
	// its set layout, so every pipeline builds against this one layout).
	// Teardown in dependency order (sets reference pool + set layout,
	// pipelines reference the pipeline layout).
	twoimg_set.clear();
	verdict_pipeline.clear();
	raster_sensor_pipeline.clear();
	splat_compare_pipeline.clear();
	vr_raster_plain_pipeline.clear();
	twoimg_pipeline_layout.clear();
	{
		vk::DescriptorSetLayoutBinding twoimg_bindings[2] = {
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
		twoimg_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
		                                                                   .bindingCount = 2,
		                                                                   .pBindings = twoimg_bindings,
		                                                           });
		vk::DescriptorPoolSize twoimg_pool_sizes[1] = {
		        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 2},
		};
		twoimg_pool = vk::raii::DescriptorPool(*device, vk::DescriptorPoolCreateInfo{
		                                                              .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		                                                              .maxSets = 1,
		                                                              .poolSizeCount = 1,
		                                                              .pPoolSizes = twoimg_pool_sizes,
		                                                      });
		auto twoimg_sets = vk::raii::DescriptorSets(*device, vk::DescriptorSetAllocateInfo{
		                                                                      .descriptorPool = *twoimg_pool,
		                                                                      .descriptorSetCount = 1,
		                                                                      .pSetLayouts = &*twoimg_layout,
		                                                              });
		twoimg_set = std::move(twoimg_sets.front());
	}
	{
		// Dedicated layout (a pipeline bakes its set layout, so the
		// two-image layout needs its own); reuses the 16 B viz_push range.
		twoimg_pipeline_layout = vk::raii::PipelineLayout(*device, vk::PipelineLayoutCreateInfo{
		                                                                          .setLayoutCount = 1,
		                                                                          .pSetLayouts = &*twoimg_layout,
		                                                                          .pushConstantRangeCount = 1,
		                                                                          .pPushConstantRanges = &push_range,
		                                                                  });
	}
	verdict_pipeline = make_pass2_pipeline("verdict_viz.frag", *twoimg_pipeline_layout);
	raster_sensor_pipeline = make_pass2_pipeline("raster_sensor_viz.frag", *twoimg_pipeline_layout);
	splat_compare_pipeline = make_pass2_pipeline("splat_compare.frag", *twoimg_pipeline_layout);
	vr_raster_plain_pipeline = make_pass2_pipeline("vr_raster_viz.frag", *twoimg_pipeline_layout);


	// Splat pass-1 state: vertex-stage image + UBO layout, own pool/set,
	// pipeline layout (vertex push block), pipeline (TRIANGLE_LIST
	// indexed mesh, depth test LESS + write, no blending) and renderpass
	// (R16F color + D16 depth). Teardown in dependency order (sets
	// reference pool + set layout, pipeline references pipeline layout).
	splat_set.clear();
	splat_pipeline.clear();
	splat_pipeline_layout.clear();
	splat_layout.clear();
	{
		vk::DescriptorSetLayoutBinding splat_bindings[2] = {
		        {
		                .binding = 0,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eVertex,
		        },
		        {
		                .binding = 1,
		                .descriptorType = vk::DescriptorType::eUniformBuffer,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eVertex,
		        },
		};
		vk::DescriptorSetLayoutCreateInfo splat_layout_info{
		        .bindingCount = 2,
		        .pBindings = splat_bindings,
		};
		splat_layout = vk::raii::DescriptorSetLayout(*device, splat_layout_info);
		vk::DescriptorPoolSize splat_pool_sizes[2] = {
		        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 3},
		        {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 3},
		};
		vk::DescriptorPoolCreateInfo splat_pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 3,
		        .poolSizeCount = 2,
		        .pPoolSizes = splat_pool_sizes,
		};
		// (Re)allocated below; clear first so no set outlives its pool
		// across the waitIdle'd rebuild (union_merge ensure() precedent).
		splat_set.clear();
		splat_fill_sets.clear();
		splat_pool = vk::raii::DescriptorPool(*device, splat_pool_info);
		vk::DescriptorSetAllocateInfo splat_alloc{
		        .descriptorPool = *splat_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*splat_layout,
		};
		auto splat_sets = vk::raii::DescriptorSets(*device, splat_alloc);
		splat_set = std::move(splat_sets.front());
		// Fill-dedicated ping-pong pair (A1: the mask-phase heightfield
		// fill and the viz splat share the transient across command
		// buffers on async frames — host writes to one set/UBO while
		// the GPU reads the other would corrupt the warp. Merge
		// descriptor precedent: alternate per call.)
		splat_fill_sets.clear();
		for (int i = 0; i < 2; ++i)
		{
			auto fill_sets = vk::raii::DescriptorSets(*device, splat_alloc);
			splat_fill_sets.emplace_back(std::move(fill_sets.front()));
		}
	}
	vk::PushConstantRange splat_push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eVertex,
	        .offset = 0,
	        .size = sizeof(splat_push),
	};
	{
		vk::PipelineLayoutCreateInfo splat_pl_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*splat_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &splat_push_range,
		};
		splat_pipeline_layout = vk::raii::PipelineLayout(*device, splat_pl_info);
	}
	{
		vk::AttachmentDescription splat_color{
		        .format = vk::Format::eR16Sfloat,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eClear,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eUndefined,
		        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		vk::AttachmentDescription splat_depth_att{
		        .format = vk::Format::eD16Unorm,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eClear,
		        .storeOp = vk::AttachmentStoreOp::eDontCare,
		        .initialLayout = vk::ImageLayout::eUndefined,
		        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::AttachmentDescription splat_atts[2] = {splat_color, splat_depth_att};
		vk::AttachmentReference splat_color_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference splat_depth_ref{
		        .attachment = 1,
		        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::SubpassDescription splat_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &splat_color_ref,
		        .pDepthStencilAttachment = &splat_depth_ref,
		};
		vk::RenderPassCreateInfo splat_rp_info{
		        .attachmentCount = 2,
		        .pAttachments = splat_atts,
		        .subpassCount = 1,
		        .pSubpasses = &splat_subpass,
		};
		splat_renderpass = vk::raii::RenderPass(*device, splat_rp_info);
	}
	{
		auto splat_vert = load_shader(*device, "sensor_splat.vert");
		auto splat_frag = load_shader(*device, "sensor_splat.frag");
		vk::PipelineShaderStageCreateInfo splat_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **splat_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **splat_frag,
		                .pName = "main",
		        },
		};
		vk::PipelineVertexInputStateCreateInfo splat_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo splat_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineDepthStencilStateCreateInfo splat_depth{
		        .depthTestEnable = VK_TRUE,
		        .depthWriteEnable = VK_TRUE,
		        .depthCompareOp = vk::CompareOp::eLess,
		};
		vk::PipelineColorBlendAttachmentState splat_blend_off{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo splat_blend{
		        .attachmentCount = 1,
		        .pAttachments = &splat_blend_off,
		};
		vk::GraphicsPipelineCreateInfo splat_pipeline_info{
		        .stageCount = 2,
		        .pStages = splat_stages,
		        .pVertexInputState = &splat_vertex_input,
		        .pInputAssemblyState = &splat_input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pDepthStencilState = &splat_depth,
		        .pColorBlendState = &splat_blend,
		        .pDynamicState = &dynamic,
		        .layout = *splat_pipeline_layout,
		        .renderPass = *splat_renderpass,
		};
		splat_pipeline = vk::raii::Pipeline(*device, nullptr, splat_pipeline_info);
	}
	// Tracked-hand draws (3d-depth-passthrough): LOAD twin of the splat
	// RP so the sensor splat survives (loadOp is not part of
	// compatibility, so the per-eye transient framebuffers are shared);
	// position-only vertex input, push-only layout (96B VERT, inside
	// the Vulkan minimum), same frag as the sensor splat.
	{
		vk::AttachmentDescription hand_color{
		        .format = vk::Format::eR16Sfloat,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eLoad,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentDescription hand_depth_att{
		        .format = vk::Format::eD16Unorm,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eLoad,
		        .storeOp = vk::AttachmentStoreOp::eDontCare,
		        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
		        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		        .initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::AttachmentDescription hand_atts[2] = {hand_color, hand_depth_att};
		vk::AttachmentReference hand_color_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference hand_depth_ref{
		        .attachment = 1,
		        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};
		vk::SubpassDescription hand_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &hand_color_ref,
		        .pDepthStencilAttachment = &hand_depth_ref,
		};
		vk::RenderPassCreateInfo hand_rp_info{
		        .attachmentCount = 2,
		        .pAttachments = hand_atts,
		        .subpassCount = 1,
		        .pSubpasses = &hand_subpass,
		};
		splat_load_rp = vk::raii::RenderPass(*device, hand_rp_info);
	}
	struct hand_pc
	{
		float view[16];
		float tan[4];
		float near_far[2];
		int32_t eye = 0;
		int32_t pad = 0;
	};
	static_assert(sizeof(hand_pc) == 96);
	vk::PushConstantRange hand_push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eVertex,
	        .offset = 0,
	        .size = sizeof(hand_pc),
	};
	{
		vk::PipelineLayoutCreateInfo hand_pl_info{
		        .setLayoutCount = 0,
		        .pSetLayouts = nullptr,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &hand_push_range,
		};
		hand_pipeline_layout = vk::raii::PipelineLayout(*device, hand_pl_info);
	}
	{
		auto hand_vert = load_shader(*device, "depth_hand_mv.vert");
		auto hand_frag = load_shader(*device, "sensor_splat.frag");
		vk::PipelineShaderStageCreateInfo hand_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **hand_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **hand_frag,
		                .pName = "main",
		        },
		};
		vk::VertexInputBindingDescription hand_binding{
		        .binding = 0,
		        .stride = sizeof(float) * 3,
		        .inputRate = vk::VertexInputRate::eVertex,
		};
		vk::VertexInputAttributeDescription hand_attribute{
		        .location = 0,
		        .binding = 0,
		        .format = vk::Format::eR32G32B32Sfloat,
		        .offset = 0,
		};
		vk::PipelineVertexInputStateCreateInfo hand_vertex_input{
		        .vertexBindingDescriptionCount = 1,
		        .pVertexBindingDescriptions = &hand_binding,
		        .vertexAttributeDescriptionCount = 1,
		        .pVertexAttributeDescriptions = &hand_attribute,
		};
		vk::PipelineInputAssemblyStateCreateInfo hand_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineDepthStencilStateCreateInfo hand_depth{
		        .depthTestEnable = VK_TRUE,
		        .depthWriteEnable = VK_TRUE,
		        .depthCompareOp = vk::CompareOp::eLess,
		};
		vk::PipelineColorBlendAttachmentState hand_blend_off{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo hand_blend{
		        .attachmentCount = 1,
		        .pAttachments = &hand_blend_off,
		};
		vk::GraphicsPipelineCreateInfo hand_pipeline_info{
		        .stageCount = 2,
		        .pStages = hand_stages,
		        .pVertexInputState = &hand_vertex_input,
		        .pInputAssemblyState = &hand_input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pDepthStencilState = &hand_depth,
		        .pColorBlendState = &hand_blend,
		        .pDynamicState = &dynamic,
		        .layout = *hand_pipeline_layout,
		        .renderPass = *splat_load_rp,
		};
		hand_pipeline = vk::raii::Pipeline(*device, nullptr, hand_pipeline_info);
	}

	// Converted YCbCr states bake this renderpass into their pipelines:
	// rebuild them against the new one (plain views need nothing — the
	// shared pipelines above are fresh). waitIdle'd by the caller.
	if (*virt_layout)
		create_converted_state(virt_ycbcr_built, virt_immutable_src, false);
	if (*vr_raster_layout)
		create_converted_state(vr_raster_ycbcr_built, vr_raster_immutable_src, true);
}

void depth_overlay_renderer::record(vk::raii::CommandBuffer & cmd, size_t image_index,
                                    const std::array<vk::Extent2D, 2> & extents,
                                    vk::ImageView depth_view,
                                    const viz_params & params)
{
	if (image_index >= targets.size())
		return;

	vk::DescriptorImageInfo image_info{
	        .sampler = *sampler,
	        .imageView = depth_view,
	        // The runtime hands the acquired image ready to sample
	        // (acquire is the sync point); keep whatever layout it is in.
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	// Full footprint every record (Adreno partial-write rule).
	std::array<vk::WriteDescriptorSet, 1> writes = {{
	        {
	                .dstSet = *descriptor_set,
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &image_info,
	        },
	}};
	device->updateDescriptorSets(writes, nullptr);

	// Fresh CLEAR images every frame: no barrier, no layout history.
	draw_viz_eyes(cmd, *pipeline_layout, *pipeline, *descriptor_set, extents, image_index,
	              params, true);
}

void depth_overlay_renderer::draw_viz_eyes(vk::raii::CommandBuffer & cmd, vk::PipelineLayout layout, vk::Pipeline pipeline,
                                           vk::DescriptorSet set, const std::array<vk::Extent2D, 2> & extents, size_t image_index,
                                           const viz_params & params, bool have_data)
{
	if (image_index >= targets.size())
		return;
	if (have_data)
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, set, {});
	}
	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = targets[image_index][eye];
		if (have_data)
			cmd.pushConstants<viz_push>(layout, vk::ShaderStageFlagBits::eFragment, 0,
			                            viz_push{(int32_t)eye, params.near_z, params.far_z, params.opacity});
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)extents[eye].width,
		                        .height = (float)extents[eye].height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extents[eye]});
		vk::ClearValue clear{};
		clear.color.float32.fill(0);
		if (not have_data)
			clear = black_clear(params.opacity);
		vk::RenderPassBeginInfo begin_info{
		        .renderPass = *renderpass,
		        .framebuffer = *tgt.framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = extents[eye]},
		        .clearValueCount = 1,
		        .pClearValues = &clear,
		};
		cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
		if (have_data)
			cmd.draw(3, 1, 0, 0);
		cmd.endRenderPass();
	}
}


void depth_overlay_renderer::retire_converted_state(bool with_layer)
{
	// Teardown in dependency order: sets reference pool + set layout,
	// pipelines reference the pipeline layout. Pool destruction follows
	// the set (an abandoned set would strand it).
	auto & set = with_layer ? vr_raster_set : virt_set;
	auto & pipeline = with_layer ? vr_raster_pipeline : virt_pipeline;
	auto & pipeline_layout = with_layer ? vr_raster_pipeline_layout : virt_pipeline_layout;
	auto & layout = with_layer ? vr_raster_layout : virt_layout;
	auto & pool = with_layer ? vr_raster_pool : virt_pool;
	auto & built = with_layer ? vr_raster_ycbcr_built : virt_ycbcr_built;
	auto & immutable_src = with_layer ? vr_raster_immutable_src : virt_immutable_src;
	set.clear();
	pipeline.clear();
	pipeline_layout.clear();
	layout.clear();
	pool.clear();
	built = wivrn::decoder::ycbcr_definition{};
	immutable_src = vk::Sampler{};
}

void depth_overlay_renderer::create_converted_state(wivrn::decoder::ycbcr_definition def, vk::Sampler generation_sampler, bool with_layer)
{
	// Converted views only (plain views use the shared pipelines on the
	// shared layouts). Caller guarantees no in-flight readers (a
	// waitIdle'd rebuild).
	retire_converted_state(with_layer);
	// Bake the handle's generation sampler as immutable (a separately
	// built conversion proved unidentifiable to validation despite
	// matching parameters, so the layout shares the handle's object
	// instead of owning one). The second (layer) binding stays mutable
	// (plain R16F array view).
	auto & immutable_src = with_layer ? vr_raster_immutable_src : virt_immutable_src;
	immutable_src = generation_sampler;
	vk::Sampler immutable = immutable_src;
	vk::DescriptorSetLayoutBinding bindings[2] = {
	        {
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	                .pImmutableSamplers = &immutable,
	        },
	        {
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	const uint32_t binding_count = with_layer ? 2 : 1;
	auto & layout = with_layer ? vr_raster_layout : virt_layout;
	layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                   .bindingCount = binding_count,
	                                                                   .pBindings = bindings,
	                                                           });
	vk::DescriptorPoolSize pool_sizes[1] = {
	        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = binding_count},
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	        .maxSets = 1,
	        .poolSizeCount = 1,
	        .pPoolSizes = pool_sizes,
	};
	auto & pool = with_layer ? vr_raster_pool : virt_pool;
	pool = vk::raii::DescriptorPool(*device, pool_info);
	auto & set = with_layer ? vr_raster_set : virt_set;
	vk::DescriptorSetAllocateInfo set_alloc{
	        .descriptorPool = *pool,
	        .descriptorSetCount = 1,
	        .pSetLayouts = &*layout,
	};
	auto sets = vk::raii::DescriptorSets(*device, set_alloc);
	set = std::move(sets.front());
	// Pipeline bakes the set layout: rebuilt alongside it, never apart.
	// Reuses the 16 B viz_push range.
	vk::PushConstantRange push{
	        .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = sizeof(viz_push),
	};
	vk::PipelineLayoutCreateInfo layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*layout,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &push,
	};
	auto & pipeline_layout = with_layer ? vr_raster_pipeline_layout : virt_pipeline_layout;
	pipeline_layout = vk::raii::PipelineLayout(*device, layout_info);
	auto & pipeline = with_layer ? vr_raster_pipeline : virt_pipeline;
	pipeline = make_pass2_pipeline(with_layer ? "vr_raster_viz.frag" : "vr_depth_viz.frag", *pipeline_layout);
	auto & built = with_layer ? vr_raster_ycbcr_built : virt_ycbcr_built;
	built = def;
}

void depth_overlay_renderer::ensure_virt_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler)
{
	if (not def.has_conversion)
	{
		// Plain views use the shared base state: retire any converted
		// state (decoder generation changed converted->plain). Rare.
		if (*virt_layout)
		{
			device->waitIdle();
			retire_converted_state(false);
		}
		return;
	}
	// Rebuild when the definition changes OR the handle sampler is a
	// new generation (the baked immutable handle must track the sampled
	// generation).
	if (virt_ycbcr_built == def and virt_immutable_src == generation_sampler)
		return;
	// Depth format change (rare: renegotiation only): retire prior
	// submissions before rebuilding what they may still read.
	device->waitIdle();
	create_converted_state(def, generation_sampler, false);
	spdlog::info("Depth overlay virt layout rebuilt (YCbCr immutable, handle-generation sampler)");
}

void depth_overlay_renderer::ensure_vr_raster_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler)
{
	if (not def.has_conversion)
	{
		// Plain views use the shared two-image state: retire any
		// converted state (decoder generation changed
		// converted->plain). Rare.
		if (*vr_raster_layout)
		{
			device->waitIdle();
			retire_converted_state(true);
		}
		return;
	}
	// Rebuild when the definition changes OR the handle sampler is a
	// new generation (the baked immutable handle must track the sampled
	// generation).
	if (vr_raster_ycbcr_built == def and vr_raster_immutable_src == generation_sampler)
		return;
	// Depth format change (rare: renegotiation only): retire prior
	// submissions before rebuilding what they may still read.
	device->waitIdle();
	create_converted_state(def, generation_sampler, true);
	spdlog::info("Depth overlay VR+raster layout rebuilt (YCbCr immutable, handle-generation sampler)");
}


void depth_overlay_renderer::upload_splat_params(buffer_allocation & ubo, const splat_warp_params & params)
{
	std::memcpy(ubo.map(), &params, sizeof(params));
	ubo.unmap();
	// Non-coherent host-visible memory: the GPU reads stale values
	// forever without an explicit flush (vmaUnmapMemory does not flush).
	vmaFlushAllocation(vk_allocator::instance(), ubo, 0, sizeof(params));
}

void depth_overlay_renderer::splat_pass1(vk::raii::CommandBuffer & cmd,
                                         vk::ImageView depth_view,
                                         int sens_w, int sens_h,
                                         vk::DescriptorSet set,
                                         buffer_allocation & ubo)
{
	vk::DescriptorImageInfo image_info{
	        .sampler = *sampler,
	        .imageView = depth_view,
	        // The runtime hands the acquired image ready to sample
	        // (acquire is the sync point); keep whatever layout it is in.
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	vk::DescriptorBufferInfo buffer_info{
	        .buffer = vk::Buffer(ubo),
	        .offset = 0,
	        .range = sizeof(splat_warp_params),
	};
	// Full footprint every record (Adreno partial-write rule).
	std::array<vk::WriteDescriptorSet, 2> writes = {{
	        {
	                .dstSet = set,
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &image_info,
	        },
	        {
	                .dstSet = set,
	                .dstBinding = 1,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .pBufferInfo = &buffer_info,
	        },
	}};
	device->updateDescriptorSets(writes, nullptr);

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *splat_pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *splat_pipeline_layout, 0, set, {});

	// Clear color is FAR (65504 saturates the pass-2 ramp to red) with
	// clear depth at far (1.0): uncovered pixels keep both and read as
	// honest FAR holes. Depth test LESS keeps the nearest splat.
	std::array<vk::ClearValue, 2> clears{};
	clears[0].color.float32 = std::array<float, 4>{65504.f, 0.f, 0.f, 0.f};
	clears[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
	ensure_splat_index(sens_w, sens_h);
	bool indexed = (bool)splat_index and sens_w >= 2 and sens_h >= 2;
	uint32_t index_count = indexed ? (uint32_t)(sens_w - 1) * (uint32_t)(sens_h - 1) * 6u : 0;
	if (indexed)
		cmd.bindIndexBuffer(vk::Buffer(splat_index), 0, vk::IndexType::eUint32);
	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = splat_targets[eye];
		cmd.pushConstants<splat_push>(*splat_pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0,
		                              splat_push{(int32_t)eye, (int32_t)sens_w, (int32_t)sens_h});
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)splat_extent_.width,
		                        .height = (float)splat_extent_.height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = splat_extent_});
		vk::RenderPassBeginInfo begin_info{
		        .renderPass = *splat_renderpass,
		        .framebuffer = *tgt.framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = splat_extent_},
		        .clearValueCount = (uint32_t)clears.size(),
		        .pClearValues = clears.data(),
		};
		cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
		if (indexed)
			cmd.drawIndexed(index_count, 1, 0, 0, 0);
		cmd.endRenderPass();
	}
}

void depth_overlay_renderer::ensure_splat_index(int sens_w, int sens_h)
{
	if (sens_w == splat_idx_w and sens_h == splat_idx_h and (bool)splat_index)
		return;
	// waitIdle'd here: sensor-size changes are rare (provider restart),
	// and the rebuild must retire in-flight reads of the old index.
	// Host-visible fill + flush, no staging (UBO pattern above).
	device->waitIdle();
	splat_index = buffer_allocation{};
	splat_idx_w = sens_w;
	splat_idx_h = sens_h;
	if (sens_w < 2 or sens_h < 2)
		return;
	// Center grid: sens_w*sens_h vertices at texel centers,
	// (sens_w-1)*(sens_h-1) quads. Degenerate lines stay no-mesh.
	uint32_t quads_w = (uint32_t)sens_w - 1;
	uint32_t quads_h = (uint32_t)sens_h - 1;
	vk::DeviceSize index_count = vk::DeviceSize(quads_w) * vk::DeviceSize(quads_h) * 6;
	splat_index = buffer_allocation(
	        *device,
	        vk::BufferCreateInfo{
	                .size = index_count * sizeof(uint32_t),
	                .usage = vk::BufferUsageFlagBits::eIndexBuffer,
	        },
	        VmaAllocationCreateInfo{
	                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "depth splat index");
	auto * out = static_cast<uint32_t *>(splat_index.map());
	size_t m = 0;
	for (int32_t y = 0; y < sens_h - 1; ++y)
	{
		for (int32_t x = 0; x < sens_w - 1; ++x)
		{
			// TL-first in both triangles: the TL corner is the
			// provoking vertex whose flat background depth the fragment
			// reads. Order is load-bearing, keep it.
			uint32_t a = (uint32_t)y * (uint32_t)sens_w + (uint32_t)x;
			uint32_t b = a + 1;
			uint32_t c = a + (uint32_t)sens_w;
			uint32_t d = c + 1;
			out[m++] = a;
			out[m++] = b;
			out[m++] = d;
			out[m++] = a;
			out[m++] = c;
			out[m++] = d;
		}
	}
	splat_index.unmap();
	vmaFlushAllocation(vk_allocator::instance(), splat_index, 0, index_count * sizeof(uint32_t));
	spdlog::info("Depth splat index: {}x{} ({} tris)", sens_w, sens_h, quads_w * quads_h * 2);
}

void depth_overlay_renderer::splat_pass1_and_barrier(vk::raii::CommandBuffer & cmd,
                                                         vk::ImageView sensor_view,
                                                         const splat_warp_params & params,
                                                         int sens_w, int sens_h)
{
	upload_splat_params(splat_ubo, params);
	splat_pass1(cmd, sensor_view, sens_w, sens_h, *splat_set, splat_ubo);
	splat_read_barrier(cmd, vk::Image(splat_image));
}

void depth_overlay_renderer::fill_sync_heightfield(vk::raii::CommandBuffer & cmd,
                                                  vk::ImageView depth_view,
                                                  const splat_warp_params & params,
                                                  int sens_w, int sens_h,
                                                  const hand_splat * hands)
{
	// Fill-dedicated ping-pong slot (A1): the mask-phase fill and the
	// viz splat share the transient across command buffers on async
	// frames — host writes to one set/UBO while the GPU reads the other
	// would corrupt the warp. Merge descriptor precedent.
	splat_fill_parity = !splat_fill_parity;
	auto & fill_ubo = splat_fill_ubo[splat_fill_parity ? 1 : 0];
	if (not fill_ubo or fill_ubo.info().size < sizeof(splat_warp_params))
	{
		fill_ubo = buffer_allocation(
		        *device,
		        vk::BufferCreateInfo{
		                .size = sizeof(splat_warp_params),
		                .usage = vk::BufferUsageFlagBits::eUniformBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "depth fill splat params");
	}
	upload_splat_params(fill_ubo, params);
	splat_pass1(cmd, depth_view, sens_w, sens_h, *splat_fill_sets[splat_fill_parity ? 1 : 0], fill_ubo);
	if (hands and hands->index_count > 0 and (vk::Buffer)hands->verts and (vk::Buffer)hands->indices)
	{
		// Sensor splat RP ends SHADER_READ: transition back for the
		// hand LOAD passes, then hand the transient back SHADER_READ
		// for the reveal/write-back taps below.
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
		                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                    {},
		                    {},
		                    {},
		                    vk::ImageMemoryBarrier{
		                            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
		                            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		                            .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		                            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
		                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                            .image = vk::Image(splat_image),
		                            .subresourceRange = {
		                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
		                                    .levelCount = 1,
		                                    .layerCount = 2,
		                            },
		                    });
		struct hand_push
		{
			float view[16];
			float tan[4];
			float near_far[2];
			int32_t eye = 0;
			int32_t pad = 0;
		};
		static_assert(sizeof(hand_push) == 96);
		for (uint32_t eye = 0; eye < 2; ++eye)
		{
			auto & tgt = splat_targets[eye];
			hand_push pc{};
			std::memcpy(pc.view, hands->view[eye], sizeof(pc.view));
			std::memcpy(pc.tan, hands->tan[eye], sizeof(pc.tan));
			pc.near_far[0] = hands->near_far[0];
			pc.near_far[1] = hands->near_far[1];
			pc.eye = (int32_t)eye;
			vk::RenderPassBeginInfo begin_info{
			        .renderPass = *splat_load_rp,
			        .framebuffer = *tgt.framebuffer,
			        .renderArea = {.offset = {0, 0}, .extent = splat_extent_},
			        .clearValueCount = 0,
			        .pClearValues = nullptr,
			};
			cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *hand_pipeline);
			cmd.bindVertexBuffers(0, hands->verts, (vk::DeviceSize)0);
			cmd.bindIndexBuffer(hands->indices, 0, vk::IndexType::eUint16);
			cmd.setViewport(0, vk::Viewport{
			                        .x = 0,
			                        .y = 0,
			                        .width = (float)splat_extent_.width,
			                        .height = (float)splat_extent_.height,
			                        .minDepth = 0,
			                        .maxDepth = 1,
			                });
			cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = splat_extent_});
			cmd.pushConstants<hand_push>(*hand_pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, pc);
			cmd.drawIndexed(hands->index_count, 1, 0, 0, 0);
			cmd.endRenderPass();
		}
		// Hand passes end COLOR_ATTACHMENT: hand back SHADER_READ for
		// the reveal/write-back taps (same access transition the
		// splat-read barrier below encodes for the no-hands path).
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                    vk::PipelineStageFlagBits::eFragmentShader,
		                    {},
		                    {},
		                    {},
		                    vk::ImageMemoryBarrier{
		                            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		                            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
		                            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
		                            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                            .image = vk::Image(splat_image),
		                            .subresourceRange = {
		                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
		                                    .levelCount = 1,
		                                    .layerCount = 2,
		                            },
		                    });
		return;
	}
	splat_read_barrier(cmd, vk::Image(splat_image));
}

vk::ImageView depth_overlay_renderer::heightfield_view() const
{
	return *splat_view_array;
}

void depth_overlay_renderer::record_sensor_splat(vk::raii::CommandBuffer & cmd, size_t image_index,
                                                 const std::array<vk::Extent2D, 2> & extents,
                                                 vk::ImageView depth_view,
                                                 const splat_warp_params & params,
                                                 int sens_w, int sens_h)
{
	if (image_index >= targets.size() or sens_w <= 0 or sens_h <= 0)
		return;

	splat_pass1_and_barrier(cmd, depth_view, params, sens_w, sens_h);

	// Pass 2: false-color the splatted depth (base 1-image set, splat
	// transient sampled in place of the sensor view).
	vk::DescriptorImageInfo image_info{
	        .sampler = *sampler,
	        .imageView = *splat_view_array,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	// Full footprint every record (Adreno partial-write rule).
	std::array<vk::WriteDescriptorSet, 1> writes = {{
	        {
	                .dstSet = *descriptor_set,
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &image_info,
	        },
	}};
	device->updateDescriptorSets(writes, nullptr);

	// Fresh CLEAR images every frame: no barrier, no layout history.
	draw_viz_eyes(cmd, *pipeline_layout, *splat_colorize_pipeline, *descriptor_set, extents, image_index,
	              viz_params{params.near_z, params.far_z, params.opacity}, true);
}


void depth_overlay_renderer::record_layer(vk::raii::CommandBuffer & cmd, size_t image_index,
                                          const std::array<vk::Extent2D, 2> & extents,
                                          vk::Image layer_image, vk::ImageView layer_view,
                                          const viz_params & params)
{
	if (image_index >= targets.size())
		return;

	{
		// Pack valid images front for the cross-CB read barrier.
		vk::Image packed[1];
		size_t m = 0;
		if ((bool)layer_view and layer_image != vk::Image{})
			packed[m++] = layer_image;
		viz_read_barrier(cmd, std::span<const vk::Image>(packed, m));
	}

	bool have_data = (bool)layer_view and layer_image != vk::Image{};
	if (have_data)
	{
		vk::DescriptorImageInfo image_info{
		        .sampler = *sampler,
		        .imageView = layer_view,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		// Full footprint every record (Adreno partial-write rule).
		std::array<vk::WriteDescriptorSet, 1> writes = {{
		        {
		                .dstSet = *descriptor_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &image_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
	}

	draw_viz_eyes(cmd, *pipeline_layout, *layer_pipeline, *descriptor_set, extents, image_index,
	              params, have_data);
}


void depth_overlay_renderer::record_verdict(vk::raii::CommandBuffer & cmd, size_t image_index,
                                            const std::array<vk::Extent2D, 2> & extents,
                                            vk::Image verdict_image, vk::ImageView verdict_view,
                                            vk::Image layer_image, vk::ImageView layer_view,
                                            const viz_params & params)
{
	if (image_index >= targets.size())
		return;

	{
		// Pack valid images front for the cross-CB read barrier.
		vk::Image packed[2];
		size_t m = 0;
		if ((bool)verdict_view and verdict_image != vk::Image{})
			packed[m++] = verdict_image;
		if ((bool)layer_view and layer_image != vk::Image{})
			packed[m++] = layer_image;
		viz_read_barrier(cmd, std::span<const vk::Image>(packed, m));
	}

	bool have_data = (bool)verdict_view and (bool)layer_view;
	if (have_data)
	{
		vk::DescriptorImageInfo verdict_info{
		        .sampler = *sampler,
		        .imageView = verdict_view,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		vk::DescriptorImageInfo layer_info{
		        .sampler = *sampler,
		        .imageView = layer_view,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		// Full footprint every record (Adreno partial-write rule).
		std::array<vk::WriteDescriptorSet, 2> writes = {{
		        {
		                .dstSet = *twoimg_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &verdict_info,
		        },
		        {
		                .dstSet = *twoimg_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &layer_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
	}

	draw_viz_eyes(cmd, *twoimg_pipeline_layout, *verdict_pipeline, *twoimg_set, extents, image_index,
	              params, have_data);
}


void depth_overlay_renderer::record_raster_sensor(vk::raii::CommandBuffer & cmd, size_t image_index,
                                                                  const std::array<vk::Extent2D, 2> & extents,
                                                                  vk::Image layer_image, vk::ImageView layer_view,
                                                                  vk::ImageView sensor_view,
                                                                  const viz_params & params)
{
	if (image_index >= targets.size())
		return;

	{
		// Pack valid images front for the cross-CB read barrier. The
		// sensor view rides no barrier (the runtime hands the acquired
		// image ready to sample, same as the sensor path).
		vk::Image packed[1];
		size_t m = 0;
		if ((bool)layer_view and layer_image != vk::Image{})
			packed[m++] = layer_image;
		viz_read_barrier(cmd, std::span<const vk::Image>(packed, m));
	}

	bool have_data = (bool)layer_view and layer_image != vk::Image{} and (bool)sensor_view;
	if (have_data)
	{
		vk::DescriptorImageInfo layer_info{
		        .sampler = *sampler,
		        .imageView = layer_view,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		vk::DescriptorImageInfo sensor_info{
		        .sampler = *sampler,
		        .imageView = sensor_view,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		// Full footprint every record (Adreno partial-write rule).
		std::array<vk::WriteDescriptorSet, 2> writes = {{
		        {
		                .dstSet = *twoimg_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &layer_info,
		        },
		        {
		                .dstSet = *twoimg_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &sensor_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
	}

	draw_viz_eyes(cmd, *twoimg_pipeline_layout, *raster_sensor_pipeline, *twoimg_set, extents, image_index,
	              params, have_data);
}


void depth_overlay_renderer::record_raster_splat(vk::raii::CommandBuffer & cmd, size_t image_index,
                                                  const std::array<vk::Extent2D, 2> & extents,
                                                  vk::Image layer_image, vk::ImageView layer_view,
                                                  vk::ImageView sensor_view,
                                                  const splat_warp_params & params,
                                                  int sens_w, int sens_h)
{
	if (image_index >= targets.size() or sens_w <= 0 or sens_h <= 0)
		return;

	splat_pass1_and_barrier(cmd, sensor_view, params, sens_w, sens_h);
	{
		// Pack valid images front for the cross-CB read barrier. The
		// sensor view rides no barrier (the runtime hands the acquired
		// image ready to sample, same as the sensor path).
		vk::Image packed[1];
		size_t m = 0;
		if ((bool)layer_view and layer_image != vk::Image{})
			packed[m++] = layer_image;
		viz_read_barrier(cmd, std::span<const vk::Image>(packed, m));
	}

	bool have_data = (bool)layer_view and layer_image != vk::Image{} and (bool)sensor_view;
	if (have_data)
	{
		vk::DescriptorImageInfo layer_info{
		        .sampler = *sampler,
		        .imageView = layer_view,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		vk::DescriptorImageInfo splat_info{
		        .sampler = *sampler,
		        .imageView = *splat_view_array,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		// Full footprint every record (Adreno partial-write rule).
		std::array<vk::WriteDescriptorSet, 2> writes = {{
		        {
		                .dstSet = *twoimg_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &layer_info,
		        },
		        {
		                .dstSet = *twoimg_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &splat_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
	}

	draw_viz_eyes(cmd, *twoimg_pipeline_layout, *splat_compare_pipeline, *twoimg_set, extents, image_index,
	              viz_params{params.near_z, params.far_z, params.opacity}, have_data);
}


void depth_overlay_renderer::record_virt(vk::raii::CommandBuffer & cmd, size_t image_index,
                                         const std::array<vk::Extent2D, 2> & extents,
                                         vk::ImageView virt_view, vk::Sampler virt_sampler,
                                         const wivrn::decoder::ycbcr_definition & ycbcr,
                                         const viz_params & params)
{
	if (image_index >= targets.size())
		return;

	bool have_data = (bool)virt_view and (bool)virt_sampler;
	if (have_data)
	{
		// YCbCr immutable-sampler layout (VUID-02738): rebuild when the
		// view's conversion definition or the handle's generation
		// sampler changes. waitIdle'd, rare. Plain views use the shared
		// base state (no per-definition objects).
		ensure_virt_layout(ycbcr, virt_sampler);
		// Decoder SBS image: GENERAL layout at frame top (same
		// steady-state discipline as the video path). The write sampler
		// is ignored when the layout baked an immutable one; pass the
		// handle's generation sampler either way.
		vk::DescriptorImageInfo image_info{
		        .sampler = virt_sampler,
		        .imageView = virt_view,
		        .imageLayout = vk::ImageLayout::eGeneral,
		};
		// Full footprint every record (Adreno partial-write rule).
		std::array<vk::WriteDescriptorSet, 1> writes = {{
		        {
		                .dstSet = ycbcr.has_conversion ? *virt_set : *descriptor_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &image_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
		if (ycbcr.has_conversion)
			draw_viz_eyes(cmd, *virt_pipeline_layout, *virt_pipeline, *virt_set, extents, image_index,
			              params, true);
		else
			draw_viz_eyes(cmd, *pipeline_layout, *virt_plain_pipeline, *descriptor_set, extents, image_index,
			              params, true);
		return;
	}

	draw_viz_eyes(cmd, *pipeline_layout, *virt_plain_pipeline, *descriptor_set, extents, image_index,
	              params, false);
}


void depth_overlay_renderer::record_vr_raster(vk::raii::CommandBuffer & cmd, size_t image_index,
                                              const std::array<vk::Extent2D, 2> & extents,
                                              vk::ImageView virt_view, vk::Sampler virt_sampler,
                                              const wivrn::decoder::ycbcr_definition & ycbcr,
                                              vk::Image layer_image, vk::ImageView layer_view,
                                              const viz_params & params)
{
	if (image_index >= targets.size())
		return;

	{
		// Pack valid images front for the cross-CB read barrier. The
		// decoder SBS image needs none (GENERAL steady-state, same as
		// the VR-depth path).
		vk::Image packed[1];
		size_t m = 0;
		if ((bool)layer_view and layer_image != vk::Image{})
			packed[m++] = layer_image;
		viz_read_barrier(cmd, std::span<const vk::Image>(packed, m));
	}

	// Virt is required (it is the base image); without it the frame
	// records opaque black. A null layer falls through to the dedicated
	// VR-depth path (same image, no overlay): unbound image bindings
	// with a null view are invalid, so the layer tap needs a valid view.
	bool have_data = (bool)virt_view and (bool)virt_sampler;
	bool have_layer = (bool)layer_view and layer_image != vk::Image{};
	if (have_data and not have_layer)
	{
		// No layer to composite: plain VR depth through the dedicated
		// path (same image, no overlay).
		record_virt(cmd, image_index, extents, virt_view, virt_sampler, ycbcr, params);
		return;
	}
	if (have_data)
	{
		// YCbCr immutable-sampler layout (VUID-02738): rebuild when the
		// view's conversion definition or the handle's generation
		// sampler changes. waitIdle'd, rare. Plain views use the shared
		// two-image state (no per-definition objects).
		ensure_vr_raster_layout(ycbcr, virt_sampler);
		// Decoder SBS image: GENERAL layout at frame top (same
		// steady-state discipline as the video path). The write sampler
		// is ignored when the layout baked an immutable one; pass the
		// handle's generation sampler either way.
		vk::DescriptorImageInfo virt_info{
		        .sampler = virt_sampler,
		        .imageView = virt_view,
		        .imageLayout = vk::ImageLayout::eGeneral,
		};
		vk::DescriptorImageInfo valid_layer_info{
		        .sampler = *sampler,
		        .imageView = layer_view,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		// Full footprint every record (Adreno partial-write rule).
		std::array<vk::WriteDescriptorSet, 2> writes = {{
		        {
		                .dstSet = ycbcr.has_conversion ? *vr_raster_set : *twoimg_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &virt_info,
		        },
		        {
		                .dstSet = ycbcr.has_conversion ? *vr_raster_set : *twoimg_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &valid_layer_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
		if (ycbcr.has_conversion)
			draw_viz_eyes(cmd, *vr_raster_pipeline_layout, *vr_raster_pipeline, *vr_raster_set, extents, image_index,
			              params, true);
		else
			draw_viz_eyes(cmd, *twoimg_pipeline_layout, *vr_raster_plain_pipeline, *twoimg_set, extents, image_index,
			              params, true);
		return;
	}

	draw_viz_eyes(cmd, *twoimg_pipeline_layout, *vr_raster_plain_pipeline, *twoimg_set, extents, image_index,
	              params, false);
}
