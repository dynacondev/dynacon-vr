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

namespace
{
struct push
{
	int32_t eye;
};
} // namespace

static_assert(sizeof(depth_overlay_renderer::frame_params) == 16);

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
	// Warp framebuffers share the color views above (declared after
	// targets so they die first).
	ensure_warp_targets();
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

	// Straight alpha blend over transparent (global opacity in the UBO).
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

	vk::DescriptorSetLayoutBinding bindings[2] = {
	        {
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	        {
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	vk::DescriptorSetLayoutCreateInfo set_layout_info{
	        .bindingCount = 2,
	        .pBindings = bindings,
	};
	// Free the old set before its layout and pool are retired: the set
	// references both, so destroying either first would strand it
	// (union_merge ensure() precedent).
	descriptor_set.clear();
	pipeline.clear();
	set_layout = vk::raii::DescriptorSetLayout(*device, set_layout_info);
	vk::DescriptorPoolSize pool_sizes[2] = {
	        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 8},
	        {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 4},
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	        .maxSets = 4,
	        .poolSizeCount = 2,
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

	if (not ubo or ubo.info().size < sizeof(frame_params))
	{
		ubo = buffer_allocation(
		        *device,
		        vk::BufferCreateInfo{
		                .size = sizeof(frame_params),
		                .usage = vk::BufferUsageFlagBits::eUniformBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "depth overlay");
	}

	vk::PushConstantRange push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = sizeof(int32_t),
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

	// Isolation viz pipelines share the sensor state above (same set
	// layout, pipeline layout with the eye push, renderpass, UBO): only
	// the fragment module differs. No mat3/ternaries on this driver path.
	auto make_plain_pipeline = [&](const char * frag_name) {
		auto frag = load_shader(*device, frag_name);
		vk::PipelineShaderStageCreateInfo plain_stages[2] = {
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
		vk::GraphicsPipelineCreateInfo plain_info{
		        .stageCount = 2,
		        .pStages = plain_stages,
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
		return vk::raii::Pipeline(*device, nullptr, plain_info);
	};
	layer_pipeline = make_plain_pipeline("layer_depth_viz.frag");

	// Verdict viz samples two array images (verdict + layer for the
	// no-window mask) plus the UBO: own layout/pool/set, same push +
	// renderpass + blend as the sensor path.
	verdict_viz_set.clear();
	verdict_pipeline.clear();
	verdict_viz_pipeline_layout.clear();
	{
		vk::DescriptorSetLayoutBinding viz_bindings[3] = {
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
		                .binding = 2,
		                .descriptorType = vk::DescriptorType::eUniformBuffer,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		};
		vk::DescriptorSetLayoutCreateInfo viz_layout_info{
		        .bindingCount = 3,
		        .pBindings = viz_bindings,
		};
		verdict_viz_layout = vk::raii::DescriptorSetLayout(*device, viz_layout_info);
		vk::DescriptorPoolSize viz_pool_sizes[2] = {
		        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 2},
		        {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1},
		};
		vk::DescriptorPoolCreateInfo viz_pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 1,
		        .poolSizeCount = 2,
		        .pPoolSizes = viz_pool_sizes,
		};
		verdict_viz_pool = vk::raii::DescriptorPool(*device, viz_pool_info);
		vk::DescriptorSetAllocateInfo viz_alloc{
		        .descriptorPool = *verdict_viz_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*verdict_viz_layout,
		};
		auto viz_sets = vk::raii::DescriptorSets(*device, viz_alloc);
		verdict_viz_set = std::move(viz_sets.front());
	}
	{
		auto viz_frag = load_shader(*device, "verdict_viz.frag");
		vk::PipelineShaderStageCreateInfo viz_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **viz_frag,
		                .pName = "main",
		        },
		};
		vk::PipelineLayoutCreateInfo viz_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*verdict_viz_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &push_range,
		};
		// Reuse pipeline_layout's push shape via a dedicated layout
		// (a pipeline bakes its set layout, so the two-image layout
		// needs its own).
		verdict_viz_pipeline_layout = vk::raii::PipelineLayout(*device, viz_layout_info);
		vk::GraphicsPipelineCreateInfo viz_pipeline_info{
		        .stageCount = 2,
		        .pStages = viz_stages,
		        .pVertexInputState = &vertex_input,
		        .pInputAssemblyState = &input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pColorBlendState = &blend,
		        .pDynamicState = &dynamic,
		        .layout = *verdict_viz_pipeline_layout,
		        .renderPass = *renderpass,
		};
		verdict_pipeline = vk::raii::Pipeline(*device, nullptr, viz_pipeline_info);
	}

	// Raster-vs-sensor viz: same two-image + UBO shape as the verdict
	// viz (verdict pattern above), own objects so both modes stay bound
	// independently.
	raster_sensor_set.clear();
	raster_sensor_pipeline.clear();
	raster_sensor_pipeline_layout.clear();
	{
		vk::DescriptorSetLayoutBinding rs_bindings[3] = {
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
		                .binding = 2,
		                .descriptorType = vk::DescriptorType::eUniformBuffer,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		};
		vk::DescriptorSetLayoutCreateInfo rs_layout_info{
		        .bindingCount = 3,
		        .pBindings = rs_bindings,
		};
		raster_sensor_layout = vk::raii::DescriptorSetLayout(*device, rs_layout_info);
		vk::DescriptorPoolSize rs_pool_sizes[2] = {
		        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 2},
		        {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1},
		};
		vk::DescriptorPoolCreateInfo rs_pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 1,
		        .poolSizeCount = 2,
		        .pPoolSizes = rs_pool_sizes,
		};
		raster_sensor_pool = vk::raii::DescriptorPool(*device, rs_pool_info);
		vk::DescriptorSetAllocateInfo rs_alloc{
		        .descriptorPool = *raster_sensor_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*raster_sensor_layout,
		};
		auto rs_sets = vk::raii::DescriptorSets(*device, rs_alloc);
		raster_sensor_set = std::move(rs_sets.front());
	}
	{
		auto rs_frag = load_shader(*device, "raster_sensor_viz.frag");
		vk::PipelineShaderStageCreateInfo rs_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **rs_frag,
		                .pName = "main",
		        },
		};
		vk::PipelineLayoutCreateInfo rs_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*raster_sensor_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &push_range,
		};
		// Dedicated layout (a pipeline bakes its set layout, so the
		// two-image layout needs its own).
		raster_sensor_pipeline_layout = vk::raii::PipelineLayout(*device, rs_layout_info);
		vk::GraphicsPipelineCreateInfo rs_pipeline_info{
		        .stageCount = 2,
		        .pStages = rs_stages,
		        .pVertexInputState = &vertex_input,
		        .pInputAssemblyState = &input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pColorBlendState = &blend,
		        .pDynamicState = &dynamic,
		        .layout = *raster_sensor_pipeline_layout,
		        .renderPass = *renderpass,
		};
		raster_sensor_pipeline = vk::raii::Pipeline(*device, nullptr, rs_pipeline_info);
	}

	// VR-depth viz initial plain state (converted views rebuild the
	// immutable-sampler layout via ensure_virt_layout below).
	create_virt_state(wivrn::decoder::ycbcr_definition{}, vk::Sampler{});

	// VR+raster viz initial plain state (same rebuild discipline via
	// ensure_vr_raster_layout below).
	create_vr_raster_state(wivrn::decoder::ycbcr_definition{}, vk::Sampler{});

	// 6DOF sensor warp pipelines (no target dependency).
	init_warp();
}

void depth_overlay_renderer::record(vk::raii::CommandBuffer & cmd, size_t image_index,
                                    const std::array<vk::Extent2D, 2> & extents,
                                    vk::ImageView depth_view,
                                    const frame_params & params)
{
	if (image_index >= targets.size())
		return;

	std::memcpy(ubo.map(), &params, sizeof(params));
	ubo.unmap();
	// Non-coherent host-visible memory: the GPU reads stale values
	// forever without an explicit flush (vmaUnmapMemory does not flush).
	vmaFlushAllocation(vk_allocator::instance(), ubo, 0, sizeof(params));

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
	        .range = sizeof(frame_params),
	};
	std::array<vk::WriteDescriptorSet, 2> writes = {{
	        {
	                .dstSet = *descriptor_set,
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &image_info,
	        },
	        {
	                .dstSet = *descriptor_set,
	                .dstBinding = 1,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .pBufferInfo = &buffer_info,
	        },
	}};
	device->updateDescriptorSets(writes, nullptr);

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, *descriptor_set, {});

	// Fresh CLEAR images every frame: no barrier, no layout history.
	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = targets[image_index][eye];
		cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{(int32_t)eye});
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
		vk::RenderPassBeginInfo begin_info{
		        .renderPass = *renderpass,
		        .framebuffer = *tgt.framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = extents[eye]},
		        .clearValueCount = 1,
		        .pClearValues = &clear,
		};
		cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
		cmd.draw(3, 1, 0, 0);
		cmd.endRenderPass();
	}
}

namespace
{
// std140 mirror of the warp vertex UBO (single eye; rewritten per eye
// draw like the verdict SSBO discipline — an in-flight async record
// never shares this renderer).
struct warp_ubo_data
{
	float disp_vp[16];
	float sens_mount[16];
	float sens_tan[4];
	float sens_nf[4]; // near, far, 0, 0
};

struct warp_push
{
	int32_t eye;
};
} // namespace

void depth_overlay_renderer::init_warp()
{
	auto wvert = load_shader(*device, "sensor_reproject.vert");
	auto wfrag = load_shader(*device, "sensor_reproject.frag");
	vk::PipelineShaderStageCreateInfo stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **wvert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **wfrag,
	                .pName = "main",
	        },
	};

	// One vec4 per corner: xy = corner UV (texture space, for the
	// interpolated re-tap varying), zw = owning texel id (exact small
	// integers: the vertex fetch below is deterministic texelFetch,
	// never a rounding-defined NEAREST tap on a texel boundary).
	// Single vec4 attribute: two vec2 attributes would leave the
	// shader's zw defaulted to (0,1), collapsing every vertex to one
	// texel (total holes + misplaced survivors).
	vk::VertexInputBindingDescription vertex_bind{
	        .binding = 0,
	        .stride = sizeof(float) * 4,
	        .inputRate = vk::VertexInputRate::eVertex,
	};
	vk::VertexInputAttributeDescription vertex_attr{
	        .location = 0,
	        .binding = 0,
	        .format = vk::Format::eR32G32B32A32Sfloat,
	        .offset = 0,
	};
	vk::PipelineVertexInputStateCreateInfo vertex_input{
	        .vertexBindingDescriptionCount = 1,
	        .pVertexBindingDescriptions = &vertex_bind,
	        .vertexAttributeDescriptionCount = 1,
	        .pVertexAttributeDescriptions = &vertex_attr,
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
	        .lineWidth = 1,
	};
	vk::PipelineMultisampleStateCreateInfo multisample{
	        .rasterizationSamples = vk::SampleCountFlagBits::e1,
	};
	// Nearest-wins fixed-function depth: overlapping splats resolve to
	// the foreground surface (explicit finite-far projection below, so
	// plain LESS with a 1.0 clear; no reversed-Z subtlety in this pass).
	vk::PipelineDepthStencilStateCreateInfo depth_stencil{
	        .depthTestEnable = VK_TRUE,
	        .depthWriteEnable = VK_TRUE,
	        .depthCompareOp = vk::CompareOp::eLess,
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

	// Own set layout: sensor sampler (vertex fetch + fragment re-tap),
	// warp UBO (vertex only), frame-params UBO (fragment ramp).
	// Clear the framebuffers first: they reference the color views that
	// ensure() is about to destroy (this runs inside ensure's waitIdle,
	// before targets.clear()).
	warp_targets.clear();
	for (auto & s: warp_sets)
		s.clear();
	warp_pipeline.clear();
	warp_uv_pipeline.clear();
	warp_pipeline_layout.clear();
	vk::DescriptorSetLayoutBinding bindings[3] = {
	        {
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
	        },
	        {
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eVertex,
	        },
	        {
	                .binding = 2,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	warp_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                   .bindingCount = 3,
	                                                                   .pBindings = bindings,
	                                                           });
	vk::DescriptorPoolSize pool_sizes[2] = {
	        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 2},
	        {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 4},
	};
	warp_pool = vk::raii::DescriptorPool(*device, vk::DescriptorPoolCreateInfo{
	                                                              .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	                                                              .maxSets = 2,
	                                                              .poolSizeCount = 2,
	                                                              .pPoolSizes = pool_sizes,
	                                                      });
	vk::DescriptorSetAllocateInfo alloc_info{
	        .descriptorPool = *warp_pool,
	        .descriptorSetCount = 1,
	        .pSetLayouts = &*warp_layout,
	};
	warp_sets.clear();
	for (uint32_t eye = 0; eye < 2; ++eye)
		warp_sets.emplace_back(std::move(device->allocateDescriptorSets(alloc_info)[0]));

	// Two UBO slots (one per eye): both draws in one command buffer must
	// read their own matrices (host writes complete before submit, so a
	// rewritten single slot would show eye 1 to both draws).
	if (not warp_ubo or warp_ubo.info().size < 2 * sizeof(warp_ubo_data))
	{
		warp_ubo = buffer_allocation(
		        *device,
		        vk::BufferCreateInfo{
		                .size = 2 * sizeof(warp_ubo_data),
		                .usage = vk::BufferUsageFlagBits::eUniformBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "sensor warp");
	}

	// Eye push visible to both stages (vertex selects the sensor layer
	// for the texel fetch, fragment for the validation re-tap).
	vk::PushConstantRange push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = sizeof(int32_t),
	};
	warp_pipeline_layout = vk::raii::PipelineLayout(*device, vk::PipelineLayoutCreateInfo{
	                                                                           .setLayoutCount = 1,
	                                                                           .pSetLayouts = &*warp_layout,
	                                                                           .pushConstantRangeCount = 1,
	                                                                           .pPushConstantRanges = &push_range,
	                                                                   });

	// Own renderpass: color (same CLEAR/STORE/GENERAL discipline as the
	// direct tap) + depth (CLEAR 1.0, depth discarded after).
	vk::AttachmentDescription attachments[2] = {
	        {
	                .format = format_,
	                .samples = vk::SampleCountFlagBits::e1,
	                .loadOp = vk::AttachmentLoadOp::eClear,
	                .storeOp = vk::AttachmentStoreOp::eStore,
	                .initialLayout = vk::ImageLayout::eUndefined,
	                .finalLayout = vk::ImageLayout::eGeneral,
	        },
	        {
	                .format = vk::Format::eD16Unorm,
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
	vk::SubpassDescription subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &color_ref,
	        .pDepthStencilAttachment = &depth_ref,
	};
	warp_renderpass = vk::raii::RenderPass(*device, vk::RenderPassCreateInfo{
	                                                              .attachmentCount = 2,
	                                                              .pAttachments = attachments,
	                                                              .subpassCount = 1,
	                                                              .pSubpasses = &subpass,
	                                                      });

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
	        .layout = *warp_pipeline_layout,
	        .renderPass = *warp_renderpass,
	};
	warp_pipeline = vk::raii::Pipeline(*device, nullptr, pipeline_info);

	// UV-placement debug twin: identical vertex processing, fragment
	// outputs interpolated sensor UV as color (no depth decode, no
	// discard). Same layout, renderpass and push shape.
	{
		auto wuv_frag = load_shader(*device, "sensor_reproject_uv.frag");
		vk::PipelineShaderStageCreateInfo uv_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **wvert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **wuv_frag,
		                .pName = "main",
		        },
		};
		vk::GraphicsPipelineCreateInfo uv_pipeline_info{
		        .stageCount = 2,
		        .pStages = uv_stages,
		        .pVertexInputState = &vertex_input,
		        .pInputAssemblyState = &input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pDepthStencilState = &depth_stencil,
		        .pColorBlendState = &blend,
		        .pDynamicState = &dynamic,
		        .layout = *warp_pipeline_layout,
		        .renderPass = *warp_renderpass,
		};
		warp_uv_pipeline = vk::raii::Pipeline(*device, nullptr, uv_pipeline_info);
	}
}

void depth_overlay_renderer::ensure_warp_targets()
{
	// Framebuffer-sized depth (only the half-extent quadrant renders,
	// but every attachment must match the framebuffer size). One shared
	// depth image: passes run sequentially, CLEAR each time.
	warp_targets.clear();
	warp_depth_view.clear();
	warp_depth_image = image_allocation{
	        *device,
	        vk::ImageCreateInfo{
	                .imageType = vk::ImageType::e2D,
	                .format = vk::Format::eD16Unorm,
	                .extent = {extent_.width, extent_.height, 1},
	                .mipLevels = 1,
	                .arrayLayers = 1,
	                .samples = vk::SampleCountFlagBits::e1,
	                .tiling = vk::ImageTiling::eOptimal,
	                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
	                .sharingMode = vk::SharingMode::eExclusive,
	                .initialLayout = vk::ImageLayout::eUndefined,
	        },
	        VmaAllocationCreateInfo{
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "sensor warp depth",
	};
	warp_depth_view = vk::raii::ImageView(
	        *device,
	        vk::ImageViewCreateInfo{
	                .image = warp_depth_image,
	                .viewType = vk::ImageViewType::e2D,
	                .format = vk::Format::eD16Unorm,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
	                        .baseMipLevel = 0,
	                        .levelCount = 1,
	                        .baseArrayLayer = 0,
	                        .layerCount = 1,
	                },
	        });
	for (size_t i = 0; i < targets.size(); ++i)
	{
		warp_targets.emplace_back();
		for (uint32_t eye = 0; eye < 2; ++eye)
		{
			vk::ImageView attachments[2] = {*targets[i][eye].view, *warp_depth_view};
			vk::FramebufferCreateInfo fb_info{
			        .renderPass = *warp_renderpass,
			        .attachmentCount = 2,
			        .pAttachments = attachments,
			        .width = extent_.width,
			        .height = extent_.height,
			        .layers = 1,
			};
			warp_targets[i][eye].framebuffer = vk::raii::Framebuffer(*device, fb_info);
		}
	}
}

void depth_overlay_renderer::ensure_warp_mesh(uint32_t sens_w, uint32_t sens_h)
{
	if (sens_w == warp_mesh_w and sens_h == warp_mesh_h and warp_index_count != 0)
		return;
	if (sens_w == 0 or sens_h == 0)
		return;
	// Rare path (sensor resolution change): nothing outstanding (acquire
	// + release pair within one frame, always paired below).
	device->waitIdle();
	// Native-texel grid: one vertex per sensor texel (true measured
	// samples, never resampled blends); rasterization into the
	// half-res target gives sub-texel geometric precision while the
	// fragment re-tap keeps every value truthful.
	const size_t cols = (size_t)sens_w + 1, rows = (size_t)sens_h + 1;
	std::vector<float> verts;
	verts.reserve(cols * rows * 4);
	for (size_t j = 0; j < rows; ++j)
	{
		for (size_t i = 0; i < cols; ++i)
		{
			// Corner UV for coverage; owning texel clamped inside so
			// the vertex fetch below is an exact texelFetch (a corner
			// sits on a 4-way boundary and belongs to no texel).
			// Row mirror: texture v=1 is the frustum top (pinned by
			// the direct tap's 1-v flip rendering upright), i.e.
			// texelFetch row h-1 — while mesh row j=0 is the top.
			size_t ti = i < sens_w ? i : sens_w - 1;
			size_t tj = j < sens_h ? j : sens_h - 1;
			verts.push_back((float)i / (float)sens_w);
			verts.push_back(1.0f - (float)j / (float)sens_h);
			verts.push_back((float)ti);
			verts.push_back((float)(sens_h - 1 - tj));
		}
	}
	std::vector<uint32_t> indices;
	indices.reserve((size_t)sens_w * sens_h * 6);
	for (uint32_t j = 0; j < sens_h; ++j)
	{
		for (uint32_t i = 0; i < sens_w; ++i)
		{
			uint32_t a = j * (sens_w + 1) + i;
			uint32_t b = a + 1;
			uint32_t c = a + sens_w + 1;
			uint32_t d = c + 1;
			indices.insert(indices.end(), {a, c, b, b, c, d});
		}
	}
	warp_vertex_buffer = buffer_allocation(
	        *device,
	        vk::BufferCreateInfo{
	                .size = verts.size() * sizeof(float),
	                .usage = vk::BufferUsageFlagBits::eVertexBuffer,
	        },
	        VmaAllocationCreateInfo{
	                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "sensor warp grid");
	std::memcpy(warp_vertex_buffer.map(), verts.data(), verts.size() * sizeof(float));
	warp_vertex_buffer.unmap();
	vmaFlushAllocation(vk_allocator::instance(), warp_vertex_buffer, 0, verts.size() * sizeof(float));
	warp_index_buffer = buffer_allocation(
	        *device,
	        vk::BufferCreateInfo{
	                .size = indices.size() * sizeof(uint32_t),
	                .usage = vk::BufferUsageFlagBits::eIndexBuffer,
	        },
	        VmaAllocationCreateInfo{
	                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "sensor warp grid indices");
	std::memcpy(warp_index_buffer.map(), indices.data(), indices.size() * sizeof(uint32_t));
	warp_index_buffer.unmap();
	vmaFlushAllocation(vk_allocator::instance(), warp_index_buffer, 0, indices.size() * sizeof(uint32_t));
	warp_mesh_w = sens_w;
	warp_mesh_h = sens_h;
	warp_index_count = (uint32_t)indices.size();
	spdlog::info("Sensor warp grid: {}x{} texels ({} indices)", sens_w, sens_h, indices.size());
}

void depth_overlay_renderer::record_sensor_reproject(vk::raii::CommandBuffer & cmd, size_t image_index,
                                                     const std::array<vk::Extent2D, 2> & half_extents,
                                                     vk::ImageView sensor_view,
                                                     const reproject_params & warp,
                                                     const frame_params & params)
{
	if (image_index >= targets.size() or image_index >= warp_targets.size())
		return;
	ensure_warp_mesh(warp.sens_w, warp.sens_h);
	if (warp_index_count == 0)
		return;

	upload_viz_params(params);

	// Fill both UBO slots and write both sets up front: no set is ever
	// updated after bind, and each draw reads its own slot.
	vk::DescriptorImageInfo image_info{
	        .sampler = *sampler,
	        .imageView = sensor_view,
	        // The runtime hands the acquired image ready to sample
	        // (acquire is the sync point); keep whatever layout it is in.
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	vk::DescriptorBufferInfo params_buffer_info{
	        .buffer = vk::Buffer(ubo),
	        .offset = 0,
	        .range = sizeof(frame_params),
	};
	{
		uint8_t * slots = reinterpret_cast<uint8_t *>(warp_ubo.map());
		for (uint32_t eye = 0; eye < 2; ++eye)
		{
			warp_ubo_data data{};
			std::memcpy(data.disp_vp, warp.disp_vp[eye].data(), sizeof(data.disp_vp));
			std::memcpy(data.sens_mount, warp.sens_mount[eye].data(), sizeof(data.sens_mount));
			std::memcpy(data.sens_tan, warp.sens_tan[eye].data(), sizeof(data.sens_tan));
			data.sens_nf[0] = warp.sens_nf[eye][0];
			data.sens_nf[1] = warp.sens_nf[eye][1];
			std::memcpy(slots + eye * sizeof(data), &data, sizeof(data));

			vk::DescriptorBufferInfo warp_buffer_info{
			        .buffer = vk::Buffer(warp_ubo),
			        .offset = eye * sizeof(warp_ubo_data),
			        .range = sizeof(warp_ubo_data),
			};
			std::array<vk::WriteDescriptorSet, 3> writes = {{
			        {
			                .dstSet = *warp_sets[eye],
			                .dstBinding = 0,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			                .pImageInfo = &image_info,
			        },
			        {
			                .dstSet = *warp_sets[eye],
			                .dstBinding = 1,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eUniformBuffer,
			                .pBufferInfo = &warp_buffer_info,
			        },
			        {
			                .dstSet = *warp_sets[eye],
			                .dstBinding = 2,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eUniformBuffer,
			                .pBufferInfo = &params_buffer_info,
			        },
			}};
			device->updateDescriptorSets(writes, nullptr);
		}
		warp_ubo.unmap();
		vmaFlushAllocation(vk_allocator::instance(), warp_ubo, 0, 2 * sizeof(warp_ubo_data));
	}

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, warp.uv_debug ? *warp_uv_pipeline : *warp_pipeline);
	cmd.bindVertexBuffers(0, (vk::Buffer)warp_vertex_buffer, (vk::DeviceSize)0);
	cmd.bindIndexBuffer((vk::Buffer)warp_index_buffer, 0, vk::IndexType::eUint32);

	// Missing/streak-discarded pixels stay cleared: clear to the ramp's
	// far end (red) with the global opacity = "far", not "no data".
	vk::ClearValue clear_color{};
	clear_color.color.float32[0] = 0.9f;
	clear_color.color.float32[1] = 0.15f;
	clear_color.color.float32[2] = 0.1f;
	clear_color.color.float32[3] = params.opacity;
	vk::ClearValue clear_depth{};
	clear_depth.depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *warp_pipeline_layout, 0, *warp_sets[eye], {});
		cmd.pushConstants<warp_push>(*warp_pipeline_layout,
		                             vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		                             0,
		                             warp_push{(int32_t)eye});
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)half_extents[eye].width,
		                        .height = (float)half_extents[eye].height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = half_extents[eye]});
		std::array<vk::ClearValue, 2> clears = {clear_color, clear_depth};
		vk::RenderPassBeginInfo begin_info{
		        .renderPass = *warp_renderpass,
		        .framebuffer = *warp_targets[image_index][eye].framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = half_extents[eye]},
		        .clearValueCount = (uint32_t)clears.size(),
		        .pClearValues = clears.data(),
		};
		cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
		cmd.drawIndexed(warp_index_count, 1, 0, 0, 0);
		cmd.endRenderPass();
	}
}

void depth_overlay_renderer::create_virt_state(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler)
{
	// Full teardown in dependency order (verdict pattern): sets
	// reference pool + set layout, pipeline references pipeline layout,
	// pipeline layout references the set layout. Caller guarantees no
	// in-flight readers (fresh object, or a waitIdle'd rebuild).
	virt_set.clear();
	virt_pipeline.clear();
	virt_pipeline_layout.clear();
	virt_layout.clear();
	// Bake the handle's generation sampler as immutable (a separately
	// built conversion proved unidentifiable to validation despite
	// matching parameters, so the layout shares the handle's object
	// instead of owning one).
	virt_immutable_src = def.has_conversion ? generation_sampler : vk::Sampler{};
	vk::Sampler immutable = virt_immutable_src;
	vk::DescriptorSetLayoutBinding bindings[2] = {
	        {
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	                .pImmutableSamplers = def.has_conversion ? &immutable : nullptr,
	        },
	        {
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	virt_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                   .bindingCount = 2,
	                                                                   .pBindings = bindings,
	                                                           });
	vk::DescriptorPoolSize pool_sizes[2] = {
	        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1},
	        {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1},
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	        .maxSets = 1,
	        .poolSizeCount = 2,
	        .pPoolSizes = pool_sizes,
	};
	virt_pool = vk::raii::DescriptorPool(*device, pool_info);
	vk::DescriptorSetAllocateInfo set_alloc{
	        .descriptorPool = *virt_pool,
	        .descriptorSetCount = 1,
	        .pSetLayouts = &*virt_layout,
	};
	auto sets = vk::raii::DescriptorSets(*device, set_alloc);
	virt_set = std::move(sets.front());
	// Pipeline bakes the set layout: rebuilt alongside it, never apart.
	vk::PushConstantRange virt_push{
	        .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = sizeof(int32_t),
	};
	vk::PipelineLayoutCreateInfo virt_layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*virt_layout,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &virt_push,
	};
	virt_pipeline_layout = vk::raii::PipelineLayout(*device, virt_layout_info);
	auto vert = load_shader(*device, "depth_viz.vert");
	auto frag = load_shader(*device, "vr_depth_viz.frag");
	vk::PipelineShaderStageCreateInfo virt_stages[2] = {
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
	        .pStages = virt_stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *virt_pipeline_layout,
	        .renderPass = *renderpass,
	};
	virt_pipeline = vk::raii::Pipeline(*device, nullptr, pipeline_info);
	virt_ycbcr_built = def;
}

void depth_overlay_renderer::ensure_virt_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler)
{
	// Rebuild when the definition changes OR the handle sampler is a
	// new generation (the baked immutable handle must track the sampled
	// generation). Plain layouts don't bake a sampler, so only
	// converted state compares handles.
	if (virt_ycbcr_built == def and (not def.has_conversion or virt_immutable_src == generation_sampler))
		return;
	// Depth format change (rare: renegotiation only): retire prior
	// submissions before rebuilding what they may still read.
	device->waitIdle();
	create_virt_state(def, generation_sampler);
	spdlog::info("Depth overlay virt layout rebuilt (YCbCr {}, handle-generation immutable sampler)", def.has_conversion ? "immutable" : "plain");
}

void depth_overlay_renderer::create_vr_raster_state(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler)
{
	// Full teardown in dependency order (verdict pattern): sets
	// reference pool + set layout, pipeline references pipeline layout,
	// pipeline layout references the set layout. Caller guarantees no
	// in-flight readers (fresh object, or a waitIdle'd rebuild).
	vr_raster_set.clear();
	vr_raster_pipeline.clear();
	vr_raster_pipeline_layout.clear();
	vr_raster_layout.clear();
	// Bake the handle's generation sampler as the immutable sampler
	// for the SBS binding (a separately built conversion proved
	// unidentifiable to validation despite matching parameters, so the
	// layout shares the handle's object instead of owning one). The layer binding
	// stays mutable (plain R16F array view).
	vr_raster_immutable_src = def.has_conversion ? generation_sampler : vk::Sampler{};
	vk::Sampler immutable = vr_raster_immutable_src;
	vk::DescriptorSetLayoutBinding bindings[3] = {
	        {
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	                .pImmutableSamplers = def.has_conversion ? &immutable : nullptr,
	        },
	        {
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	        {
	                .binding = 2,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	vr_raster_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                          .bindingCount = 3,
	                                                                          .pBindings = bindings,
	                                                                  });
	vk::DescriptorPoolSize pool_sizes[2] = {
	        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 2},
	        {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1},
	};
	vk::DescriptorPoolCreateInfo pool_info{
	        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	        .maxSets = 1,
	        .poolSizeCount = 2,
	        .pPoolSizes = pool_sizes,
	};
	vr_raster_pool = vk::raii::DescriptorPool(*device, pool_info);
	vk::DescriptorSetAllocateInfo set_alloc{
	        .descriptorPool = *vr_raster_pool,
	        .descriptorSetCount = 1,
	        .pSetLayouts = &*vr_raster_layout,
	};
	auto sets = vk::raii::DescriptorSets(*device, set_alloc);
	vr_raster_set = std::move(sets.front());
	// Pipeline bakes the set layout: rebuilt alongside it, never apart.
	vk::PushConstantRange vr_push{
	        .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        .offset = 0,
	        .size = sizeof(int32_t),
	};
	vk::PipelineLayoutCreateInfo vr_layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*vr_raster_layout,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &vr_push,
	};
	vr_raster_pipeline_layout = vk::raii::PipelineLayout(*device, vr_layout_info);
	auto vert = load_shader(*device, "depth_viz.vert");
	auto frag = load_shader(*device, "vr_raster_viz.frag");
	vk::PipelineShaderStageCreateInfo vr_stages[2] = {
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
	        .pStages = vr_stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *vr_raster_pipeline_layout,
	        .renderPass = *renderpass,
	};
	vr_raster_pipeline = vk::raii::Pipeline(*device, nullptr, pipeline_info);
	vr_raster_ycbcr_built = def;
}

void depth_overlay_renderer::ensure_vr_raster_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler)
{
	// Rebuild when the definition changes OR the handle sampler is a
	// new generation (the baked immutable handle must track the sampled
	// generation). Plain layouts don't bake a sampler, so only
	// converted state compares handles.
	if (vr_raster_ycbcr_built == def and (not def.has_conversion or vr_raster_immutable_src == generation_sampler))
		return;
	// Depth format change (rare: renegotiation only): retire prior
	// submissions before rebuilding what they may still read.
	device->waitIdle();
	create_vr_raster_state(def, generation_sampler);
	spdlog::info("Depth overlay VR+raster layout rebuilt (YCbCr {}, handle-generation immutable sampler)", def.has_conversion ? "immutable" : "plain");
}

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
} // namespace

void depth_overlay_renderer::upload_viz_params(const frame_params & params)
{
	std::memcpy(ubo.map(), &params, sizeof(params));
	ubo.unmap();
	// Non-coherent host-visible memory: the GPU reads stale values
	// forever without an explicit flush (vmaUnmapMemory does not flush).
	vmaFlushAllocation(vk_allocator::instance(), ubo, 0, sizeof(params));
}

void depth_overlay_renderer::record_layer(vk::raii::CommandBuffer & cmd, size_t image_index,
                                          const std::array<vk::Extent2D, 2> & extents,
                                          vk::Image layer_image, vk::ImageView layer_view,
                                          const frame_params & params)
{
	if (image_index >= targets.size())
		return;

	upload_viz_params(params);
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
		vk::DescriptorBufferInfo buffer_info{
		        .buffer = vk::Buffer(ubo),
		        .offset = 0,
		        .range = sizeof(frame_params),
		};
		std::array<vk::WriteDescriptorSet, 2> writes = {{
		        {
		                .dstSet = *descriptor_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &image_info,
		        },
		        {
		                .dstSet = *descriptor_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eUniformBuffer,
		                .pBufferInfo = &buffer_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *layer_pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, *descriptor_set, {});
	}

	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = targets[image_index][eye];
		cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{(int32_t)eye});
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

void depth_overlay_renderer::record_verdict(vk::raii::CommandBuffer & cmd, size_t image_index,
                                            const std::array<vk::Extent2D, 2> & extents,
                                            vk::Image verdict_image, vk::ImageView verdict_view,
                                            vk::Image layer_image, vk::ImageView layer_view,
                                            const frame_params & params)
{
	if (image_index >= targets.size())
		return;

	upload_viz_params(params);
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
		vk::DescriptorBufferInfo buffer_info{
		        .buffer = vk::Buffer(ubo),
		        .offset = 0,
		        .range = sizeof(frame_params),
		};
		std::array<vk::WriteDescriptorSet, 3> writes = {{
		        {
		                .dstSet = *verdict_viz_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &verdict_info,
		        },
		        {
		                .dstSet = *verdict_viz_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &layer_info,
		        },
		        {
		                .dstSet = *verdict_viz_set,
		                .dstBinding = 2,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eUniformBuffer,
		                .pBufferInfo = &buffer_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *verdict_pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *verdict_viz_pipeline_layout, 0, *verdict_viz_set, {});
	}

	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = targets[image_index][eye];
		cmd.pushConstants<push>(*verdict_viz_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{(int32_t)eye});
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

void depth_overlay_renderer::record_raster_sensor(vk::raii::CommandBuffer & cmd, size_t image_index,
                                                                                  const std::array<vk::Extent2D, 2> & extents,
                                                                                  vk::Image layer_image, vk::ImageView layer_view,
                                                                                  vk::ImageView sensor_view,
                                                                                  const frame_params & params)
{
	if (image_index >= targets.size())
		return;

	upload_viz_params(params);
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
		vk::DescriptorBufferInfo buffer_info{
		        .buffer = vk::Buffer(ubo),
		        .offset = 0,
		        .range = sizeof(frame_params),
		};
		std::array<vk::WriteDescriptorSet, 3> writes = {{
		        {
		                .dstSet = *raster_sensor_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &layer_info,
		        },
		        {
		                .dstSet = *raster_sensor_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &sensor_info,
		        },
		        {
		                .dstSet = *raster_sensor_set,
		                .dstBinding = 2,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eUniformBuffer,
		                .pBufferInfo = &buffer_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *raster_sensor_pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *raster_sensor_pipeline_layout, 0, *raster_sensor_set, {});
	}

	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = targets[image_index][eye];
		cmd.pushConstants<push>(*raster_sensor_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{(int32_t)eye});
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

void depth_overlay_renderer::record_virt(vk::raii::CommandBuffer & cmd, size_t image_index,
                                         const std::array<vk::Extent2D, 2> & extents,
                                         vk::ImageView virt_view, vk::Sampler virt_sampler,
                                         const wivrn::decoder::ycbcr_definition & ycbcr,
                                         const frame_params & params)
{
	if (image_index >= targets.size())
		return;

	upload_viz_params(params);

	bool have_data = (bool)virt_view and (bool)virt_sampler;
	if (have_data)
	{
		// YCbCr immutable-sampler layout (VUID-02738): rebuild when the
		// view's conversion definition or the handle's generation
		// sampler changes. waitIdle'd, rare.
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
		vk::DescriptorBufferInfo buffer_info{
		        .buffer = vk::Buffer(ubo),
		        .offset = 0,
		        .range = sizeof(frame_params),
		};
		std::array<vk::WriteDescriptorSet, 2> writes = {{
		        {
		                .dstSet = *virt_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &image_info,
		        },
		        {
		                .dstSet = *virt_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eUniformBuffer,
		                .pBufferInfo = &buffer_info,
		        },
		}};
		device->updateDescriptorSets(writes, nullptr);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *virt_pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *virt_pipeline_layout, 0, *virt_set, {});
	}

	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = targets[image_index][eye];
		cmd.pushConstants<push>(*virt_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{(int32_t)eye});
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

void depth_overlay_renderer::record_vr_raster(vk::raii::CommandBuffer & cmd, size_t image_index,
                                              const std::array<vk::Extent2D, 2> & extents,
                                              vk::ImageView virt_view, vk::Sampler virt_sampler,
                                              const wivrn::decoder::ycbcr_definition & ycbcr,
                                              vk::Image layer_image, vk::ImageView layer_view,
                                              const frame_params & params)
{
	if (image_index >= targets.size())
		return;

	upload_viz_params(params);
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
	// records opaque black. A null layer simply shows plain VR depth:
	// the shader reads no-window as absent and keeps the VR color.
	bool have_data = (bool)virt_view and (bool)virt_sampler;
	bool have_layer = (bool)layer_view and layer_image != vk::Image{};
	if (have_data)
	{
		// YCbCr immutable-sampler layout (VUID-02738): rebuild when the
		// view's conversion definition or the handle's generation
		// sampler changes. waitIdle'd, rare.
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
		vk::DescriptorBufferInfo buffer_info{
		        .buffer = vk::Buffer(ubo),
		        .offset = 0,
		        .range = sizeof(frame_params),
		};
		// Unbound image bindings with a null view are invalid, so
		// without a layer the dedicated VR-depth path records instead
		// (same image, no overlay).
		if (have_layer)
		{
			vk::DescriptorImageInfo valid_layer_info{
			        .sampler = *sampler,
			        .imageView = layer_view,
			        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			};
			std::array<vk::WriteDescriptorSet, 3> writes = {{
			        {
			                .dstSet = *vr_raster_set,
			                .dstBinding = 0,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			                .pImageInfo = &virt_info,
			        },
			        {
			                .dstSet = *vr_raster_set,
			                .dstBinding = 1,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			                .pImageInfo = &valid_layer_info,
			        },
			        {
			                .dstSet = *vr_raster_set,
			                .dstBinding = 2,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eUniformBuffer,
			                .pBufferInfo = &buffer_info,
			        },
			}};
			device->updateDescriptorSets(writes, nullptr);
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *vr_raster_pipeline);
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *vr_raster_pipeline_layout, 0, *vr_raster_set, {});
		}
		else
		{
			// No layer to composite: plain VR depth through the
			// dedicated path (same image, no overlay).
			record_virt(cmd, image_index, extents, virt_view, virt_sampler, ycbcr, params);
			return;
		}
	}

	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = targets[image_index][eye];
		cmd.pushConstants<push>(*vr_raster_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{(int32_t)eye});
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
