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
#include <cstring>

#include <spdlog/spdlog.h>

namespace
{
// Mirrors hidden_dots.vert/frag.glsl PushConstants: mvp (64) + color and
// pitch (16) + view params (16) = 96B, within the 128-byte minimum.
struct push
{
	glm::mat4 mvp;
	glm::vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
	glm::vec4 view_params; // x: view W px, y: view H px, z: proj[1][1], w: unused
};
static_assert(sizeof(push) == 96);
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

void hidden_dots_renderer::ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h)
{
	// Half resolution: dots stay crisp (world-size pattern, still several
	// px per dot at room range) while clear/store/shading quarter. The
	// compositor upscales the submitted rect to the full FOV.
	uint32_t hw = ((uint32_t)w + 1) / 2;
	uint32_t hh = ((uint32_t)h + 1) / 2;
	if (ready() and format_ == format and extent_.width == hw and extent_.height == hh)
		return;
	// Rare path (first frame, video resize): nothing outstanding (we
	// acquire + release within one frame, always paired below).
	device->waitIdle();
	format_ = format;
	extent_ = {hw, hh};
	swapchain_ = xr::swapchain(inst, sess, *device, format, (int)hw, (int)hh, 1, 2);
	spdlog::info("Hidden dots swapchain: {}x{} ({} images)", hw, hh, swapchain_.image_count());
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
			vk::ImageView raw_view = *targets.back()[eye].view;
			vk::FramebufferCreateInfo fb_info{
			        .renderPass = *renderpass,
			        .attachmentCount = 1,
			        .pAttachments = &raw_view,
			        .width = extent_.width,
			        .height = extent_.height,
			        .layers = 1,
			};
			targets.back()[eye].framebuffer = vk::raii::Framebuffer(*device, fb_info);
		}
	}
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
	auto vert = load_shader(*device, "hidden_dots.vert");
	auto frag = load_shader(*device, "hidden_dots.frag");

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

	// No depth attachment, no test, no writes: dots draw over whatever the
	// layers below composed.
	vk::PipelineDepthStencilStateCreateInfo depth_stencil{
	        .depthTestEnable = VK_FALSE,
	        .depthWriteEnable = VK_FALSE,
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
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &push_range,
	};
	pipeline_layout = vk::raii::PipelineLayout(*device, layout_info);

	// Color CLEAR on UNDEFINED every frame (no layout history ever assumed,
	// gizmo precedent). Final GENERAL for the color (universally samplable
	// at submit, mask precedent).
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
	        .pDepthStencilState = &depth_stencil,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *pipeline_layout,
	        .renderPass = *renderpass,
	};
	pipeline = vk::raii::Pipeline(*device, *pipe_cache, pipeline_info);
}

bool hidden_dots_renderer::record(vk::raii::CommandBuffer & cmd, size_t image_index,
                                  const std::array<vk::Extent2D, 2> & extents,
                                  const std::vector<instance_draw> & draws)
{
	if (draws.empty() or image_index >= targets.size())
		return false;

	for (uint32_t eye = 0; eye < 2; ++eye)
	{
		auto & tgt = targets[image_index][eye];
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)extents[eye].width,
		                        .height = (float)extents[eye].height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		// Union of the draws' stream-side scissor rects (computed once per
		// object in the shared silhouette build): shading tracks object
		// size instead of the fullscreen. The render-area CLEAR stays full
		// (correctness); only draw fragments are culled. A zero union for
		// this eye still begins/ends the pass (clear-only): the image
		// starts UNDEFINED and the compositor samples the full rect.
		vk::Rect2D scissor{.offset = {0, 0}, .extent = extents[eye]};
		bool any = false;
		{
			int32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
			for (const auto & d: draws)
			{
				const auto & r = d.scissor[eye];
				int32_t rx0 = std::max(r.offset.x, 0);
				int32_t ry0 = std::max(r.offset.y, 0);
				int32_t rx1 = std::min(r.offset.x + (int32_t)r.extent.width, (int32_t)extents[eye].width);
				int32_t ry1 = std::min(r.offset.y + (int32_t)r.extent.height, (int32_t)extents[eye].height);
				if (rx1 <= rx0 or ry1 <= ry0)
					continue;
				if (not any)
				{
					x0 = rx0;
					y0 = ry0;
					x1 = rx1;
					y1 = ry1;
					any = true;
				}
				else
				{
					x0 = std::min(x0, rx0);
					y0 = std::min(y0, ry0);
					x1 = std::max(x1, rx1);
					y1 = std::max(y1, ry1);
				}
			}
			if (any)
				scissor = vk::Rect2D{.offset = {x0, y0},
				                     .extent = {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
		}
		cmd.setScissor(0, scissor);
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
		if (any)
		{
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
			cmd.bindVertexBuffers(0, (vk::Buffer)quad_, (vk::DeviceSize)0);
			for (const auto & d: draws)
			{
				if (d.dot_count == 0)
					continue;
				// Baked dot instances borrowed from the feather hidden
				// renderer (owns upload + eviction); bound fresh per draw.
				cmd.bindVertexBuffers(1, d.dot_buffer, (vk::DeviceSize)0);
				push pc{
				        .mvp = d.mvp[eye],
				        .color_pitch = glm::vec4(d.color, d.pitch),
				        .view_params = glm::vec4(d.view_params[eye], 0.f),
				};
				cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
				cmd.draw(6, d.dot_count, 0, 0);
			}
		}
		cmd.endRenderPass();
	}
	return true;
}
