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
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace
{
// Mirrors hidden_dots.vert/frag.glsl PushConstants: mvp (64) + color and
// pitch (16) = 80B, within the 128-byte minimum.
struct push
{
	glm::mat4 mvp;
	glm::vec4 color_pitch; // rgb: dot color, w: dot pitch (m)
};
static_assert(sizeof(push) == 80);
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

void hidden_dots_renderer::set_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup)
{
	if (soup.vertices.empty() or soup.indices.empty())
	{
		remove_mesh(key);
		return;
	}
	auto & mesh = meshes[key];
	vk::DeviceSize vsize = soup.vertices.size() * sizeof(XrVector3f);
	vk::DeviceSize isize = soup.indices.size() * sizeof(uint32_t);
	mesh.vertex_buffer = buffer_allocation(
	        *device,
	        vk::BufferCreateInfo{
	                .size = vsize,
	                .usage = vk::BufferUsageFlagBits::eVertexBuffer,
	        },
	        VmaAllocationCreateInfo{
	                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "hidden dots verts");
	mesh.index_buffer = buffer_allocation(
	        *device,
	        vk::BufferCreateInfo{
	                .size = isize,
	                .usage = vk::BufferUsageFlagBits::eIndexBuffer,
	        },
	        VmaAllocationCreateInfo{
	                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "hidden dots indices");
	std::memcpy(mesh.vertex_buffer.map(), soup.vertices.data(), vsize);
	mesh.vertex_buffer.unmap();
	std::memcpy(mesh.index_buffer.map(), soup.indices.data(), isize);
	mesh.index_buffer.unmap();
	// Non-coherent host-visible memory: flush once, meshes are static.
	vmaFlushAllocation(vk_allocator::instance(), mesh.vertex_buffer, 0, vsize);
	vmaFlushAllocation(vk_allocator::instance(), mesh.index_buffer, 0, isize);
	mesh.index_count = (uint32_t)soup.indices.size();
	// Model-space bounds for the stream-side frustum skip (upload-time
	// only; per-frame tests transform the 8 corners by the draw MVP).
	// Seeded from the first vertex: infinity is UB under -ffast-math.
	glm::vec3 mn(soup.vertices[0].x, soup.vertices[0].y, soup.vertices[0].z);
	glm::vec3 mx = mn;
	for (const auto & v: soup.vertices)
	{
		mn = glm::min(mn, glm::vec3(v.x, v.y, v.z));
		mx = glm::max(mx, glm::vec3(v.x, v.y, v.z));
	}
	mesh.aabb_min = mn;
	mesh.aabb_max = mx;
}

void hidden_dots_renderer::remove_mesh(const std::string & key)
{
	meshes.erase(key);
}

std::optional<std::pair<glm::vec3, glm::vec3>> hidden_dots_renderer::mesh_bounds(const std::string & key) const
{
	if (auto it = meshes.find(key); it != meshes.end())
		return std::make_pair(it->second.aabb_min, it->second.aabb_max);
	return std::nullopt;
}

void hidden_dots_renderer::init_pipelines(vk::Format format)
{
	pipe_cache = pipeline_cache(*device);
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
	        // Front faces only, no depth test: closed meshes show near-side
	        // dots only, always on top. Open meshes read from the front.
	        .cullMode = vk::CullModeFlagBits::eBack,
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
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)extents[eye].width,
		                        .height = (float)extents[eye].height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		// Tight scissor from the draws' projected bounds: shading tracks
		// object size instead of the fullscreen. The render-area CLEAR
		// stays full (correctness); only draw fragments are culled. Y-flip
		// safe: min/max over corners is symmetric under the flip.
		vk::Rect2D scissor{.offset = {0, 0}, .extent = extents[eye]};
		{
			float x0 = 0, y0 = 0, x1 = (float)extents[eye].width, y1 = (float)extents[eye].height;
			bool full = false, any = false;
			for (const auto & d: draws)
			{
				auto mit = meshes.find(d.mesh);
				if (mit == meshes.end())
				{
					full = true;
					break;
				}
				for (int corner = 0; corner < 8 and not full; ++corner)
				{
					glm::vec4 q4((corner & 1) ? mit->second.aabb_max.x : mit->second.aabb_min.x,
					             (corner & 2) ? mit->second.aabb_max.y : mit->second.aabb_min.y,
					             (corner & 4) ? mit->second.aabb_max.z : mit->second.aabb_min.z, 1);
					glm::vec4 c = d.mvp[eye] * q4;
					if (c.w <= 0)
					{
						full = true;
						break;
					}
					float px = (c.x / c.w * 0.5f + 0.5f) * (float)extents[eye].width;
					float py = (c.y / c.w * 0.5f + 0.5f) * (float)extents[eye].height;
					if (not any)
					{
						x0 = x1 = px;
						y0 = y1 = py;
						any = true;
					}
					else
					{
						x0 = std::min(x0, px);
						x1 = std::max(x1, px);
						y0 = std::min(y0, py);
						y1 = std::max(y1, py);
					}
				}
			}
			if (not full and any)
			{
				int32_t ix0 = (int32_t)std::max(0.f, x0 - 1.f);
				int32_t iy0 = (int32_t)std::max(0.f, y0 - 1.f);
				int32_t ix1 = (int32_t)std::min((float)extents[eye].width, x1 + 1.f);
				int32_t iy1 = (int32_t)std::min((float)extents[eye].height, y1 + 1.f);
				if (ix1 > ix0 and iy1 > iy0)
					scissor = vk::Rect2D{.offset = {ix0, iy0},
					                     .extent = {(uint32_t)(ix1 - ix0), (uint32_t)(iy1 - iy0)}};
			}
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
		for (const auto & d: draws)
		{
			auto mit = meshes.find(d.mesh);
			if (mit == meshes.end() or mit->second.index_count == 0)
				continue;
			cmd.bindVertexBuffers(0, vk::Buffer(mit->second.vertex_buffer), (vk::DeviceSize)0);
			cmd.bindIndexBuffer(vk::Buffer(mit->second.index_buffer), (vk::DeviceSize)0, vk::IndexType::eUint32);
			push pc{
			        .mvp = d.mvp[eye],
			        .color_pitch = glm::vec4(d.color, d.pitch),
			};
			cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
			cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
		}
		cmd.endRenderPass();
	}
	return true;
}
