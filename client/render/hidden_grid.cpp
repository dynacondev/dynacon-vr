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

#include "render/hidden_grid.h"

#include "vk/shader.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace
{
// Mirrors hidden_grid.vert/frag.glsl PushConstants. Stays within the
// 128-byte minimum push guarantee: mvp (64) + placement (32) + style (32).
struct push
{
	glm::mat4 mvp;
	glm::vec4 obj_pos_scale; // xyz: world pos (m), w: uniform scale
	glm::vec4 obj_quat; // xyzw
	glm::vec4 grid_col_scale; // rgb: grid color, w: spacing (m)
	glm::vec4 cam_pos_width; // xyz: eye world pos (m), w: line width (px)
};
static_assert(sizeof(push) == 128);

// Depth for the grid raster (D16 depth-test-only: never sampled,
// device-local; D16 attachment support is mandatory per spec).
constexpr vk::Format depth_format = vk::Format::eD16Unorm;

uint32_t find_memory_type(vk::raii::PhysicalDevice & physical_device, uint32_t type_bits, vk::MemoryPropertyFlags properties)
{
	auto mem = physical_device.getMemoryProperties();
	for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
	{
		if ((type_bits & (1u << i)) and (mem.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	throw std::runtime_error("hidden_grid: no suitable memory type");
}
} // namespace

hidden_grid_renderer::hidden_grid_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device) :
        device(&device),
        physical_device(&physical_device)
{
}

void hidden_grid_renderer::ensure(xr::instance & inst, xr::session & sess, vk::Format format, int w, int h)
{
	if (ready() and format_ == format and (int)extent_.width == w and (int)extent_.height == h)
		return;
	// Rare path (first frame, video resize): nothing outstanding (we
	// acquire + release within one frame, always paired below).
	device->waitIdle();
	format_ = format;
	extent_ = {(uint32_t)w, (uint32_t)h};
	swapchain_ = xr::swapchain(inst, sess, *device, format, w, h, 1, 2);
	spdlog::info("Hidden grid swapchain: {}x{} ({} images)", w, h, swapchain_.image_count());
	init_pipelines(format);
	targets.clear();
	// Shared depth pair (single frame records at a time).
	{
		vk::ImageCreateInfo img_info{
		        .imageType = vk::ImageType::e2D,
		        .format = depth_format,
		        .extent = {extent_.width, extent_.height, 1},
		        .mipLevels = 1,
		        .arrayLayers = 2,
		        .samples = vk::SampleCountFlagBits::e1,
		        .tiling = vk::ImageTiling::eOptimal,
		        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
		        .sharingMode = vk::SharingMode::eExclusive,
		        .initialLayout = vk::ImageLayout::eUndefined,
		};
		depth_image = vk::raii::Image(*device, img_info);
		auto mem_reqs = depth_image.getMemoryRequirements();
		uint32_t mem_idx = find_memory_type(
		        *physical_device, mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		depth_memory = vk::raii::DeviceMemory(
		        *device, vk::MemoryAllocateInfo{.allocationSize = mem_reqs.size, .memoryTypeIndex = mem_idx});
		depth_image.bindMemory(*depth_memory, 0);
		for (uint32_t eye = 0; eye < 2; ++eye)
		{
			vk::ImageViewCreateInfo dview_info{
			        .image = *depth_image,
			        .viewType = vk::ImageViewType::e2DArray,
			        .format = depth_format,
			        .subresourceRange = {
			                .aspectMask = vk::ImageAspectFlagBits::eDepth,
			                .baseMipLevel = 0,
			                .levelCount = 1,
			                .baseArrayLayer = eye,
			                .layerCount = 1,
			        },
			};
			depth_views[eye] = vk::raii::ImageView(*device, dview_info);
		}
	}
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
			vk::ImageView raw_views[2] = {*targets.back()[eye].view, *depth_views[eye]};
			vk::FramebufferCreateInfo fb_info{
			        .renderPass = *renderpass,
			        .attachmentCount = 2,
			        .pAttachments = raw_views,
			        .width = extent_.width,
			        .height = extent_.height,
			        .layers = 1,
			};
			targets.back()[eye].framebuffer = vk::raii::Framebuffer(*device, fb_info);
		}
	}
}

int hidden_grid_renderer::acquire()
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

void hidden_grid_renderer::release()
{
	swapchain_.release();
}

void hidden_grid_renderer::set_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup)
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
	        "hidden grid verts");
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
	        "hidden grid indices");
	std::memcpy(mesh.vertex_buffer.map(), soup.vertices.data(), vsize);
	mesh.vertex_buffer.unmap();
	std::memcpy(mesh.index_buffer.map(), soup.indices.data(), isize);
	mesh.index_buffer.unmap();
	// Non-coherent host-visible memory: flush once, meshes are static.
	vmaFlushAllocation(vk_allocator::instance(), mesh.vertex_buffer, 0, vsize);
	vmaFlushAllocation(vk_allocator::instance(), mesh.index_buffer, 0, isize);
	mesh.index_count = (uint32_t)soup.indices.size();
	glm::vec3 mn(std::numeric_limits<float>::infinity());
	glm::vec3 mx(-std::numeric_limits<float>::infinity());
	for (const auto & v: soup.vertices)
	{
		mn = glm::min(mn, glm::vec3(v.x, v.y, v.z));
		mx = glm::max(mx, glm::vec3(v.x, v.y, v.z));
	}
	mesh.aabb_min = mn;
	mesh.aabb_max = mx;
}

void hidden_grid_renderer::remove_mesh(const std::string & key)
{
	meshes.erase(key);
}

std::optional<std::pair<glm::vec3, glm::vec3>> hidden_grid_renderer::mesh_bounds(const std::string & key) const
{
	if (auto it = meshes.find(key); it != meshes.end())
		return std::make_pair(it->second.aabb_min, it->second.aabb_max);
	return std::nullopt;
}

void hidden_grid_renderer::init_pipelines(vk::Format format)
{
	auto vert = load_shader(*device, "hidden_grid.vert");
	auto frag = load_shader(*device, "hidden_grid.frag");

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
	        .cullMode = vk::CullModeFlagBits::eBack,
	        .frontFace = vk::FrontFace::eCounterClockwise,
	        .lineWidth = 1,
	};

	vk::PipelineMultisampleStateCreateInfo multisample{
	        .rasterizationSamples = vk::SampleCountFlagBits::e1,
	};

	// Self-occluding silhouette: GREATER, not LESS. scene::projection_matrix
	// is reversed-Z (near = 1, far = 0, matching scene_renderer and the
	// feather depth path); LESS here would invert the occlusion.
	vk::PipelineDepthStencilStateCreateInfo depth_stencil{
	        .depthTestEnable = VK_TRUE,
	        .depthWriteEnable = VK_TRUE,
	        .depthCompareOp = vk::CompareOp::eGreater,
	};

	// Straight alpha blend over transparent.
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
	// gizmo precedent), depth CLEAR (reversed-Z far = 0). Final GENERAL for
	// the color (universally samplable at submit, mask precedent).
	vk::AttachmentDescription attachments[2] = {
	        {
	                .format = format,
	                .samples = vk::SampleCountFlagBits::e1,
	                .loadOp = vk::AttachmentLoadOp::eClear,
	                .storeOp = vk::AttachmentStoreOp::eStore,
	                .initialLayout = vk::ImageLayout::eUndefined,
	                .finalLayout = vk::ImageLayout::eGeneral,
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
	vk::SubpassDescription subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &color_ref,
	        .pDepthStencilAttachment = &depth_ref,
	};
	vk::RenderPassCreateInfo renderpass_info{
	        .attachmentCount = 2,
	        .pAttachments = attachments,
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
	pipeline = vk::raii::Pipeline(*device, nullptr, pipeline_info);
}

void hidden_grid_renderer::record(vk::raii::CommandBuffer & cmd, size_t image_index,
                                   const std::array<vk::Extent2D, 2> & extents,
                                   const std::vector<instance_draw> & draws)
{
	if (draws.empty() or image_index >= targets.size())
		return;

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
		cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extents[eye]});
		std::array<vk::ClearValue, 2> clears{};
		clears[0].color.float32.fill(0);
		clears[1].depthStencil = vk::ClearDepthStencilValue{0.f, 0};
		vk::RenderPassBeginInfo begin_info{
		        .renderPass = *renderpass,
		        .framebuffer = *tgt.framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = extents[eye]},
		        .clearValueCount = (uint32_t)clears.size(),
		        .pClearValues = clears.data(),
		};
		cmd.beginRenderPass(begin_info, vk::SubpassContents::eInline);
		for (const auto & d: draws)
		{
			auto mit = meshes.find(d.mesh);
			if (mit == meshes.end() or mit->second.index_count == 0)
				continue;
			cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
			cmd.bindIndexBuffer(*mit->second.index_buffer, (vk::DeviceSize)0, vk::IndexType::eUint32);
			push pc{
			        .mvp = d.mvp[eye],
			        .obj_pos_scale = glm::vec4(d.obj_pos, d.obj_scale),
			        .obj_quat = glm::vec4(d.obj_quat.x, d.obj_quat.y, d.obj_quat.z, d.obj_quat.w),
			        .grid_col_scale = glm::vec4(d.grid_color, d.grid_scale),
			        .cam_pos_width = glm::vec4(d.cam_pos[eye], d.line_width_px),
			};
			cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
			cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
		}
		cmd.endRenderPass();
	}
}
