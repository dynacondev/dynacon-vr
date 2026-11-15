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
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
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

// RAII section timer for the record cpu_stats (steady_clock,
// milliseconds): adds the scoped wall time to *acc on destruction.
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

// Half res (W = half of full), floored like the feather tier
// intermediates. Tier-1-only is load-bearing (W == seed size).
vk::Extent2D half_extent(vk::Extent2D full)
{
	return {std::max(8u, full.width / 2), std::max(8u, full.height / 2)};
}

// R8 single-channel mask convention (same as the feather blur
// intermediates): identical 8-bit precision at a quarter the bytes of
// RGBA8. Declared here (not shared) to avoid a cross-TU extern.
constexpr vk::Format blur_format = vk::Format::eR8Unorm;
} // namespace

union_merge_renderer::union_merge_renderer(vk::raii::Device & device) :
        device(&device)
{
}

std::shared_ptr<vk::raii::PipelineCache> union_merge_renderer::pipeline_cache(vk::raii::Device & device)
{
	// Same process-wide object shared with the feather mask renderer
	// (one entry per VkDevice).
	return feather_mask_renderer::pipeline_cache(device);
}

void union_merge_renderer::warm_pipelines(vk::raii::Device & device, vk::Format format)
{
	// Throwaway renderer: its pipeline compile populates the shared
	// cache above, so the first real merge pays ~nothing.
	union_merge_renderer warm(device);
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
	// (destroys the use of) any sets still held. Single sets first
	// (same hazard: their pools die in init_pipelines).
	descriptor_sets.clear();
	volume_virt_set.clear();
	verdict_set.clear();
	apply_set.clear();
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
	// layers=2 framebuffer; one draw broadcasts to both eyes.
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
		        .layers = 2,
		};
		targets.back().framebuffer = vk::raii::Framebuffer(*device, fb_info);
	}
	// Commit only after the whole rebuild succeeded (see above).
	format_ = new_format;
	extent_ = new_extent;
	// Layer working set derives from the full extent (W = half of it).
	// record_base (earlier every frame) already ensured the borrowed
	// layer/verdict/virt at this size, so this is normally a merged-only
	// build; the gate inside handles both cases (a format change forces
	// the merged rebuild via merged_ready_ below).
	merged_ready_ = false;
	ensure_layer_targets(half_extent(new_extent));
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

void union_merge_renderer::set_layer_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup)
{
	if (soup.indices.empty() or soup.vertices.empty())
	{
		remove_layer_mesh(key);
		return;
	}

	// Replace path only (map change): make sure no in-flight frame
	// still reads the old buffers before replacing them. New keys have
	// no live buffers (in-flight draws miss the key and skip), so first
	// sight pays no waitIdle hitch mid-session. Rare either way;
	// render-thread only, no threading change.
	if (meshes.find(key) != meshes.end())
		device->waitIdle();

	layer_mesh & mesh = meshes[key];
	vk::DeviceSize vertex_bytes = soup.vertices.size() * sizeof(XrVector3f);
	vk::DeviceSize index_bytes = soup.indices.size() * sizeof(uint32_t);
	vk::DeviceSize total = vertex_bytes + index_bytes;

	if (mesh.staging_size < total)
	{
		mesh.staging_buffer = buffer_allocation{
		        *device,
		        vk::BufferCreateInfo{
		                .size = total,
		                .usage = vk::BufferUsageFlagBits::eTransferSrc,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "merge layer staging",
		};
		mesh.staging_size = total;
	}
	void * mapped = mesh.staging_buffer.map();
	std::memcpy(mapped, soup.vertices.data(), vertex_bytes);
	std::memcpy((char *)mapped + vertex_bytes, soup.indices.data(), index_bytes);
	mesh.staging_buffer.unmap();
	mesh.vertex_buffer = buffer_allocation{
	        *device,
	        vk::BufferCreateInfo{
	                .size = vertex_bytes,
	                .usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
	        },
	        VmaAllocationCreateInfo{
	                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	        },
	        "merge layer verts",
	};
	mesh.index_buffer = buffer_allocation{
	        *device,
	        vk::BufferCreateInfo{
	                .size = index_bytes,
	                .usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
	        },
	        VmaAllocationCreateInfo{
	                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	        },
	        "merge layer indices",
	};
	mesh.pending_upload = true;
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
	auto it = meshes.find(key);
	if (it == meshes.end())
		return;
	// Same in-flight hazard as replace.
	device->waitIdle();
	meshes.erase(it);
}

std::optional<std::pair<glm::vec3, glm::vec3>> union_merge_renderer::layer_mesh_bounds(const std::string & key) const
{
	auto it = meshes.find(key);
	if (it == meshes.end())
		return std::nullopt;
	return std::make_pair(it->second.aabb_min, it->second.aabb_max);
}

void union_merge_renderer::flush_upload(vk::raii::CommandBuffer & cmd, layer_mesh & mesh)
{
	if (not mesh.pending_upload)
		return;
	mesh.pending_upload = false;
	// Byte counts derive from the upload (VMA sizes round up and must
	// never size the copies: the index span starts at vertex_bytes in
	// staging by construction).
	vk::DeviceSize index_bytes = (vk::DeviceSize)mesh.index_count * sizeof(uint32_t);
	vk::DeviceSize vertex_bytes = mesh.staging_size - index_bytes;
	vk::BufferCopy vertex_copy{
	        .srcOffset = 0,
	        .dstOffset = 0,
	        .size = vertex_bytes,
	};
	cmd.copyBuffer((vk::Buffer)mesh.staging_buffer, (vk::Buffer)mesh.vertex_buffer, vertex_copy);
	vk::BufferCopy index_copy{
	        .srcOffset = vertex_bytes,
	        .dstOffset = 0,
	        .size = index_bytes,
	};
	cmd.copyBuffer((vk::Buffer)mesh.staging_buffer, (vk::Buffer)mesh.index_buffer, index_copy);
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                    vk::PipelineStageFlagBits::eVertexInput,
	                    {},
	                    {},
	                    {vk::BufferMemoryBarrier{
	                             .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                             .dstAccessMask = vk::AccessFlagBits::eVertexAttributeRead,
	                             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                             .buffer = (vk::Buffer)mesh.vertex_buffer,
	                             .offset = 0,
	                             .size = vertex_bytes,
	                     },
	                     vk::BufferMemoryBarrier{
	                             .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                             .dstAccessMask = vk::AccessFlagBits::eIndexRead,
	                             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                             .buffer = (vk::Buffer)mesh.index_buffer,
	                             .offset = 0,
	                             .size = index_bytes,
	                     }},
	                    {});
}

static vk::raii::ImageView make_layer_view(vk::raii::Device & device,
                                             image_allocation & image,
                                             vk::Format format)
{
	return vk::raii::ImageView(device, vk::ImageViewCreateInfo{
	        .image = (VkImage)image,
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
}

void union_merge_renderer::ensure_layer_targets(vk::Extent2D w)
{
	if ((layer_extent_.width == w.width and layer_extent_.height == w.height and w.width != 0) and
	    (merged_ready_ or format_ == vk::Format::eUndefined or extent_.width == 0))
		return;
	// Rare path (first use, resize, first ensure): waitIdle retires
	// in-flight frames using the old views (the feather borrowed
	// framebuffer keys on the image handle, so a stale entry can never
	// alias a live image). A mid-frame rebuild only ever replaces
	// targets this command buffer has not referenced yet (see below).
	device->waitIdle();
	auto make_target = [&](layer_target & tgt, vk::Format fmt, vk::Extent2D e, vk::raii::RenderPass & rp, const char * name) {
		tgt.image = image_allocation{
		        *device,
		        vk::ImageCreateInfo{
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
		        },
		        VmaAllocationCreateInfo{
		                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		        },
		        name,
		};
		tgt.view_array = make_layer_view(*device, tgt.image, fmt);
		vk::ImageView raw = *tgt.view_array;
		tgt.framebuffer = vk::raii::Framebuffer(*device, vk::FramebufferCreateInfo{
		        .renderPass = *rp,
		        .attachmentCount = 1,
		        .pAttachments = &raw,
		        .width = e.width,
		        .height = e.height,
		        .layers = 2,
		});
	};
	bool layer_current = layer_extent_.width == w.width and layer_extent_.height == w.height and w.width != 0;
	if (!layer_current)
	{
		// Full rebuild (first use or W resize): runs at record_base
		// time, before anything in this command buffer references the
		// images. The merged target joins only when the swapchain
		// format/extent are committed (ensure() runs later on early
		// frames); otherwise it defers to the merged-only path below.
		make_target(layer, layer_format, w, layer_renderpass, "merge layer intermediate");
		make_target(verdict, blur_format, w, verdict_renderpass, "merge layer verdict");
		make_target(virt, virt_depth_format, w, volume_virt_renderpass, "merge layer virt");
		layer_extent_ = w;
		layer_needs_discard_ = true;
		merged_ready_ = false;
		if (format_ != vk::Format::eUndefined and extent_.width != 0)
		{
			make_target(merged, format_, extent_, renderpass, "merge layer merged");
			merged_ready_ = true;
		}
		spdlog::info("Union merge layer targets: {}x{} (R16F intermediate + R8 verdict + merged)", w.width, w.height);
	}
	else if (!merged_ready_ and format_ != vk::Format::eUndefined and extent_.width != 0)
	{
		// Merged-only (ensure() after record_base already built the
		// borrowed set this frame): the feather depth stage may already
		// have recorded into the layer in this command buffer, so the
		// borrowed set is never touched here.
		make_target(merged, format_, extent_, renderpass, "merge layer merged");
		merged_ready_ = true;
	}
}

vk::Image union_merge_renderer::layer_image()
{
	// Extent-gated (never dereference a possibly-null view: set only on
	// successful build, see ensure_layer_targets).
	if (layer_extent_.width == 0)
		return vk::Image{};
	return (VkImage)layer.image;
}

vk::ImageView union_merge_renderer::layer_view()
{
	if (layer_extent_.width == 0)
		return vk::ImageView{};
	return *layer.view_array;
}

vk::Extent2D union_merge_renderer::layer_extent()
{
	return layer_extent_;
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

	// Single-attachment multiview renderpass (CLEAR/STORE/OPTIMAL:
	// submitted image, keeps Adreno UBWC live like the feather groups).
	vk::AttachmentDescription attachment{
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

	// Point twin of the merge sampler (exact taps: the verdict layer
	// input and the merged alpha must read texel-exact values, never
	// bilinear blends — the verdict bilinear-upscales afterwards).
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

	// Common-layer geometry raster: linear meters (geom_depth_mv) MIN
	// over the FAR clear, CULL_BACK (single-sided windows correct from
	// outside; closed meshes resolve to the front wall either way).
	// Own R16F CLEAR/STORE/OPTIMAL MV renderpass (stage-0 writes the
	// image fresh every frame, so CLEAR — the feather sensor stage
	// re-begins it under LOAD) + 128B VERT|FRAG push layout. No
	// maxPushConstantsSize throw here (no physical device in this
	// renderer): the feather ctor throws below 132B on the same device,
	// which subsumes this 128B range.
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
		geom_layout = vk::raii::PipelineLayout(*device, geom_layout_info);
	}
	uint32_t layer_mv_mask = 0b11, layer_mv_corr = 0b11;
	vk::RenderPassMultiviewCreateInfo layer_mv_info{
	        .subpassCount = 1,
	        .pViewMasks = &layer_mv_mask,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &layer_mv_corr,
	};
	vk::AttachmentDescription layer_attachment{
	        .format = layer_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference layer_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription layer_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &layer_ref,
	};
	layer_renderpass = vk::raii::RenderPass(*device, vk::RenderPassCreateInfo{
	        .pNext = &layer_mv_info,
	        .attachmentCount = 1,
	        .pAttachments = &layer_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &layer_subpass,
	});
	{
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
		vk::PipelineInputAssemblyStateCreateInfo geom_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo geom_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo geom_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eBack,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo geom_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState geom_blend_attachment{
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
		vk::PipelineColorBlendStateCreateInfo geom_blend{
		        .attachmentCount = 1,
		        .pAttachments = &geom_blend_attachment,
		};
		vk::DynamicState geom_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo geom_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = geom_dynamics,
		};
		vk::GraphicsPipelineCreateInfo geom_pipeline_info{
		        .stageCount = 2,
		        .pStages = geom_stages,
		        .pVertexInputState = &geom_vertex_input,
		        .pInputAssemblyState = &geom_input_assembly,
		        .pViewportState = &geom_viewport_state,
		        .pRasterizationState = &geom_rasterization,
		        .pMultisampleState = &geom_multisample,
		        .pColorBlendState = &geom_blend,
		        .pDynamicState = &geom_dynamic,
		        .layout = *geom_layout,
		        .renderPass = *layer_renderpass,
		};
		geom_pipeline = vk::raii::Pipeline(*device, *pipe_cache, geom_pipeline_info);
	}
	// Virtual-depth SBS -> D canvas (moved verbatim from the feather
	// renderer): fullscreen MV at W, meters->D with the volume
	// near/far push. Canvas cleared to far (1.0); unpaired frames clear
	// it to far without drawing (neutral: the layer always passes the
	// virt test). Written with plain STORE (every texel covered).
	{
		auto virt_frag = load_shader(*device, "volume_virt_mv.frag");
		vk::PipelineShaderStageCreateInfo virt_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **vert,
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
		volume_virt_set_layout = vk::raii::DescriptorSetLayout(*device, virt_set_info);
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
		volume_virt_layout = vk::raii::PipelineLayout(*device, virt_pipe_layout_info);
		vk::DescriptorPoolSize virt_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 1,
		};
		vk::DescriptorPoolCreateInfo virt_pool_info{
		        .maxSets = 1,
		        .poolSizeCount = 1,
		        .pPoolSizes = &virt_pool_size,
		};
		volume_virt_pool = vk::raii::DescriptorPool(*device, virt_pool_info);
		vk::DescriptorSetAllocateInfo virt_alloc_info{
		        .descriptorPool = *volume_virt_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*volume_virt_set_layout,
		};
		volume_virt_set = std::move(device->allocateDescriptorSets(virt_alloc_info)[0]);
		uint32_t virt_mv_mask = 0b11, virt_mv_corr = 0b11;
		vk::RenderPassMultiviewCreateInfo virt_mv_rp_info{
		        .subpassCount = 1,
		        .pViewMasks = &virt_mv_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &virt_mv_corr,
		};
		vk::AttachmentDescription virt_attachment{
		        .format = virt_depth_format,
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
		volume_virt_renderpass = vk::raii::RenderPass(*device, vk::RenderPassCreateInfo{
		        .pNext = &virt_mv_rp_info,
		        .attachmentCount = 1,
		        .pAttachments = &virt_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &virt_subpass,
		});
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
		volume_virt_pipeline = vk::raii::Pipeline(*device, *pipe_cache, virt_pipeline_info);
	}
	// Exact global verdict (R16F layer vs virt D, nearest/nearest) into
	// an R8 canvas at W. Own CLEAR renderpass + 8B near/far push
	// (linearizes virt; the layer is already meters). No bias anywhere
	// global: crisp occlusion edges are intentional.
	{
		auto verdict_frag = load_shader(*device, "depth_verdict_mv.frag");
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
		vk::DescriptorSetLayoutBinding verdict_bindings[2] = {
		        {
		                // R16F common layer (linear meters, NEAREST).
		                .binding = 0,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		        {
		                // Virt D canvas (NEAREST: exact compare).
		                .binding = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		};
		vk::DescriptorSetLayoutCreateInfo verdict_set_info{
		        .bindingCount = 2,
		        .pBindings = verdict_bindings,
		};
		verdict_set_layout = vk::raii::DescriptorSetLayout(*device, verdict_set_info);
		vk::PushConstantRange verdict_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(depth_verdict_push),
		};
		vk::PipelineLayoutCreateInfo verdict_pipe_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*verdict_set_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &verdict_push_range,
		};
		verdict_pipe_layout = vk::raii::PipelineLayout(*device, verdict_pipe_layout_info);
		vk::DescriptorPoolSize verdict_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 2,
		};
		vk::DescriptorPoolCreateInfo verdict_pool_info{
		        .maxSets = 1,
		        .poolSizeCount = 1,
		        .pPoolSizes = &verdict_pool_size,
		};
		verdict_pool = vk::raii::DescriptorPool(*device, verdict_pool_info);
		vk::DescriptorSetAllocateInfo verdict_alloc_info{
		        .descriptorPool = *verdict_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*verdict_set_layout,
		};
		verdict_set = std::move(device->allocateDescriptorSets(verdict_alloc_info)[0]);
		uint32_t verdict_mv_mask = 0b11, verdict_mv_corr = 0b11;
		vk::RenderPassMultiviewCreateInfo verdict_mv_rp_info{
		        .subpassCount = 1,
		        .pViewMasks = &verdict_mv_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &verdict_mv_corr,
		};
		vk::AttachmentDescription verdict_attachment{
		        .format = blur_format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eClear,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eUndefined,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference verdict_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription verdict_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &verdict_ref,
		};
		verdict_renderpass = vk::raii::RenderPass(*device, vk::RenderPassCreateInfo{
		        .pNext = &verdict_mv_rp_info,
		        .attachmentCount = 1,
		        .pAttachments = &verdict_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &verdict_subpass,
		});
		vk::PipelineVertexInputStateCreateInfo verdict_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo verdict_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo verdict_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo verdict_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo verdict_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState verdict_blend_attachment{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo verdict_blend{
		        .attachmentCount = 1,
		        .pAttachments = &verdict_blend_attachment,
		};
		vk::DynamicState verdict_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo verdict_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = verdict_dynamics,
		};
		vk::GraphicsPipelineCreateInfo verdict_pipeline_info{
		        .stageCount = 2,
		        .pStages = verdict_stages,
		        .pVertexInputState = &verdict_vertex_input,
		        .pInputAssemblyState = &verdict_input_assembly,
		        .pViewportState = &verdict_viewport_state,
		        .pRasterizationState = &verdict_rasterization,
		        .pMultisampleState = &verdict_multisample,
		        .pColorBlendState = &verdict_blend,
		        .pDynamicState = &verdict_dynamic,
		        .layout = *verdict_pipe_layout,
		        .renderPass = *verdict_renderpass,
		};
		verdict_pipeline = vk::raii::Pipeline(*device, *pipe_cache, verdict_pipeline_info);
	}
	// Verdict apply (merged alpha x bilinear verdict -> submit). Own
	// 2-binding layout (merged NEAREST + verdict LINEAR), no push.
	// Renders under the submit renderpass (same RGBA CLEAR/STORE/OPTIMAL
	// MV shape as the merge; loadOp is not part of compatibility).
	{
		auto apply_frag = load_shader(*device, "depth_apply_mv.frag");
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
		vk::DescriptorSetLayoutBinding apply_bindings[2] = {
		        {
		                // Merged contributor alpha (full res, NEAREST).
		                .binding = 0,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		        {
		                // Binary verdict at W (LINEAR: the upscale).
		                .binding = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 1,
		                .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        },
		};
		vk::DescriptorSetLayoutCreateInfo apply_set_info{
		        .bindingCount = 2,
		        .pBindings = apply_bindings,
		};
		apply_set_layout = vk::raii::DescriptorSetLayout(*device, apply_set_info);
		vk::PipelineLayoutCreateInfo apply_pipe_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*apply_set_layout,
		};
		apply_pipe_layout = vk::raii::PipelineLayout(*device, apply_pipe_layout_info);
		vk::DescriptorPoolSize apply_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 2,
		};
		vk::DescriptorPoolCreateInfo apply_pool_info{
		        .maxSets = 1,
		        .poolSizeCount = 1,
		        .pPoolSizes = &apply_pool_size,
		};
		apply_pool = vk::raii::DescriptorPool(*device, apply_pool_info);
		vk::DescriptorSetAllocateInfo apply_alloc_info{
		        .descriptorPool = *apply_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*apply_set_layout,
		};
		apply_set = std::move(device->allocateDescriptorSets(apply_alloc_info)[0]);
		vk::PipelineVertexInputStateCreateInfo apply_vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo apply_input_assembly{
		        .topology = vk::PrimitiveTopology::eTriangleList,
		};
		vk::PipelineViewportStateCreateInfo apply_viewport_state{
		        .viewportCount = 1,
		        .scissorCount = 1,
		};
		vk::PipelineRasterizationStateCreateInfo apply_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo apply_multisample{
		        .rasterizationSamples = vk::SampleCountFlagBits::e1,
		};
		vk::PipelineColorBlendAttachmentState apply_blend_attachment{
		        .blendEnable = VK_FALSE,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		};
		vk::PipelineColorBlendStateCreateInfo apply_blend{
		        .attachmentCount = 1,
		        .pAttachments = &apply_blend_attachment,
		};
		vk::DynamicState apply_dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo apply_dynamic{
		        .dynamicStateCount = 2,
		        .pDynamicStates = apply_dynamics,
		};
		vk::GraphicsPipelineCreateInfo apply_pipeline_info{
		        .stageCount = 2,
		        .pStages = apply_stages,
		        .pVertexInputState = &apply_vertex_input,
		        .pInputAssemblyState = &apply_input_assembly,
		        .pViewportState = &apply_viewport_state,
		        .pRasterizationState = &apply_rasterization,
		        .pMultisampleState = &apply_multisample,
		        .pColorBlendState = &apply_blend,
		        .pDynamicState = &apply_dynamic,
		        .layout = *apply_pipe_layout,
		        .renderPass = *renderpass,
		};
		apply_pipeline = vk::raii::Pipeline(*device, *pipe_cache, apply_pipeline_info);
	}
}



void union_merge_renderer::record_base(vk::raii::CommandBuffer & cmd,
                                       vk::Extent2D work_extent,
                                       const std::vector<feather_mask_renderer::instance_draw> & window_draws,
                                       vk::QueryPool ts_pool,
                                       cpu_stats * cpu)
{
	double ms = 0;
	section_clock clk(cpu ? &cpu->base_ms : &ms);
	// Slot 0 (top): the scene reset the pool just before (same pattern
	// as the feather record: reset-per-frame, never here, so a throw in
	// a later phase leaves a partial but honest pool).
	if (ts_pool != vk::QueryPool{})
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, ts_pool, 0);

	// Layer target always exists after this (change-gated, waitIdle on
	// rebuild): a LOAD without defined contents is garbage, so the
	// empty-draws path below still CLEARs.
	ensure_layer_targets(work_extent);
	for (auto & [key, mesh]: meshes)
		flush_upload(cmd, mesh);

	// Pre-write: prior reads (last frame's verdict/add or a fresh build)
	// out, color write in. Fresh builds discard (UNDEFINED); recorded
	// frames transition from READ_ONLY (the invariant every phase ends
	// in — the feather prologue relies on it too).
	vk::Image layer_img = (VkImage)layer.image;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
	                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    {},
	                    {},
	                    {},
	                    vk::ImageMemoryBarrier{
	                            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
	                            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	                            .oldLayout = layer_needs_discard_ ? vk::ImageLayout::eUndefined
	                                                             : vk::ImageLayout::eShaderReadOnlyOptimal,
	                            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
	                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                            .image = layer_img,
	                            .subresourceRange = {
	                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                    .levelCount = 1,
	                                    .layerCount = 2,
	                            },
	                    });
	layer_needs_discard_ = false;
	cmd.setViewport(0, vk::Viewport{
	                        .x = 0,
	                        .y = 0,
	                        .width = (float)work_extent.width,
	                        .height = (float)work_extent.height,
	                        .minDepth = 0,
	                        .maxDepth = 1,
	                });
	cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = work_extent});
	vk::ClearValue far{};
	const float farv[4] = {layer_far_clear, layer_far_clear, layer_far_clear, layer_far_clear};
	std::memcpy(far.color.float32, farv, sizeof(farv));
	vk::RenderPassBeginInfo begin{
	        .renderPass = *layer_renderpass,
	        .framebuffer = *layer.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = work_extent},
	        .clearValueCount = 1,
	        .pClearValues = &far,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *geom_pipeline);
	for (const auto & d: window_draws)
	{
		auto mit = meshes.find(d.mesh);
		if (mit == meshes.end() or mit->second.index_count == 0)
			continue;
		cmd.bindVertexBuffers(0, (vk::Buffer)mit->second.vertex_buffer, (vk::DeviceSize)0);
		cmd.bindIndexBuffer((vk::Buffer)mit->second.index_buffer, 0, vk::IndexType::eUint32);
		cmd.pushConstants<geom_depth_push>(*geom_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                                  geom_depth_push{.mvp = {{d.mvp[0], d.mvp[1]}}});
		cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
	}
	cmd.endRenderPass();
	// Every frame ends READ_ONLY (see above): the feather geometry
	// prologue re-begins from there, and the verdict samples it.
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
	                            .image = layer_img,
	                            .subresourceRange = {
	                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                    .levelCount = 1,
	                                    .layerCount = 2,
	                            },
	                    });
	if (ts_pool != vk::QueryPool{})
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, ts_pool, 1);
}

bool union_merge_renderer::record(vk::raii::CommandBuffer & cmd,
                                  vk::Extent2D extent,
                                  std::span<const input> inputs,
                                  vk::QueryPool ts_pool,
                                  cpu_stats * cpu)
{
	double ms = 0;
	section_clock clk(cpu ? &cpu->merge_ms : &ms);
	if (inputs.empty() or inputs.size() > max_inputs or !merged_ready_)
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
	vk::ClearValue clear{};
	clear.color.float32 = std::array<float, 4>{0, 0, 0, 0};
	vk::RenderPassBeginInfo begin{
	        .renderPass = *renderpass,
	        .framebuffer = *merged.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = extent},
	        .clearValueCount = 1,
	        .pClearValues = &clear,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, set, {});
	// Active input count for the uniform tap early-out (volume_virt
	// FRAG-only precedent): identical for every invocation in the
	// draw, so skipped taps never execute. inputs.size() is 1..8
	// (checked on entry).
	cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{.count = (int32_t)inputs.size()});
	// Fullscreen triangle (blur.vert: 3 verts, no vertex buffers).
	cmd.draw(3, 1, 0, 0);
	cmd.endRenderPass();

	// Post-write: color write out, apply read in. The merged target is
	// an owned image (not the submit): no compositor barrier needed,
	// that rides the apply below.
	vk::Image merged_img = (VkImage)merged.image;
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
	                            .image = merged_img,
	                            .subresourceRange = {
	                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                    .levelCount = 1,
	                                    .layerCount = 2,
	                            },
	                    });
	// Slot 8 (merge end). Write order on the shared pool is
	// 0,1,...,5 then 8,6,7,9: the scene maps merge=8-5, verdict=6-8,
	// apply=7-6 (a chronological 8-7 would read negative).
	if (ts_pool != vk::QueryPool{})
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, ts_pool, 8);
	return true;
}

bool union_merge_renderer::record_finish(vk::raii::CommandBuffer & cmd,
                                         size_t image_index,
                                         vk::Extent2D extent,
                                         const virt_input & virt_in,
                                         vk::QueryPool ts_pool,
                                         cpu_stats * cpu)
{
	double ms_verdict = 0, ms_apply = 0;
	if (image_index >= targets.size() or layer_extent_.width == 0)
		return false;
	auto & tgt = targets[image_index];
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

	{
		section_clock verdict_clk(cpu ? &cpu->verdict_ms : &ms_verdict);
	// Virtual-depth SBS meters->D at W (moved verbatim from the feather
	// renderer). CONVERTs with the volume near/far so the compare stays
	// in D space. Canvas cleared to far (1.0), and unpaired frames
	// clear it to far without drawing — but !valid skips the verdict
	// entirely and clears it to 1 (neutral), so the unpaired path
	// never depends on this canvas.
	vk::Image virt_img = (VkImage)virt.image;
	if (virt_in.valid and virt_in.view != vk::ImageView{})
	{
		vk::DescriptorImageInfo virt_info{
		        .sampler = virt_in.sampler,
		        .imageView = virt_in.view,
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
		device->updateDescriptorSets(virt_writes, no_virt_copies);
		vk::ClearValue vfar{};
		const float vfarv[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		std::memcpy(vfar.color.float32, vfarv, sizeof(vfarv));
		vk::RenderPassBeginInfo begin_virt{
		        .renderPass = *volume_virt_renderpass,
		        .framebuffer = *virt.framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = layer_extent_},
		        .clearValueCount = 1,
		        .pClearValues = &vfar,
		};
		cmd.beginRenderPass(begin_virt, vk::SubpassContents::eInline);
		set_full_viewport(layer_extent_);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *volume_virt_pipeline);
		std::array<vk::DescriptorSet, 1> virt_sets{*volume_virt_set};
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *volume_virt_layout, 0, virt_sets, {});
		cmd.pushConstants<volume_virt_push>(*volume_virt_layout, vk::ShaderStageFlagBits::eFragment, 0,
		                                    volume_virt_push{.near_z = virt_in.near_z, .far_z = virt_in.far_z});
		cmd.draw(3, 1, 0, 0);
		cmd.endRenderPass();
	}
	else
	{
		vk::ClearValue vfar{};
		const float vfarv[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		std::memcpy(vfar.color.float32, vfarv, sizeof(vfarv));
		vk::RenderPassBeginInfo begin_clear{
		        .renderPass = *volume_virt_renderpass,
		        .framebuffer = *virt.framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = layer_extent_},
		        .clearValueCount = 1,
		        .pClearValues = &vfar,
		};
		cmd.beginRenderPass(begin_clear, vk::SubpassContents::eInline);
		cmd.endRenderPass();
	}
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
	                            .image = virt_img,
	                            .subresourceRange = {
	                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                    .levelCount = 1,
	                                    .layerCount = 2,
	                            },
	                    });

	// Exact global verdict (intermediate linear vs virt D, exact,
	// nearest/nearest, R8 at W). !valid submits merged directly: the
	// verdict clears to 1 and the apply below is a 1:1 copy (today's
	// neutral parity, one extra blit).
	vk::DescriptorImageInfo layer_info{
	        .sampler = *sampler_nearest,
	        .imageView = *layer.view_array,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	vk::DescriptorImageInfo virt_d_info{
	        .sampler = *sampler_nearest,
	        .imageView = *virt.view_array,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	std::array<vk::CopyDescriptorSet, 0> no_verdict_copies{};
	std::array<vk::WriteDescriptorSet, 2> verdict_writes = {{
	        {
	                .dstSet = *verdict_set,
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &layer_info,
	        },
	        {
	                .dstSet = *verdict_set,
	                .dstBinding = 1,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &virt_d_info,
	        },
	}};
	device->updateDescriptorSets(verdict_writes, no_verdict_copies);
	vk::ClearValue vzero{};
	const float vzero_v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	const float vone_v[4] = {1.0f, 1.0f, 1.0f, 1.0f};
	if (virt_in.valid and virt_in.view != vk::ImageView{})
		std::memcpy(vzero.color.float32, vzero_v, sizeof(vzero_v));
	else
		std::memcpy(vzero.color.float32, vone_v, sizeof(vone_v));
	vk::RenderPassBeginInfo begin_verdict{
	        .renderPass = *verdict_renderpass,
	        .framebuffer = *verdict.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = layer_extent_},
	        .clearValueCount = 1,
	        .pClearValues = &vzero,
	};
	cmd.beginRenderPass(begin_verdict, vk::SubpassContents::eInline);
	if (virt_in.valid and virt_in.view != vk::ImageView{})
	{
		set_full_viewport(layer_extent_);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *verdict_pipeline);
		std::array<vk::DescriptorSet, 1> verdict_sets{*verdict_set};
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *verdict_pipe_layout, 0, verdict_sets, {});
		cmd.pushConstants<depth_verdict_push>(*verdict_pipe_layout, vk::ShaderStageFlagBits::eFragment, 0,
		                                      depth_verdict_push{.near_z = virt_in.near_z, .far_z = virt_in.far_z});
		cmd.draw(3, 1, 0, 0);
	}
	cmd.endRenderPass();
	vk::Image verdict_img = (VkImage)verdict.image;
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
	                            .image = verdict_img,
	                            .subresourceRange = {
	                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                    .levelCount = 1,
	                                    .layerCount = 2,
	                            },
	                    });
	if (ts_pool != vk::QueryPool{})
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, ts_pool, 6);
	}
	// Apply (merged alpha x bilinear verdict -> submit image). The
	// binary verdict (never the depths) bilinear-upsamples here, so no
	// depth halo.
	section_clock apply_clk(cpu ? &cpu->apply_ms : &ms_apply);
	{
		vk::DescriptorImageInfo merged_info{
		        .sampler = *sampler_nearest,
		        .imageView = *merged.view_array,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		vk::DescriptorImageInfo verdict_info{
		        .sampler = *sampler,
		        .imageView = *verdict.view_array,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		std::array<vk::CopyDescriptorSet, 0> no_apply_copies{};
		std::array<vk::WriteDescriptorSet, 2> apply_writes = {{
		        {
		                .dstSet = *apply_set,
		                .dstBinding = 0,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &merged_info,
		        },
		        {
		                .dstSet = *apply_set,
		                .dstBinding = 1,
		                .descriptorCount = 1,
		                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		                .pImageInfo = &verdict_info,
		        },
		}};
		device->updateDescriptorSets(apply_writes, no_apply_copies);
		vk::ClearValue clear{};
		clear.color.float32 = std::array<float, 4>{0, 0, 0, 0};
		vk::RenderPassBeginInfo begin{
		        .renderPass = *renderpass,
		        .framebuffer = *tgt.framebuffer,
		        .renderArea = {.offset = {0, 0}, .extent = extent},
		        .clearValueCount = 1,
		        .pClearValues = &clear,
		};
		cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
		set_full_viewport(extent);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *apply_pipeline);
		std::array<vk::DescriptorSet, 1> apply_sets{*apply_set};
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *apply_pipe_layout, 0, apply_sets, {});
		cmd.draw(3, 1, 0, 0);
		cmd.endRenderPass();
	}
	if (ts_pool != vk::QueryPool{})
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, ts_pool, 7);

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
	if (ts_pool != vk::QueryPool{})
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, ts_pool, 9);
	return true;
}
