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
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
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

// Common-layer geometry push: per-eye MVP pair, VERT-only. 128B is
// exactly the Vulkan-guaranteed minimum, so no device-limit throw is
// needed; the fragment declares no push block, so the shared-range
// offset trap cannot fire.
struct geom_push
{
	std::array<glm::mat4, 2> mvp;
};
static_assert(sizeof(geom_push) == 128);

// Expand compact foveation runs to a direct packed-index map per merge
// pixel, baking the merge->render scale. Layout: [x0(W)] [x1(W)]
// [y0(H)] [y1(H)] uint32s, W/H = merge extent. Runs describe packed
// texels: entry i holds n_i texels covering n_i * ratio_i source
// pixels (ratio_i = |n_ratio - i| + 1, middle run 1:1). Inverting gives
// render->packed; the merge scale then picks the packed texel per
// merge pixel (uniform floor, clamped). Empty runs fall back to a
// uniform map over the packed size (straight / pre-foveation data).
static void expand_foveation_axis(const std::vector<uint16_t> & runs, int render_n, int merge_n, int packed_n, uint32_t * out)
{
	if (render_n <= 0 or merge_n <= 0)
	{
		for (int u = 0; u < merge_n; ++u)
			out[u] = 0;
		return;
	}
	if (runs.empty())
	{
		for (int u = 0; u < merge_n; ++u)
			out[u] = packed_n > 0 ? (uint32_t)std::min<int64_t>((int64_t)u * packed_n / merge_n, packed_n - 1) : 0;
		return;
	}
	// Direct map: render pixel -> packed texel.
	std::vector<uint32_t> r2p;
	r2p.reserve((size_t)render_n);
	const int n_ratio = (int)(runs.size() - 1) / 2;
	uint32_t packed = 0;
	for (size_t i = 0; i < runs.size() and (int)r2p.size() < render_n; ++i)
	{
		int ratio = std::abs(n_ratio - (int)i) + 1;
		for (uint16_t k = 0; k < runs[i] and (int)r2p.size() < render_n; ++k)
		{
			for (int r = 0; r < ratio and (int)r2p.size() < render_n; ++r)
				r2p.push_back(packed + k);
		}
		packed += runs[i];
	}
	while ((int)r2p.size() < render_n)
		r2p.push_back(r2p.empty() ? 0 : r2p.back());
	const uint32_t packed_max = packed_n > 0 ? (uint32_t)packed_n - 1 : 0;
	for (int u = 0; u < merge_n; ++u)
	{
		int64_t r = (int64_t)u * render_n / merge_n;
		uint32_t v = r2p[(size_t)std::min<int64_t>(r, render_n - 1)];
		// Clamp to the packed range: the runs should tile exactly it,
		// but an overrun here would read the other eye's half (eye 0)
		// or ClampToEdge texels (eye 1) — silent wrong depth.
		out[u] = packed_n > 0 ? std::min(v, packed_max) : 0;
	}
}

uint32_t find_memory_type(vk::raii::PhysicalDevice & physical_device, uint32_t type_bits, vk::MemoryPropertyFlags properties)
{
	auto mem = physical_device.getMemoryProperties();
	for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
	{
		if ((type_bits & (1u << i)) and (mem.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	throw std::runtime_error("union_merge: no suitable memory type");
}

// Common depth layer (R16F linear meters, MIN-blended) and binary
// verdict (R8) formats. R16F needs COLOR_ATTACHMENT + BLEND on this
// device — checked loudly in init_pipelines (no silent fallback: a
// missing MIN blend would under/over-reveal silently).
constexpr vk::Format layer_format = vk::Format::eR16Sfloat;
constexpr float layer_far = 65504.f;
constexpr vk::Format verdict_format = vk::Format::eR8Unorm;
} // namespace

union_merge_renderer::union_merge_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device) :
        device(&device),
        physical_device(&physical_device)
{
}

std::shared_ptr<vk::raii::PipelineCache> union_merge_renderer::pipeline_cache(vk::raii::Device & device)
{
	// Same process-wide object shared with the feather mask renderer
	// (one entry per VkDevice).
	return feather_mask_renderer::pipeline_cache(device);
}

void union_merge_renderer::warm_pipelines(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device, vk::Format format)
{
	// Throwaway renderer: its pipeline compile populates the shared
	// cache above, so the first real merge pays ~nothing.
	union_merge_renderer warm(device, physical_device);
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
	// (destroys the use of) any sets still held. Fold framebuffer goes
	// first: it references views rebuilt below (waitIdle'd, so no
	// in-flight use — this just removes the transient dangle).
	fold_fbs.clear();
	descriptor_sets.clear();
	verdict_descriptor_sets.clear();
	apply_descriptor_sets.clear();
	dilate_sets.clear();
	writeback_sets.clear();
	layer_load_fb.clear();
	copy_load_fb.clear();
	copy_dilate_fb.clear();
	copy_tmp_dilate_fb.clear();
	// format_/extent_ commit AFTER the fallible rebuild below: if the
	// swapchain or pipeline construction throws, the old values stay
	// and the next active frame retries instead of the early-out
	// above suppressing recovery forever against stale targets.
	vk::Format new_format = format;
	vk::Extent2D new_extent{(uint32_t)w, (uint32_t)h};
	// Uniform-half working res (opt #U) must divide evenly: the
	// apply's 2x verdict upscale samples symmetric texel pairs only
	// for even dims (NEAREST broadcast positions; LINEAR tent over
	// them). Merge dims are 64-aligned, so this always holds; assert
	// it loudly.
	assert(w % 2 == 0 and h % 2 == 0);
	vk::Extent2D half_extent{(uint32_t)w / 2, (uint32_t)h / 2};
	swapchain_ = xr::swapchain(inst, sess, *device, new_format, w, h, 1, 2);
	spdlog::info("Union merge swapchain: {}x{} ({} images)", w, h, swapchain_.image_count());
	init_pipelines(new_format);
	// One multiview view per swapchain image: 2-layer array view
	// (the 2-layer routing comes from the multiview view mask, not
	// the framebuffer); one draw broadcasts to both eyes. The folded
	// framebuffers below bind these views with the owned targets.
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
	}
	// Owned targets (not swapchain): layer + verdict at uniform half
	// res (opt #U, relative to the merge size), merged at merge
	// (config-max) res. Rebuilt under the same
	// waitIdle above; the mesh buffers survive (extent-independent).
	// The layer and verdict keep their own framebuffers (own RPs);
	// merged rides the per-image fold framebuffer (null renderpass =
	// no FB).
	auto make_owned = [&](vk::Format fmt, vk::Extent2D e, vk::ImageUsageFlags usage, vk::raii::RenderPass * rp) {
		layer_target t;
		vk::ImageCreateInfo image_info{
		        .imageType = vk::ImageType::e2D,
		        .format = fmt,
		        .extent = {e.width, e.height, 1},
		        .mipLevels = 1,
		        .arrayLayers = 2,
		        .samples = vk::SampleCountFlagBits::e1,
		        .tiling = vk::ImageTiling::eOptimal,
		        .usage = usage,
		        .sharingMode = vk::SharingMode::eExclusive,
		        .initialLayout = vk::ImageLayout::eUndefined,
		};
		t.image = vk::raii::Image(*device, image_info);
		auto requirements = t.image.getMemoryRequirements();
		vk::MemoryAllocateInfo alloc_info{
		        .allocationSize = requirements.size,
		        .memoryTypeIndex = find_memory_type(*physical_device, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
		};
		t.memory = vk::raii::DeviceMemory(*device, alloc_info);
		t.image.bindMemory(*t.memory, 0);
		vk::ImageViewCreateInfo iv_info{
		        .image = *t.image,
		        .viewType = vk::ImageViewType::e2DArray,
		        .format = fmt,
		        .subresourceRange = {
		                .aspectMask = vk::ImageAspectFlagBits::eColor,
		                .baseMipLevel = 0,
		                .levelCount = 1,
		                .baseArrayLayer = 0,
		                .layerCount = 2,
		        },
		};
		t.view_array = vk::raii::ImageView(*device, iv_info);
		if (rp)
		{
			vk::ImageView view = *t.view_array;
			vk::FramebufferCreateInfo fb_info{
			        .renderPass = **rp,
			        .attachmentCount = 1,
			        .pAttachments = &view,
			        .width = e.width,
			        .height = e.height,
			        .layers = 1,
			};
			t.framebuffer = vk::raii::Framebuffer(*device, fb_info);
		}
		return t;
	};
	layer_ = make_owned(layer_format, half_extent,
	                     vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
	                             vk::ImageUsageFlagBits::eTransferSrc,
	                     &geom_renderpass);
	merged_ = make_owned(new_format, new_extent, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, nullptr);
	verdict_ = make_owned(verdict_format, half_extent, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, &verdict_rp);
	// Sensor-compare copy + dilate scratch (R16F half res): sampled by
	// the feather compare, written by record_copy below. TRANSFER_SRC
	// (layer clone) + TRANSFER_DST (copy receive) on top of the
	// attachment/sampled use. FBs ride the LOAD twin (copy raster) and
	// the dilate RP (H+V fullscreen).
	copy_ = make_owned(layer_format, half_extent,
	                     vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
	                             vk::ImageUsageFlagBits::eTransferDst,
	                     nullptr);
	copy_tmp_ = make_owned(layer_format, half_extent,
	                         vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
	                         nullptr);
	{
		vk::ImageView copy_view = *copy_.view_array;
		vk::FramebufferCreateInfo copy_load_fb_info{
		        .renderPass = *geom_load_rp,
		        .attachmentCount = 1,
		        .pAttachments = &copy_view,
		        .width = half_extent.width,
		        .height = half_extent.height,
		        .layers = 1,
		};
		copy_load_fb = vk::raii::Framebuffer(*device, copy_load_fb_info);
		vk::FramebufferCreateInfo copy_dilate_fb_info{
		        .renderPass = *dilate_rp,
		        .attachmentCount = 1,
		        .pAttachments = &copy_view,
		        .width = half_extent.width,
		        .height = half_extent.height,
		        .layers = 1,
		};
		copy_dilate_fb = vk::raii::Framebuffer(*device, copy_dilate_fb_info);
		vk::ImageView tmp_view = *copy_tmp_.view_array;
		vk::FramebufferCreateInfo tmp_dilate_fb_info{
		        .renderPass = *dilate_rp,
		        .attachmentCount = 1,
		        .pAttachments = &tmp_view,
		        .width = half_extent.width,
		        .height = half_extent.height,
		        .layers = 1,
		};
		copy_tmp_dilate_fb = vk::raii::Framebuffer(*device, tmp_dilate_fb_info);
	}
	// Shared-layer LOAD framebuffer for the sensor write-back (MIN blend
	// preserves the rastered windows): the write-back pass below re-begins
	// it under geom_load_rp (loadOp is not part of compatibility).
	{
		vk::ImageView layer_view = *layer_.view_array;
		layer_load_fb = vk::raii::Framebuffer(*device, vk::FramebufferCreateInfo{
		                                                        .renderPass = *geom_load_rp,
		                                                        .attachmentCount = 1,
		                                                        .pAttachments = &layer_view,
		                                                        .width = half_extent.width,
		                                                        .height = half_extent.height,
		                                                        .layers = 1,
		                                                });
	}
	layer_extent_ = new_extent;
	half_extent_ = half_extent;
	// Folded framebuffer, one per submit image (submit view varies):
	// {merged, submit}. Attachments and order must match the fold RP
	// definition in init_pipelines.
	fold_fbs.clear();
	for (auto & tgt: targets)
	{
		vk::ImageView views2[2] = {*merged_.view_array, *tgt.view};
		fold_fbs.emplace_back(*device, vk::FramebufferCreateInfo{
		                                         .renderPass = *fold_rp,
		                                         .attachmentCount = 2,
		                                         .pAttachments = views2,
		                                         .width = new_extent.width,
		                                         .height = new_extent.height,
		                                         .layers = 1,
		                                 });
	}
	// Foveation index maps (SSBO pair, host-visible): per-eye packed
	// index per half-res loop pixel, same layout as the shader reads.
	verdict_tables_bytes = vk::DeviceSize(2 * (half_extent.width + half_extent.height)) * sizeof(uint32_t);
	for (auto & t: verdict_tables)
	{
		t = buffer_allocation(
		        *device,
		        vk::BufferCreateInfo{
		                .size = verdict_tables_bytes,
		                .usage = vk::BufferUsageFlagBits::eStorageBuffer,
		        },
		        VmaAllocationCreateInfo{
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "union merge verdict tables");
	}
	spdlog::info("Union merge layer targets: layer {}x{} merged {}x{} verdict {}x{} tables {}B",
	             half_extent.width, half_extent.height, new_extent.width, new_extent.height,
	             half_extent.width, half_extent.height, verdict_tables_bytes);
	// Commit only after the whole rebuild succeeded (see above).
	format_ = new_format;
	extent_ = new_extent;
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
	// tap IS the upscale for half/quarter-res inputs, and (AA spike)
	// for full-res tier-0 inputs too, so grey coverage edges filter.
	// maxLod 0, no mipmap chain on swapchain images. The 4-tap cost on
	// 1:1 copies is accepted for the trial (interiors are bit-identical
	// either way); revert restores the NEAREST 1:1 fast path.
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

	// One fullscreen pipeline per fold subpass (subpass index baked):
	// identical stages/layout/state to the old split passes, only the
	// renderpass + subpass differ.
	auto make_fold_pipeline = [&](vk::PipelineShaderStageCreateInfo * stages,
	                              vk::raii::PipelineLayout & layout,
	                              vk::raii::RenderPass & rp, uint32_t subpass) {
		return vk::raii::Pipeline(*device, *pipe_cache, vk::GraphicsPipelineCreateInfo{
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
		                                                              .layout = *layout,
		                                                              .renderPass = *rp,
		                                                              .subpass = subpass,
		                                                      });
	};

	// Folded merge→apply renderpass (opt #8): merge s0 + apply s1
	// as subpasses of one RP. The verdict runs ahead in its own RP
	// (different size) and is sampled as a texture. Merged stays
	// tile-resident (triple precedent: GENERAL attachments sampled
	// across subpasses, subpass dep replaces the split-pass barrier).
	// Both loads DONT_CARE (fullscreen overwrites); merged is
	// DONT_CARE-stored (no consumer outside the RP); submit STOREs +
	// ends OPTIMAL (compositor unchanged).
	vk::AttachmentDescription fold_attachments[2] = {
	        {
	                .format = format,
	                .samples = vk::SampleCountFlagBits::e1,
	                .loadOp = vk::AttachmentLoadOp::eDontCare,
	                .storeOp = vk::AttachmentStoreOp::eDontCare,
	                .initialLayout = vk::ImageLayout::eUndefined,
	                .finalLayout = vk::ImageLayout::eGeneral,
	        },
	        {
	                .format = format,
	                .samples = vk::SampleCountFlagBits::e1,
	                .loadOp = vk::AttachmentLoadOp::eDontCare,
	                .storeOp = vk::AttachmentStoreOp::eStore,
	                .initialLayout = vk::ImageLayout::eUndefined,
	                .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	        },
	};
	vk::AttachmentReference fold_refs[2] = {
	        {.attachment = 0, .layout = vk::ImageLayout::eGeneral},
	        {.attachment = 1, .layout = vk::ImageLayout::eGeneral},
	};
	vk::SubpassDescription fold_subpasses[2] = {
	        {
	                .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	                .colorAttachmentCount = 1,
	                .pColorAttachments = &fold_refs[0],
	        },
	        {
	                .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	                .colorAttachmentCount = 1,
	                .pColorAttachments = &fold_refs[1],
	        },
	};
	vk::SubpassDependency fold_dep{
	        .srcSubpass = 0,
	        .dstSubpass = 1,
	        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
	        .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
	        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
	};
	uint32_t fold_view_masks[2] = {0b11, 0b11};
	uint32_t fold_corr_mask = 0b11;
	vk::RenderPassMultiviewCreateInfo fold_mv_info{
	        .subpassCount = 2,
	        .pViewMasks = fold_view_masks,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &fold_corr_mask,
	};
	vk::RenderPassCreateInfo fold_rp_info{
	        .pNext = &fold_mv_info,
	        .attachmentCount = 2,
	        .pAttachments = fold_attachments,
	        .subpassCount = 2,
	        .pSubpasses = fold_subpasses,
	        .dependencyCount = 1,
	        .pDependencies = &fold_dep,
	};
	fold_rp = vk::raii::RenderPass(*device, fold_rp_info);

	// Merge subpass-0 pipeline (same stages/push layout as the old
	// split merge; subpass index baked).
	fold_merge_pipeline = make_fold_pipeline(stages, pipeline_layout, fold_rp, 0);

	// Common depth layer format gate (see above): R16F must support
	// COLOR_ATTACHMENT + BLEND on this device. No silent fallback.
	{
		auto props = physical_device->getFormatProperties(layer_format);
		bool ok_color = (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachment) != vk::FormatFeatureFlags{};
		bool ok_blend = (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eColorAttachmentBlend) != vk::FormatFeatureFlags{};
		if (not ok_color or not ok_blend)
			throw std::runtime_error("union_merge: R16F lacks COLOR_ATTACHMENT+BLEND (common depth layer unsupported on this device)");
	}

	// Geometry raster: position-only vertex input, MIN blend over the
	// FAR clear (near-wins, order-independent), CULL_NONE like the
	// silhouette raster (windows reveal from both sides). VERT-only
	// MVP-pair push range; the fragment declares no push block.
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
	vk::PipelineRasterizationStateCreateInfo geom_rasterization{
	        .polygonMode = vk::PolygonMode::eFill,
	        .cullMode = vk::CullModeFlagBits::eNone,
	        .frontFace = vk::FrontFace::eCounterClockwise,
	        .lineWidth = 1,
	};
	vk::PipelineColorBlendAttachmentState geom_blend_attachment{
	        .blendEnable = VK_TRUE,
	        .srcColorBlendFactor = vk::BlendFactor::eOne,
	        .dstColorBlendFactor = vk::BlendFactor::eOne,
	        .colorBlendOp = vk::BlendOp::eMin,
	        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
	        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
	        .alphaBlendOp = vk::BlendOp::eMin,
	        .colorWriteMask = vk::ColorComponentFlagBits::eR,
	};
	vk::PipelineColorBlendStateCreateInfo geom_blend{
	        .attachmentCount = 1,
	        .pAttachments = &geom_blend_attachment,
	};
	vk::PushConstantRange geom_push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eVertex,
	        .offset = 0,
	        .size = sizeof(geom_push),
	};
	vk::PipelineLayoutCreateInfo geom_layout_info{
	        .setLayoutCount = 0,
	        .pSetLayouts = nullptr,
	        .pushConstantRangeCount = 1,
	        .pPushConstantRanges = &geom_push_range,
	};
	geom_layout = vk::raii::PipelineLayout(*device, geom_layout_info);
	vk::AttachmentDescription geom_attachment{
	        .format = layer_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::AttachmentReference geom_color_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription geom_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &geom_color_ref,
	};
	uint32_t geom_view_mask = 0b11, geom_corr_mask = 0b11;
	vk::RenderPassMultiviewCreateInfo geom_mv_rp_info{
	        .subpassCount = 1,
	        .pViewMasks = &geom_view_mask,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &geom_corr_mask,
	};
	vk::RenderPassCreateInfo geom_renderpass_info{
	        .pNext = &geom_mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &geom_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &geom_subpass,
	};
	geom_renderpass = vk::raii::RenderPass(*device, geom_renderpass_info);
	vk::GraphicsPipelineCreateInfo geom_pipeline_info{
	        .stageCount = 2,
	        .pStages = geom_stages,
	        .pVertexInputState = &geom_vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &geom_rasterization,
	        .pMultisampleState = &multisample,
	        .pDepthStencilState = &depth_stencil,
	        .pColorBlendState = &geom_blend,
	        .pDynamicState = &dynamic,
	        .layout = *geom_layout,
	        .renderPass = *geom_renderpass,
	};
	geom_pipeline = vk::raii::Pipeline(*device, *pipe_cache, geom_pipeline_info);

	// LOAD twin of the geom RP (same R16F attachment, LOAD/STORE):
	// loadOp is not part of render-pass compatibility, so geom_pipeline
	// binds here too (feather cutout precedent). Multiview both eyes.
	{
		vk::AttachmentDescription load_attachment{
		        .format = layer_format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eLoad,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference load_color_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription load_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &load_color_ref,
		};
		uint32_t load_view_mask = 0b11, load_corr_mask = 0b11;
		vk::RenderPassMultiviewCreateInfo load_mv_info{
		        .subpassCount = 1,
		        .pViewMasks = &load_view_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &load_corr_mask,
		};
		vk::RenderPassCreateInfo load_rp_info{
		        .pNext = &load_mv_info,
		        .attachmentCount = 1,
		        .pAttachments = &load_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &load_subpass,
		};
		geom_load_rp = vk::raii::RenderPass(*device, load_rp_info);
	}
	// Back-wall pipe (CULL_FRONT, otherwise geom-identical): volume
	// interiors resolve to the exit wall. Baked against geom_renderpass,
	// bound under the compatible geom_load_rp above.
	{
		vk::PipelineRasterizationStateCreateInfo backwall_rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eFront,
		        .frontFace = vk::FrontFace::eCounterClockwise,
		        .lineWidth = 1,
		};
		vk::GraphicsPipelineCreateInfo backwall_pipeline_info{
		        .stageCount = 2,
		        .pStages = geom_stages,
		        .pVertexInputState = &geom_vertex_input,
		        .pInputAssemblyState = &input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &backwall_rasterization,
		        .pMultisampleState = &multisample,
		        .pDepthStencilState = &depth_stencil,
		        .pColorBlendState = &geom_blend,
		        .pDynamicState = &dynamic,
		        .layout = *geom_layout,
		        .renderPass = *geom_renderpass,
		};
		backwall_pipeline = vk::raii::Pipeline(*device, *pipe_cache, backwall_pipeline_info);
	}
	// MIN-dilate fullscreen state (R16F): erode1d box-minimum taps
	// (exact texel centers, NEAREST) widen known occluders over FAR.
	// Standalone passes with barriers (not merged: H and V read
	// different images in the same command buffer, so each pass binds
	// its own ping-ponged set, merge precedent).
	{
		vk::AttachmentDescription dilate_attachment{
		        .format = layer_format,
		        .samples = vk::SampleCountFlagBits::e1,
		        .loadOp = vk::AttachmentLoadOp::eDontCare,
		        .storeOp = vk::AttachmentStoreOp::eStore,
		        .initialLayout = vk::ImageLayout::eUndefined,
		        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::AttachmentReference dilate_color_ref{
		        .attachment = 0,
		        .layout = vk::ImageLayout::eColorAttachmentOptimal,
		};
		vk::SubpassDescription dilate_subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = 1,
		        .pColorAttachments = &dilate_color_ref,
		};
		uint32_t dilate_view_mask = 0b11, dilate_corr_mask = 0b11;
		vk::RenderPassMultiviewCreateInfo dilate_mv_info{
		        .subpassCount = 1,
		        .pViewMasks = &dilate_view_mask,
		        .correlationMaskCount = 1,
		        .pCorrelationMasks = &dilate_corr_mask,
		};
		vk::RenderPassCreateInfo dilate_rp_info{
		        .pNext = &dilate_mv_info,
		        .attachmentCount = 1,
		        .pAttachments = &dilate_attachment,
		        .subpassCount = 1,
		        .pSubpasses = &dilate_subpass,
		};
		dilate_rp = vk::raii::RenderPass(*device, dilate_rp_info);
		vk::DescriptorSetLayoutBinding dilate_binding{
		        .binding = 0,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 1,
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		};
		dilate_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
		                                                              .bindingCount = 1,
		                                                              .pBindings = &dilate_binding,
		                                                      });
		vk::DescriptorPoolSize dilate_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 4,
		};
		vk::DescriptorPoolCreateInfo dilate_pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 4,
		        .poolSizeCount = 1,
		        .pPoolSizes = &dilate_pool_size,
		};
		dilate_pool = vk::raii::DescriptorPool(*device, dilate_pool_info);
		for (int i = 0; i < 4; ++i)
		{
			vk::DescriptorSetAllocateInfo dilate_set_alloc{
			        .descriptorPool = *dilate_pool,
			        .descriptorSetCount = 1,
			        .pSetLayouts = &*dilate_layout,
			};
			dilate_sets.emplace_back(std::move(device->allocateDescriptorSets(dilate_set_alloc)[0]));
		}
		struct dilate_push
		{
			float texel[2];
			float dir[2];
			float radius = 0;
			float pad = 0;
		};
		static_assert(sizeof(dilate_push) == 24);
		vk::PushConstantRange dilate_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(dilate_push),
		};
		vk::PipelineLayoutCreateInfo dilate_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*dilate_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &dilate_push_range,
		};
		dilate_pipeline_layout = vk::raii::PipelineLayout(*device, dilate_layout_info);
		auto dilate_vert = load_shader(*device, "blur.vert");
		auto dilate_frag = load_shader(*device, "erode1d_mv.frag");
		vk::PipelineShaderStageCreateInfo dilate_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **dilate_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **dilate_frag,
		                .pName = "main",
		        },
		};
		vk::GraphicsPipelineCreateInfo dilate_pipeline_info{
		        .stageCount = 2,
		        .pStages = dilate_stages,
		        .pVertexInputState = &vertex_input,
		        .pInputAssemblyState = &input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pDepthStencilState = &depth_stencil,
		        .pColorBlendState = &blend,
		        .pDynamicState = &dynamic,
		        .layout = *dilate_pipeline_layout,
		        .renderPass = *dilate_rp,
		};
		dilate_pipeline = vk::raii::Pipeline(*device, *pipe_cache, dilate_pipeline_info);
	}
	// Sensor write-back fullscreen state (R16F, MIN blend under the
	// LOAD twin so rastered windows survive): same single test as the
	// feather reveal, 4B bias push. Ping-ponged sets (async mask CB may
	// still read the other, merge precedent).
	{
		vk::DescriptorSetLayoutBinding wb_bindings[2];
		for (uint32_t i = 0; i < 2; ++i)
		{
			wb_bindings[i] = vk::DescriptorSetLayoutBinding{
			        .binding = i,
			        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			        .descriptorCount = 1,
			        .stageFlags = vk::ShaderStageFlagBits::eFragment,
			};
		}
		writeback_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
		                                                                 .bindingCount = 2,
		                                                                 .pBindings = wb_bindings,
		                                                         });
		vk::DescriptorPoolSize wb_pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 4,
		};
		vk::DescriptorPoolCreateInfo wb_pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 2,
		        .poolSizeCount = 1,
		        .pPoolSizes = &wb_pool_size,
		};
		writeback_pool = vk::raii::DescriptorPool(*device, wb_pool_info);
		for (int i = 0; i < 2; ++i)
		{
			vk::DescriptorSetAllocateInfo wb_set_alloc{
			        .descriptorPool = *writeback_pool,
			        .descriptorSetCount = 1,
			        .pSetLayouts = &*writeback_layout,
			};
			writeback_sets.emplace_back(std::move(device->allocateDescriptorSets(wb_set_alloc)[0]));
		}
		struct writeback_push
		{
			float bias = 0;
		};
		static_assert(sizeof(writeback_push) == 4);
		vk::PushConstantRange wb_push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		        .offset = 0,
		        .size = sizeof(writeback_push),
		};
		vk::PipelineLayoutCreateInfo wb_layout_info{
		        .setLayoutCount = 1,
		        .pSetLayouts = &*writeback_layout,
		        .pushConstantRangeCount = 1,
		        .pPushConstantRanges = &wb_push_range,
		};
		writeback_pipeline_layout = vk::raii::PipelineLayout(*device, wb_layout_info);
		auto wb_vert = load_shader(*device, "blur.vert");
		auto wb_frag = load_shader(*device, "depth_writeback_mv.frag");
		vk::PipelineShaderStageCreateInfo wb_stages[2] = {
		        {
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = **wb_vert,
		                .pName = "main",
		        },
		        {
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = **wb_frag,
		                .pName = "main",
		        },
		};
		vk::PipelineColorBlendAttachmentState wb_blend_attachment{
		        .blendEnable = VK_TRUE,
		        .srcColorBlendFactor = vk::BlendFactor::eOne,
		        .dstColorBlendFactor = vk::BlendFactor::eOne,
		        .colorBlendOp = vk::BlendOp::eMin,
		        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
		        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
		        .alphaBlendOp = vk::BlendOp::eMin,
		        .colorWriteMask = vk::ColorComponentFlagBits::eR,
		};
		vk::PipelineColorBlendStateCreateInfo wb_blend{
		        .attachmentCount = 1,
		        .pAttachments = &wb_blend_attachment,
		};
		vk::GraphicsPipelineCreateInfo wb_pipeline_info{
		        .stageCount = 2,
		        .pStages = wb_stages,
		        .pVertexInputState = &vertex_input,
		        .pInputAssemblyState = &input_assembly,
		        .pViewportState = &viewport_state,
		        .pRasterizationState = &rasterization,
		        .pMultisampleState = &multisample,
		        .pDepthStencilState = &depth_stencil,
		        .pColorBlendState = &wb_blend,
		        .pDynamicState = &dynamic,
		        .layout = *writeback_pipeline_layout,
		        .renderPass = *geom_load_rp,
		};
		writeback_pipeline = vk::raii::Pipeline(*device, *pipe_cache, wb_pipeline_info);
	}

	// Standalone verdict renderpass (single attachment, multiview):
	// the verdict runs ahead of the fold in its own RP (it renders at
	// a different size — opt #U half res). CLEAR-to-1 (opt #A box) +
	// STORE + SHADER_READ final (viz samples it after the frame).
	vk::AttachmentDescription verdict_attachment{
	        .format = verdict_format,
	        .samples = vk::SampleCountFlagBits::e1,
	        .loadOp = vk::AttachmentLoadOp::eClear,
	        .storeOp = vk::AttachmentStoreOp::eStore,
	        .initialLayout = vk::ImageLayout::eUndefined,
	        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	vk::AttachmentReference verdict_color_ref{
	        .attachment = 0,
	        .layout = vk::ImageLayout::eColorAttachmentOptimal,
	};
	vk::SubpassDescription verdict_subpass{
	        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
	        .colorAttachmentCount = 1,
	        .pColorAttachments = &verdict_color_ref,
	};
	uint32_t verdict_view_mask = 0b11, verdict_corr_mask = 0b11;
	vk::RenderPassMultiviewCreateInfo verdict_mv_rp_info{
	        .subpassCount = 1,
	        .pViewMasks = &verdict_view_mask,
	        .correlationMaskCount = 1,
	        .pCorrelationMasks = &verdict_corr_mask,
	};
	vk::RenderPassCreateInfo verdict_renderpass_info{
	        .pNext = &verdict_mv_rp_info,
	        .attachmentCount = 1,
	        .pAttachments = &verdict_attachment,
	        .subpassCount = 1,
	        .pSubpasses = &verdict_subpass,
	};
	verdict_rp = vk::raii::RenderPass(*device, verdict_renderpass_info);

	// Verdict + apply sampler/descriptor state (static 2-slot CIS
	// layouts for layer/virt, FRAG-only, same Adreno-safe pattern as
	// the merge): verdict reads layer NEAREST + SBS virt (handle-generation
	// sampler); apply reads merged NEAREST (exact) + half-res verdict
	// LINEAR (2x tent-filtered upscale = smooth reveal ramp).
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
	auto make_dual_cis = [&](vk::raii::DescriptorSetLayout & layout, vk::raii::DescriptorPool & pool,
	                         std::vector<vk::raii::DescriptorSet> & sets) {
		vk::DescriptorSetLayoutBinding bindings[2];
		for (uint32_t i = 0; i < 2; ++i)
		{
			bindings[i] = vk::DescriptorSetLayoutBinding{
			        .binding = i,
			        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			        .descriptorCount = 1,
			        .stageFlags = vk::ShaderStageFlagBits::eFragment,
			};
		}
		layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
		                                                              .bindingCount = 2,
		                                                              .pBindings = bindings,
		                                                      });
		vk::DescriptorPoolSize pool_size{
		        .type = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 4,
		};
		vk::DescriptorPoolCreateInfo pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 2,
		        .poolSizeCount = 1,
		        .pPoolSizes = &pool_size,
		};
		pool = vk::raii::DescriptorPool(*device, pool_info);
		for (int i = 0; i < 2; ++i)
		{
			vk::DescriptorSetAllocateInfo set_alloc{
			        .descriptorPool = *pool,
			        .descriptorSetCount = 1,
			        .pSetLayouts = &*layout,
			};
			sets.emplace_back(std::move(device->allocateDescriptorSets(set_alloc)[0]));
		}
	};
	// Verdict pool (sets/layout/pipeline built by create_verdict_state
	// below so rebuilds share one path; the pool itself is
	// format-agnostic and never changes). Sized for the 2 image
	// bindings + the foveation-table storage buffer on each set.
	{
		vk::DescriptorPoolSize pool_sizes[2] = {
		        {
		                .type = vk::DescriptorType::eCombinedImageSampler,
		                .descriptorCount = 4,
		        },
		        {
		                .type = vk::DescriptorType::eStorageBuffer,
		                .descriptorCount = 2,
		        },
		};
		vk::DescriptorPoolCreateInfo pool_info{
		        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		        .maxSets = 2,
		        .poolSizeCount = 2,
		        .pPoolSizes = pool_sizes,
		};
		verdict_descriptor_pool = vk::raii::DescriptorPool(*device, pool_info);
	}
	make_dual_cis(apply_descriptor_layout, apply_descriptor_pool, apply_descriptor_sets);

	// Initial plain verdict state (also resets verdict_ycbcr_built after
	// a swapchain rebuild, so a stale YCbCr build can never persist).
	create_verdict_state(wivrn::decoder::ycbcr_definition{}, vk::Sampler{});

	auto apply_frag = load_shader(*device, "vr_apply_direct_mv.frag");
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
	vk::PipelineLayoutCreateInfo apply_layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*apply_descriptor_layout,
	        .pushConstantRangeCount = 0,
	        .pPushConstantRanges = nullptr,
	};
	apply_layout = vk::raii::PipelineLayout(*device, apply_layout_info);
	// Fold subpass-1 pipelines (same shaders/layout as the old split
	// applies; subpass index baked). Paired: merged x verdict (the
	// verdict is sampled as a texture, SHADER_READ from its own RP).
	// Mask: merged straight through (binding 1 written but never
	// sampled — full-footprint rule).
	fold_apply_pipeline = make_fold_pipeline(apply_stages, apply_layout, fold_rp, 1);

	// Mask-only apply: same vertex stage, renderpass and layout (the
	// fragment shader samples binding 0 only; binding 1 is written
	// but never sampled on this path — the full-footprint write in
	// record_finish keeps Adreno off the partial-write fault).
	auto apply_mask_frag = load_shader(*device, "vr_apply_mask_mv.frag");
	vk::PipelineShaderStageCreateInfo apply_mask_stages[2] = {
	        {
	                .stage = vk::ShaderStageFlagBits::eVertex,
	                .module = **vert,
	                .pName = "main",
	        },
	        {
	                .stage = vk::ShaderStageFlagBits::eFragment,
	                .module = **apply_mask_frag,
	                .pName = "main",
	        },
	};
	fold_maskapply_pipeline = make_fold_pipeline(apply_mask_stages, apply_layout, fold_rp, 1);
}

bool union_merge_renderer::record(vk::raii::CommandBuffer & cmd,
                                  vk::Extent2D extent, std::span<const input> inputs)
{
	if (inputs.empty() or inputs.size() > max_inputs)
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
	// AA spike (Nov 2026): always LINEAR. The NEAREST 1:1 fast path is
	// dropped so analytic-coverage grey edge pixels filter on upscale;
	// flat interiors sample identically either way (all 4 taps agree).
	// extent (the merge size) is currently implied by the views; kept for
	// API stability with record_finish.
	(void)extent;
	vk::Sampler merge_sampler = *sampler;
	set_parity = !set_parity;
	vk::DescriptorSet set = *descriptor_sets[set_parity ? 1 : 0];
	vk::DescriptorImageInfo infos[max_inputs];
	for (uint32_t i = 0; i < max_inputs; ++i)
	{
		auto & in = inputs[std::min<size_t>(i, inputs.size() - 1)];
		infos[i] = vk::DescriptorImageInfo{
		        .sampler = merge_sampler,
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

	// No renderpass here: the merge draw happens in record_finish's
	// fold (subpass 0), which binds the set written above and pushes
	// the count below. The set parity discipline is unchanged (see
	// above).
	merge_count_ = (int)inputs.size();
	return true;
}

void union_merge_renderer::record_base(vk::raii::CommandBuffer & cmd, std::span<const geom_draw> draws,
                                        const vk::raii::QueryPool * pool, uint32_t qafter)
{
	// Pending mesh uploads first (staging→device + VertexInput
	// barrier), mirroring the feather renderer: uploads are rare
	// (map-change only), so this is a no-op most frames.
	flush_uploads(cmd);

	// Verdict content box (opt #A): both-eyes NDC union of the mesh
	// AABBs under the draw MVPs, +2px rasterization margin, clamped to
	// the layer extent. Mirrors the draw loop's skip conditions below,
	// so the box covers every pixel the draws can write; outside it
	// the layer stays CLEAR-far and reveal-1 is exact. Any
	// behind-camera corner falls back to fullscreen (a crossing
	// triangle still rasterizes) — same rule as feather's
	// pass_scissor. The fullscreen layer raster below is untouched
	// (viz samples it whole).
	{
		float minx = 1e30f, miny = 1e30f, maxx = -1e30f, maxy = -1e30f;
		bool empty = true, behind = false;
		for (const auto & d: draws)
		{
			auto mit = layer_meshes.find(d.mesh);
			if (mit == layer_meshes.end() or mit->second.index_count == 0)
				continue;
			const glm::vec3 & mn = mit->second.aabb_min;
			const glm::vec3 & mx = mit->second.aabb_max;
			for (uint32_t view = 0; view < 2 and not behind; ++view)
			{
				for (int corner = 0; corner < 8; ++corner)
				{
					glm::vec4 c((corner & 1) ? mx.x : mn.x,
					            (corner & 2) ? mx.y : mn.y,
					            (corner & 4) ? mx.z : mn.z, 1);
					glm::vec4 q = d.mvp[view] * c;
					if (q.w <= 0)
					{
						behind = true;
						break;
					}
					float nx = q.x / q.w, ny = q.y / q.w;
					// Grazing near-plane corners (tiny positive w) project
					// to huge finite NDC whose float→int cast is UB and
					// would collapse the box: treat like behind-camera
					// (fullscreen fallback). Threshold ≫ expanded frustum.
					if (not std::isfinite(nx) or not std::isfinite(ny) or
					    std::fabs(nx) > 8.f or std::fabs(ny) > 8.f)
					{
						behind = true;
						break;
					}
					minx = std::min(minx, nx);
					miny = std::min(miny, ny);
					maxx = std::max(maxx, nx);
					maxy = std::max(maxy, ny);
					empty = false;
				}
			}
			if (behind)
				break;
		}
		if (empty or behind)
		{
			verdict_box_empty_ = empty and not behind;
			verdict_box_ = behind ? vk::Rect2D{{0, 0}, half_extent_}
			                      : vk::Rect2D{{0, 0}, {0, 0}};
		}
		else
		{
			// Half-res working pixels (opt #U): the verdict loop below runs
		// at half res, so the box is mapped there directly.
		const float W = (float)half_extent_.width, H = (float)half_extent_.height;
			int32_t px0 = std::max<int32_t>(0, (int32_t)((minx * 0.5f + 0.5f) * W) - 2);
			int32_t py0 = std::max<int32_t>(0, (int32_t)((miny * 0.5f + 0.5f) * H) - 2);
			int32_t px1 = std::min<int32_t>((int32_t)half_extent_.width, (int32_t)((maxx * 0.5f + 0.5f) * W) + 2);
			int32_t py1 = std::min<int32_t>((int32_t)half_extent_.height, (int32_t)((maxy * 0.5f + 0.5f) * H) + 2);
			verdict_box_empty_ = (px1 <= px0) or (py1 <= py0);
			verdict_box_ = verdict_box_empty_ ? vk::Rect2D{{0, 0}, {0, 0}}
			                                  : vk::Rect2D{{px0, py0}, {(uint32_t)(px1 - px0), (uint32_t)(py1 - py0)}};
		}
	}

	// Even with no draws the CLEAR runs: a LOAD of undefined contents
	// would read as geometry and wrongly gate the verdict — fail-open
	// demands the explicit FAR clear.
	vk::ClearValue clear{};
	clear.color.float32 = std::array<float, 4>{layer_far, 0, 0, 0};
	vk::RenderPassBeginInfo begin{
	        .renderPass = *geom_renderpass,
	        .framebuffer = *layer_.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = half_extent_},
	        .clearValueCount = 1,
	        .pClearValues = &clear,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	cmd.setViewport(0, vk::Viewport{
	                        .x = 0,
	                        .y = 0,
	                        .width = (float)half_extent_.width,
	                        .height = (float)half_extent_.height,
	                        .minDepth = 0,
	                        .maxDepth = 1,
	                });
	// Full-layer scissor, set explicitly: prior passes in this command
	// buffer may leave narrowed boxes behind (per-draw unions). The
	// raster covers the whole (half-res) layer by design.
	cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = half_extent_});
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *geom_pipeline);
	for (const auto & d: draws)
	{
		auto mit = layer_meshes.find(d.mesh);
		if (mit == layer_meshes.end() or mit->second.index_count == 0)
			continue;
		cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
		cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
		cmd.pushConstants<geom_push>(*geom_layout, vk::ShaderStageFlagBits::eVertex, 0,
		                             geom_push{.mvp = {{d.mvp[0], d.mvp[1]}}});
		cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
	}
	cmd.endRenderPass();

	// Write→read hazard (same command buffer): the layer is sampled by
	// the verdict in record_verdict below.
	vk::ImageMemoryBarrier barrier{
	        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
	        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
	        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .image = *layer_.image,
	        .subresourceRange = {
	                .aspectMask = vk::ImageAspectFlagBits::eColor,
	                .levelCount = 1,
	                .layerCount = 2,
	        },
	};
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    vk::PipelineStageFlagBits::eFragmentShader,
	                    {},
	                    {},
	                    {},
	                    barrier);
	if (pool)
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, **pool, qafter);
}

bool union_merge_renderer::record_copy(vk::raii::CommandBuffer & cmd, std::span<const geom_draw> occluders,
                                         std::span<const geom_draw> volumes, int dilate_texels)
{
	if (not *copy_.image or not *copy_tmp_.image)
		return false;
	flush_uploads(cmd);

	auto img_barrier = [&](vk::Image img, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
	                       vk::AccessFlags src_access, vk::AccessFlags dst_access,
	                       vk::PipelineStageFlags src_stage, vk::PipelineStageFlags dst_stage) {
		cmd.pipelineBarrier(src_stage, dst_stage, {}, {}, {},
		                    vk::ImageMemoryBarrier{
		                            .srcAccessMask = src_access,
		                            .dstAccessMask = dst_access,
		                            .oldLayout = old_layout,
		                            .newLayout = new_layout,
		                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		                            .image = img,
		                            .subresourceRange = {
		                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
		                                    .levelCount = 1,
		                                    .layerCount = 2,
		                            },
		                    });
	};
	// Clone the shared layer: both images fully overwritten here, so
	// UNDEFINED discards (no first-use tracking, no tile-load cost).
	img_barrier(*layer_.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal,
	            vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead,
	            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer);
	img_barrier(*copy_.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {}, vk::AccessFlagBits::eTransferWrite,
	            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer);
	vk::ImageCopy copy_region{
	        .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 2},
	        .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 2},
	        .extent = {half_extent_.width, half_extent_.height, 1},
	};
	cmd.copyImage(*layer_.image, vk::ImageLayout::eTransferSrcOptimal, *copy_.image,
	              vk::ImageLayout::eTransferDstOptimal, copy_region);
	// The verdict samples the shared layer later in the same command
	// buffer: restore its SHADER_READ layout now; the copy moves to
	// the LOAD raster below.
	img_barrier(*layer_.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
	            vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead,
	            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader);
	img_barrier(*copy_.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal,
	            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eColorAttachmentWrite,
	            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eColorAttachmentOutput);

	auto draw_list = [&](std::span<const geom_draw> draws, const std::map<std::string, mesh_buffers> & meshes,
	                     vk::Pipeline pipe) {
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe);
		for (const auto & d: draws)
		{
			auto mit = meshes.find(d.mesh);
			if (mit == meshes.end() or mit->second.index_count == 0)
				continue;
			cmd.bindVertexBuffers(0, (vk::Buffer)*mit->second.vertex_buffer, (vk::DeviceSize)0);
			cmd.bindIndexBuffer(*mit->second.index_buffer, 0, vk::IndexType::eUint32);
			cmd.pushConstants<geom_push>(*geom_layout, vk::ShaderStageFlagBits::eVertex, 0,
			                             geom_push{.mvp = {{d.mvp[0], d.mvp[1]}}});
			cmd.drawIndexed(mit->second.index_count, 1, 0, 0, 0);
		}
	};
	// Occluders (hidden, all placed) onto the clone: LOAD preserves the
	// windows, MIN keeps the nearer of window/occluder per pixel.
	if (not occluders.empty())
	{
		vk::RenderPassBeginInfo occl_begin{
		        .renderPass = *geom_load_rp,
		        .framebuffer = *copy_load_fb,
		        .renderArea = {.offset = {0, 0}, .extent = half_extent_},
		        .clearValueCount = 0,
		        .pClearValues = nullptr,
		};
		cmd.beginRenderPass(occl_begin, vk::SubpassContents::eInline);
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)half_extent_.width,
		                        .height = (float)half_extent_.height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = half_extent_});
		draw_list(occluders, occluder_meshes, *geom_pipeline);
		cmd.endRenderPass();
		img_barrier(*copy_.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		            vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
		            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader);
	}
	if (occluders.empty())
	{
		img_barrier(*copy_.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		            vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
		            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader);
	}
	// MIN-dilate the known occluders (H then V): nearer bleeds over
	// FAR, widening the exclusion by dilate_texels. The volume back
	// wall below stays exact (rastered after).
	int radius = std::clamp(dilate_texels, 0, 8);
	if (radius > 0)
	{
		struct dilate_push
		{
			float texel[2];
			float dir[2];
			float radius = 0;
			float pad = 0;
		};
		static_assert(sizeof(dilate_push) == 24);
		auto dilate_pass = [&](bool horizontal) {
			vk::DescriptorSet set = *dilate_sets[(dilate_parity ? 2 : 0) + (horizontal ? 0 : 1)];
			vk::ImageView src_view = horizontal ? *copy_.view_array : *copy_tmp_.view_array;
			vk::DescriptorImageInfo src_info{
			        .sampler = *sampler_nearest,
			        .imageView = src_view,
			        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			};
			std::array<vk::WriteDescriptorSet, 1> dil_writes = {{
			        {
			                .dstSet = set,
			                .dstBinding = 0,
			                .descriptorCount = 1,
			                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
			                .pImageInfo = &src_info,
			        },
			}};
			device->updateDescriptorSets(dil_writes, {});
			vk::Framebuffer fb = horizontal ? *copy_tmp_dilate_fb : *copy_dilate_fb;
			vk::RenderPassBeginInfo dil_begin{
			        .renderPass = *dilate_rp,
			        .framebuffer = fb,
			        .renderArea = {.offset = {0, 0}, .extent = half_extent_},
			        .clearValueCount = 0,
			        .pClearValues = nullptr,
			};
			cmd.beginRenderPass(dil_begin, vk::SubpassContents::eInline);
			cmd.setViewport(0, vk::Viewport{
			                        .x = 0,
			                        .y = 0,
			                        .width = (float)half_extent_.width,
			                        .height = (float)half_extent_.height,
			                        .minDepth = 0,
			                        .maxDepth = 1,
			                });
			cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = half_extent_});
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *dilate_pipeline);
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *dilate_pipeline_layout, 0, set, {});
			dilate_push pc{
			        .texel = {(float)half_extent_.width ? 1.f / (float)half_extent_.width : 0.f,
			                  (float)half_extent_.height ? 1.f / (float)half_extent_.height : 0.f},
			        .dir = {horizontal ? 1.f : 0.f, horizontal ? 0.f : 1.f},
			        .radius = (float)radius,
			};
			cmd.pushConstants<dilate_push>(*dilate_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, pc);
			cmd.draw(3, 1, 0, 0);
			cmd.endRenderPass();
		};
		dilate_parity = !dilate_parity;
		img_barrier(*copy_tmp_.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
		            vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eTopOfPipe,
		            vk::PipelineStageFlagBits::eColorAttachmentOutput);
		dilate_pass(true);
		img_barrier(*copy_tmp_.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		            vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
		            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader);
		img_barrier(*copy_.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal,
		            vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eColorAttachmentWrite,
		            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eColorAttachmentOutput);
		dilate_pass(false);
		img_barrier(*copy_.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		            vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
		            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader);
	}
	// Volume back walls (CULL_FRONT exit wall): LOAD preserves the
	// dilated occluders, MIN keeps nearer of occluder/back-wall.
	if (not volumes.empty())
	{
		img_barrier(*copy_.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eColorAttachmentOptimal,
		            vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eColorAttachmentWrite,
		            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eColorAttachmentOutput);
		vk::RenderPassBeginInfo vol_begin{
		        .renderPass = *geom_load_rp,
		        .framebuffer = *copy_load_fb,
		        .renderArea = {.offset = {0, 0}, .extent = half_extent_},
		        .clearValueCount = 0,
		        .pClearValues = nullptr,
		};
		cmd.beginRenderPass(vol_begin, vk::SubpassContents::eInline);
		cmd.setViewport(0, vk::Viewport{
		                        .x = 0,
		                        .y = 0,
		                        .width = (float)half_extent_.width,
		                        .height = (float)half_extent_.height,
		                        .minDepth = 0,
		                        .maxDepth = 1,
		                });
		cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = half_extent_});
		draw_list(volumes, volume_meshes, *backwall_pipeline);
		cmd.endRenderPass();
	}
	// Copy ends sampled: barrier only when the last writer left it as
	// an attachment (volume raster, or the untouched clone when both
	// lists were empty and no dilate ran). Otherwise it already ends
	// SHADER_READ (occluder raster or dilate V).
	if (not volumes.empty() or (occluders.empty() and radius == 0))
		img_barrier(*copy_.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		            vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
		            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader);
	return true;
}

bool union_merge_renderer::record_depth_writeback(vk::raii::CommandBuffer & cmd, vk::ImageView sensor_view,
                                                   vk::ImageView copy_view, float bias)
{
	if (not *layer_.image or sensor_view == vk::ImageView{} or copy_view == vk::ImageView{})
		return false;

	// The shared layer ends SHADER_READ (record_base barrier; feather
	// records never touch it): transition to the LOAD raster, run the
	// test fullscreen (MIN keeps the nearer of sensor and window), and
	// hand back SHADER_READ for the verdict tap below.
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
	                            .image = *layer_.image,
	                            .subresourceRange = {
	                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                    .levelCount = 1,
	                                    .layerCount = 2,
	                            },
	                    });
	writeback_parity = !writeback_parity;
	vk::DescriptorSet wset = *writeback_sets[writeback_parity ? 1 : 0];
	vk::DescriptorImageInfo sensor_info{
	        .sampler = *sampler_nearest,
	        .imageView = sensor_view,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	vk::DescriptorImageInfo copy_info{
	        .sampler = *sampler_nearest,
	        .imageView = copy_view,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	// Full footprint every record (Adreno partial-write rule).
	std::array<vk::WriteDescriptorSet, 2> wb_writes = {{
	        {
	                .dstSet = wset,
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &sensor_info,
	        },
	        {
	                .dstSet = wset,
	                .dstBinding = 1,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &copy_info,
	        },
	}};
	device->updateDescriptorSets(wb_writes, {});
	vk::RenderPassBeginInfo wb_begin{
	        .renderPass = *geom_load_rp,
	        .framebuffer = *layer_load_fb,
	        .renderArea = {.offset = {0, 0}, .extent = half_extent_},
	        .clearValueCount = 0,
	        .pClearValues = nullptr,
	};
	cmd.beginRenderPass(wb_begin, vk::SubpassContents::eInline);
	cmd.setViewport(0, vk::Viewport{
	                        .x = 0,
	                        .y = 0,
	                        .width = (float)half_extent_.width,
	                        .height = (float)half_extent_.height,
	                        .minDepth = 0,
	                        .maxDepth = 1,
	                });
	// Full-extent scissor, set explicitly: prior passes in this command
	// buffer leave narrowed boxes behind (verdict content box, per-draw
	// unions) — inheriting one would clip the write-back with no
	// validation complaint.
	cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = half_extent_});
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *writeback_pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *writeback_pipeline_layout, 0, wset, {});
	struct writeback_push
	{
		float bias = 0;
	};
	cmd.pushConstants<writeback_push>(*writeback_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0,
	                                  writeback_push{.bias = bias});
	cmd.draw(3, 1, 0, 0);
	cmd.endRenderPass();
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
	                            .image = *layer_.image,
	                            .subresourceRange = {
	                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
	                                    .levelCount = 1,
	                                    .layerCount = 2,
	                            },
	                    });
	return true;
}

void union_merge_renderer::create_verdict_state(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler)
{
	// Full teardown in dependency order: sets reference pool + set
	// layout, pipeline references pipeline layout, pipeline layout
	// references the set layout. Caller guarantees no in-flight readers
	// (fresh object, or a waitIdle'd rebuild).
	verdict_descriptor_sets.clear();
	verdict_pipeline.clear();
	verdict_layout.clear();
	verdict_descriptor_layout.clear();
	// Bake the paired handle's generation sampler as the immutable
	// sampler: a separately built conversion proved unidentifiable to
	// validation despite matching parameters, so the layout shares the
	// handle's object instead of owning one. verdict_immutable_src
	// tracks what the layout was baked with; a generation change
	// rebuilds before the next bind.
	verdict_immutable_src = def.has_conversion ? generation_sampler : vk::Sampler{};
	// Binding 0 (common layer, plain view) stays mutable; binding 1
	// (virt) is immutable exactly when the view carries a conversion.
	// Binding 2 (foveation index maps) is a plain storage buffer,
	// rewritten per verdict frame. The immutable handle is copied by
	// layout creation (defoveator precedent: pointer to a local is fine).
	vk::Sampler immutable = verdict_immutable_src;
	vk::DescriptorSetLayoutBinding bindings[3] = {
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
	                .pImmutableSamplers = def.has_conversion ? &immutable : nullptr,
	        },
	        {
	                .binding = 2,
	                .descriptorType = vk::DescriptorType::eStorageBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eFragment,
	        },
	};
	verdict_descriptor_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                                 .bindingCount = 3,
	                                                                                 .pBindings = bindings,
	                                                                         });
	for (int i = 0; i < 2; ++i)
	{
		vk::DescriptorSetAllocateInfo set_alloc{
		        .descriptorPool = *verdict_descriptor_pool,
		        .descriptorSetCount = 1,
		        .pSetLayouts = &*verdict_descriptor_layout,
		};
		verdict_descriptor_sets.emplace_back(std::move(device->allocateDescriptorSets(set_alloc)[0]));
	}
	// Pipeline bakes the set layout: rebuilt alongside it, never apart
	// (binding fresh sets with a stale pipeline is a use error). The
	// YCbCr rebuild path only ever touches this pipeline (standalone
	// verdict RP, subpass 0).
	vk::PipelineLayoutCreateInfo verdict_layout_info{
	        .setLayoutCount = 1,
	        .pSetLayouts = &*verdict_descriptor_layout,
	        .pushConstantRangeCount = 0,
	        .pPushConstantRanges = nullptr,
	};
	verdict_layout = vk::raii::PipelineLayout(*device, verdict_layout_info);
	auto vert = load_shader(*device, "blur.vert");
	auto verdict_frag = load_shader(*device, "vr_verdict_mv.frag");
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
	// Static fullscreen state, identical to the shared init block.
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
	vk::GraphicsPipelineCreateInfo verdict_pipeline_info{
	        .stageCount = 2,
	        .pStages = verdict_stages,
	        .pVertexInputState = &vertex_input,
	        .pInputAssemblyState = &input_assembly,
	        .pViewportState = &viewport_state,
	        .pRasterizationState = &rasterization,
	        .pMultisampleState = &multisample,
	        .pDepthStencilState = &depth_stencil,
	        .pColorBlendState = &blend,
	        .pDynamicState = &dynamic,
	        .layout = *verdict_layout,
	        .renderPass = *verdict_rp,
	        .subpass = 0,
	};
	verdict_pipeline = vk::raii::Pipeline(*device, *pipe_cache, verdict_pipeline_info);
	verdict_ycbcr_built = def;
}

void union_merge_renderer::ensure_verdict_layout(const wivrn::decoder::ycbcr_definition & def, vk::Sampler generation_sampler)
{
	// Rebuild when the definition changes OR the handle sampler is a
	// new generation (renegotiation hands out a new object: the baked
	// immutable handle must track the sampled generation). Plain
	// layouts don't bake a sampler, so only converted state compares
	// handles.
	if (verdict_ycbcr_built == def and (not def.has_conversion or verdict_immutable_src == generation_sampler))
		return;
	// Depth format change (rare: renegotiation only): retire prior
	// submissions (including the verdict's own previous frame) before
	// rebuilding what they may still read.
	device->waitIdle();
	create_verdict_state(def, generation_sampler);
	spdlog::info("Union merge verdict layout rebuilt (YCbCr {}, handle-generation immutable sampler)", def.has_conversion ? "immutable" : "plain");
}

void union_merge_renderer::record_verdict(vk::raii::CommandBuffer & cmd,
                                          const virt_input & virt_in,
                                          const vk::raii::QueryPool * pool, uint32_t qafter)
{
	if (not virt_in.valid or not virt_in.view or not virt_in.sampler)
		return;

	// Own single-attachment verdict RP (R8, CLEAR-to-1 + box draw):
	// the verdict cannot share the fold (different size) and must
	// precede it. CLEAR runs at begin; the s1 apply below samples the
	// result as a texture.
	vk::ClearValue verdict_clear{};
	verdict_clear.color.float32 = std::array<float, 4>{1, 1, 1, 1};
	vk::RenderPassBeginInfo begin{
	        .renderPass = *verdict_rp,
	        .framebuffer = *verdict_.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = half_extent_},
	        .clearValueCount = 1,
	        .pClearValues = &verdict_clear,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	cmd.setViewport(0, vk::Viewport{
	                        .x = 0,
	                        .y = 0,
	                        .width = (float)half_extent_.width,
	                        .height = (float)half_extent_.height,
	                        .minDepth = 0,
	                        .maxDepth = 1,
	                });
	// Full-layer scissor default; narrowed to the content box for the
	// draw when non-empty (opt #A, box already in half pixels).
	cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = half_extent_});
	// Content-box verdict: draw only the box — outside it the CLEAR
	// above already wrote reveal-1, exact. Empty box skips the update
	// + draw entirely (SSBO expand included); the timestamp below
	// still stamps so the split poll stays valid.
	if (not verdict_box_empty_)
	{
		cmd.setScissor(0, verdict_box_);
		// YCbCr immutable-sampler layout (VUID-02738): rebuild when the
		// view's conversion definition or the handle's generation
		// sampler changes. waitIdle'd, rare (own RP here, nothing bound
		// yet — pipelines aren't in flight until bound in a submitted
		// command buffer).
		ensure_verdict_layout(virt_in.ycbcr, virt_in.sampler);
		verdict_set_parity = !verdict_set_parity;
		// Foveation index maps, CPU-expanded per frame into the
		// ping-ponged SSBO (depth_overlay UBO discipline: memcpy +
		// flush, rewritten before the draw that reads them). Merge->render
		// scale baked in; packed dims from the paired depth frame.
		auto & tables = verdict_tables[verdict_set_parity ? 1 : 0];
		{
			const int W = (int)half_extent_.width, H = (int)half_extent_.height;
			const int render_w = virt_in.render_w, render_h = virt_in.render_h;
			const int packed_w = virt_in.packed_w, packed_h = virt_in.packed_h;
			assert(W > 0 and H > 0);
			verdict_map_scratch.resize((size_t)2 * (W + H));
			uint32_t * map = verdict_map_scratch.data();
			for (int eye = 0; eye < 2; ++eye)
			{
				// Server invariant: compute_params always emits odd,
				// non-empty runs (fill_param_2d ends count*2-1; the
				// non-foveated fallback emits one element).
				assert(virt_in.foveation[eye].x.size() % 2 == 1);
				assert(virt_in.foveation[eye].y.size() % 2 == 1);
				expand_foveation_axis(virt_in.foveation[eye].x, render_w, W, packed_w, map + eye * W);
				expand_foveation_axis(virt_in.foveation[eye].y, render_h, H, packed_h, map + 2 * W + eye * H);
			}
			std::memcpy(tables.map(), map, verdict_map_scratch.size() * sizeof(uint32_t));
			tables.unmap();
			// Non-coherent host-visible memory: flush, or the GPU reads
			// stale tables forever (depth_overlay precedent).
			vmaFlushAllocation(vk_allocator::instance(), tables, 0, verdict_map_scratch.size() * sizeof(uint32_t));
		}
		vk::DescriptorSet vset = *verdict_descriptor_sets[verdict_set_parity ? 1 : 0];
		vk::DescriptorImageInfo layer_info{
		        .sampler = *sampler_nearest,
		        .imageView = *layer_.view_array,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
		// Decoder SBS image: GENERAL layout at frame top (same
		// steady-state discipline as the video path). The write sampler
		// is ignored when the layout baked an immutable one; pass the
		// handle's generation sampler either way.
		vk::DescriptorImageInfo virt_info{
		        .sampler = virt_in.sampler,
		        .imageView = virt_in.view,
		        .imageLayout = vk::ImageLayout::eGeneral,
		};
		vk::DescriptorBufferInfo tables_info{
		        .buffer = vk::Buffer(tables),
		        .offset = 0,
		        .range = verdict_tables_bytes,
		};
		std::vector<vk::WriteDescriptorSet> vwrites;
		vwrites.reserve(3);
		vwrites.push_back(vk::WriteDescriptorSet{
		        .dstSet = vset,
		        .dstBinding = 0,
		        .descriptorCount = 1,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .pImageInfo = &layer_info,
		});
		vwrites.push_back(vk::WriteDescriptorSet{
		        .dstSet = vset,
		        .dstBinding = 1,
		        .descriptorCount = 1,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .pImageInfo = &virt_info,
		});
		vwrites.push_back(vk::WriteDescriptorSet{
		        .dstSet = vset,
		        .dstBinding = 2,
		        .descriptorCount = 1,
		        .descriptorType = vk::DescriptorType::eStorageBuffer,
		        .pBufferInfo = &tables_info,
		});
		device->updateDescriptorSets(vwrites, {});
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *verdict_pipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *verdict_layout, 0, vset, {});
		cmd.draw(3, 1, 0, 0);
	} // else: empty box — verdict stays CLEAR(1), exact for all-absent.
	cmd.endRenderPass();

	// Write→read hazard (same command buffer): the apply tap in the
	// fold below samples this image SHADER_READ. The RP already
	// transitioned the layout; this covers the access.
	vk::ImageMemoryBarrier verdict_barrier{
	        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
	        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
	        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	        .image = *verdict_.image,
	        .subresourceRange = {
	                .aspectMask = vk::ImageAspectFlagBits::eColor,
	                .levelCount = 1,
	                .layerCount = 2,
	        },
	};
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                    vk::PipelineStageFlagBits::eFragmentShader,
	                    {},
	                    {},
	                    {},
	                    verdict_barrier);
	if (pool)
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, **pool, qafter);
}

bool union_merge_renderer::record_finish(vk::raii::CommandBuffer & cmd,
                                         size_t image_index,
                                         vk::Extent2D extent,
                                         const virt_input & virt_in,
                                         const vk::raii::QueryPool * pool, uint32_t qafter_merge)
{
	if (image_index >= targets.size())
		return false;

	// Single merge size: the submit image and the owned merged target
	// share it (ensure() rebuilds them together); layer and verdict
	// have their own extents now. The apply below reads extent.
	assert(extent.width == layer_extent_.width and extent.height == layer_extent_.height);

	// Folded merge→apply: merge s0 (MAX into the owned tile-local
	// attachment), then apply s1 below. The verdict ran ahead in its
	// own RP when paired (record_verdict); mask-only frames take the
	// mask pipeline on the same fold.
	bool do_verdict = virt_in.valid and virt_in.view and virt_in.sampler;
	auto & fb = fold_fbs[image_index];
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
	// validation complaint. Persists into s1 below.
	cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extent});
	// DONT_CARE loads: both fullscreen triangles below overwrite every
	// pixel in the render area.
	vk::RenderPassBeginInfo begin{
	        .renderPass = *fold_rp,
	        .framebuffer = *fb,
	        .renderArea = {.offset = {0, 0}, .extent = extent},
	        .clearValueCount = 0,
	        .pClearValues = nullptr,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	// Merge set written by record(): same parity expression, nothing
	// else flips it in between (same-thread sequential use).
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *fold_merge_pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, *descriptor_sets[set_parity ? 1 : 0], {});
	// Active input count for the uniform tap early-out: identical for
	// every invocation in the draw, so skipped taps never execute.
	cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{.count = merge_count_});
	// Fullscreen triangle (blur.vert: 3 verts, no vertex buffers).
	cmd.draw(3, 1, 0, 0);
	// No endRenderPass here: the apply subpass below samples merged
	// tile-local (subpass dependency covers the write→read hazard).
	if (pool)
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, **pool, qafter_merge);
	cmd.nextSubpass(vk::SubpassContents::eInline);

	// Apply pass into the acquired submit image. Verdict path:
	// merged (NEAREST, merge-res) x verdict (LINEAR 2x upscale of the
	// half-merge-res mask: smooth reveal ramp). Mask-only path (no verdict
	// pass above): merged straight
	// through (one tap, pixel-identical to x1). RGB zero; the
	// submit-time REPLACE preserves dst color.
	apply_set_parity = !apply_set_parity;
	vk::DescriptorSet aset = *apply_descriptor_sets[apply_set_parity ? 1 : 0];
	vk::DescriptorImageInfo merged_info{
	        .sampler = *sampler_nearest,
	        .imageView = *merged_.view_array,
	        // GENERAL, not SHADER_READ: the folded attachments stay
	        // GENERAL throughout the RP (triple precedent: sampled
	        // across subpasses, no transition). Declaring SHADER_READ
	        // here would mismatch the actual layout at draw time.
	        .imageLayout = vk::ImageLayout::eGeneral,
	};
	std::vector<vk::WriteDescriptorSet> apply_writes;
	apply_writes.reserve(2);
	apply_writes.push_back(vk::WriteDescriptorSet{
	        .dstSet = aset,
	        .dstBinding = 0,
	        .descriptorCount = 1,
	        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	        .pImageInfo = &merged_info,
	});
	// Full footprint on every update, paired or mask-only: binding 1
	// carries the verdict view even when the bound pipeline never
	// samples it (mask pipeline declares binding 0 only; the layout
	// still declares both, so the write is valid). Partial writes
	// across frames (binding 0 alone, then 0+1 on the first paired
	// frame) fault Adreno inside updateDescriptorSets on release
	// (validation-silent; suspected first-touch bookkeeping for the
	// newly-written binding) — writing both bindings from the start
	// avoids that path entirely.
	// Verdict tap filter (soft edge): LINEAR, not NEAREST. The
	// half-res verdict tap IS the 2x upscale, and a tent filter turns
	// the 2px stairstep (plus any 1px upscale-classification fringe)
	// into a smooth ~2px reveal ramp instead of crawling binary edges.
	// This interpolates the post-decision binary mask (pure alpha
	// blending), never depths: no halo possible. Flat areas are
	// bit-identical (all taps agree). The sampler is the long-lived
	// merge sampler; the full-footprint write pattern is unchanged.
	vk::DescriptorImageInfo apply_verdict_info{
	        .sampler = *sampler,
	        .imageView = *verdict_.view_array,
	        // SHADER_READ: the verdict is written by its own RP ahead
	        // of the fold (ends SHADER_READ + access barrier in
	        // record_verdict) and sampled here as a texture. (Mask-only
	        // frames write this binding but never sample it — the mask
	        // shader declares binding 0 only — so the layout is moot
	        // there.)
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	apply_writes.push_back(vk::WriteDescriptorSet{
	        .dstSet = aset,
	        .dstBinding = 1,
	        .descriptorCount = 1,
	        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	        .pImageInfo = &apply_verdict_info,
	});
	device->updateDescriptorSets(apply_writes, {});
	// Apply subpass s1: merged (tile-local GENERAL from s0 above) x
	// verdict (SHADER_READ texture from its own RP), or merged straight
	// through on the mask path. Merged stays NEAREST (exact alpha);
	// the verdict LINEAR tap is the upscale. Fullscreen scissor
	// inherited from s0.
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, do_verdict ? *fold_apply_pipeline : *fold_maskapply_pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *apply_layout, 0, aset, {});
	cmd.draw(3, 1, 0, 0);
	cmd.endRenderPass();

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
	return true;
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> union_merge_renderer::make_buffer(vk::DeviceSize size,
                                                                                      vk::BufferUsageFlags usage,
                                                                                      vk::MemoryPropertyFlags properties)
{
	vk::BufferCreateInfo buffer_info{
	        .size = size,
	        .usage = usage,
	        .sharingMode = vk::SharingMode::eExclusive,
	};
	vk::raii::Buffer buffer(*device, buffer_info);
	auto requirements = buffer.getMemoryRequirements();
	vk::MemoryAllocateInfo alloc_info{
	        .allocationSize = requirements.size,
	        .memoryTypeIndex = find_memory_type(*physical_device, requirements.memoryTypeBits, properties),
	};
	vk::raii::DeviceMemory memory(*device, alloc_info);
	buffer.bindMemory(*memory, 0);
	return {std::move(buffer), std::move(memory)};
}

void union_merge_renderer::flush_uploads(vk::raii::CommandBuffer & cmd)
{
	auto flush_map = [&](std::map<std::string, mesh_buffers> & meshes) {
	for (auto & [key, mesh]: meshes)
	{
		(void)key;
		if (not mesh.pending_upload)
			continue;
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
	};
	flush_map(layer_meshes);
	flush_map(occluder_meshes);
	flush_map(volume_meshes);
}

void union_merge_renderer::set_mesh_in(std::map<std::string, mesh_buffers> & meshes, const std::string & key,
                                       const passthrough_mesh::triangle_soup & soup)
{
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

void union_merge_renderer::set_layer_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup)
{
	if (soup.indices.empty() or soup.vertices.empty())
	{
		remove_layer_mesh(key);
		return;
	}
	// Rare path (map change): make sure no in-flight frame still reads
	// the old buffers before replacing them.
	device->waitIdle();
	set_mesh_in(layer_meshes, key, soup);
}

void union_merge_renderer::remove_layer_mesh(const std::string & key)
{
	auto it = layer_meshes.find(key);
	if (it == layer_meshes.end())
		return;
	// Same in-flight hazard as replace.
	device->waitIdle();
	layer_meshes.erase(it);
}

std::optional<std::pair<glm::vec3, glm::vec3>> union_merge_renderer::layer_mesh_bounds(const std::string & key) const
{
	auto it = layer_meshes.find(key);
	if (it == layer_meshes.end())
		return std::nullopt;
	return std::make_pair(it->second.aabb_min, it->second.aabb_max);
}

void union_merge_renderer::set_occluder_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup)
{
	if (soup.indices.empty() or soup.vertices.empty())
	{
		remove_occluder_mesh(key);
		return;
	}
	device->waitIdle();
	set_mesh_in(occluder_meshes, key, soup);
}

void union_merge_renderer::remove_occluder_mesh(const std::string & key)
{
	auto it = occluder_meshes.find(key);
	if (it == occluder_meshes.end())
		return;
	device->waitIdle();
	occluder_meshes.erase(it);
}

std::optional<std::pair<glm::vec3, glm::vec3>> union_merge_renderer::occluder_mesh_bounds(const std::string & key) const
{
	auto it = occluder_meshes.find(key);
	if (it == occluder_meshes.end())
		return std::nullopt;
	return std::make_pair(it->second.aabb_min, it->second.aabb_max);
}

void union_merge_renderer::set_volume_mesh(const std::string & key, const passthrough_mesh::triangle_soup & soup)
{
	if (soup.indices.empty() or soup.vertices.empty())
	{
		remove_volume_mesh(key);
		return;
	}
	device->waitIdle();
	set_mesh_in(volume_meshes, key, soup);
}

void union_merge_renderer::remove_volume_mesh(const std::string & key)
{
	auto it = volume_meshes.find(key);
	if (it == volume_meshes.end())
		return;
	device->waitIdle();
	volume_meshes.erase(it);
}

std::optional<std::pair<glm::vec3, glm::vec3>> union_merge_renderer::volume_mesh_bounds(const std::string & key) const
{
	auto it = volume_meshes.find(key);
	if (it == volume_meshes.end())
		return std::nullopt;
	return std::make_pair(it->second.aabb_min, it->second.aabb_max);
}
