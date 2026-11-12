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
#include <cstdint>
#include <vector>

#include <spdlog/spdlog.h>

namespace
{
// Mirrors union_merge_mv.frag.glsl Push: active mask input count
// (1..8) for the uniform tap early-out plus the dots-sidecar present
// flag. 8B, far below the 128-byte Vulkan minimum (no
// maxPushConstantsSize check needed); FRAG-only range, and the vertex
// shader declares no push block, so the mask_mv offset trap
// (mismatched vert/frag blocks under a shared range) is structurally
// impossible.
struct push
{
	int32_t count = 0;
	int32_t use_dots = 0;
};
static_assert(sizeof(push) == 8);
} // namespace

union_merge_renderer::union_merge_renderer(vk::raii::Device & device, vk::raii::PhysicalDevice & physical_device) :
        device(&device),
        physical_device(&physical_device)
{
}

std::shared_ptr<vk::raii::PipelineCache> union_merge_renderer::pipeline_cache(vk::raii::Device & device)
{
	// Same process-wide object the feather mask and dots renderers
	// share (one entry per VkDevice).
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
	// (destroys the use of) any sets still held.
	descriptor_sets.clear();
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
	// 9 static sampler2DArray bindings, fragment-only (bindings 0..7
	// mask inputs, binding 8 dots sidecar). No vertex input, no push
	// constants — the reduction is count-independent.
	static constexpr uint32_t binding_count = max_inputs + 1;
	vk::DescriptorSetLayoutBinding bindings[binding_count];
	for (uint32_t i = 0; i < binding_count; ++i)
	{
		bindings[i] = vk::DescriptorSetLayoutBinding{
		        .binding = i,
		        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 1,
		        .stageFlags = vk::ShaderStageFlagBits::eFragment,
		};
	}
	descriptor_layout = vk::raii::DescriptorSetLayout(*device, vk::DescriptorSetLayoutCreateInfo{
	                                                                 .bindingCount = binding_count,
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
	        .descriptorCount = 2 * (max_inputs + 1),
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
	// No blending: a single write per pixel (MAX mask alpha holed by
	// dots coverage, premultiplied dots RGB); the submit-time blend
	// does color ADDITIVE + alpha REPLACE (ONE, ONE / ONE, ZERO).
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
}

bool union_merge_renderer::record(vk::raii::CommandBuffer & cmd, size_t image_index,
                                  vk::Extent2D extent, std::span<const input> inputs,
                                  const input * dots)
{
	if (inputs.empty() or inputs.size() > max_inputs or image_index >= targets.size())
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
	// Dots sidecar (same borrowed lifetime, same-frame use only): a
	// null view with a non-null pointer is the same retired-image
	// hazard as above — refuse the whole merge, never a partial one.
	// Null pointer = masks only (valid: no dots recorded this frame).
	bool use_dots = dots != nullptr;
	if (use_dots and dots->view_array == vk::ImageView{})
		return false;

	// Write→read hazard (same command buffer): every input was just
	// written as a color attachment and ends OPTIMAL. Transition to
	// SHADER_READ for the merge; every next record re-begins
	// UNDEFINED+CLEAR, so no transition back is needed.
	std::vector<vk::ImageMemoryBarrier> barriers;
	barriers.reserve(inputs.size() + (use_dots ? 1 : 0));
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
	if (use_dots)
	{
		barriers.push_back(vk::ImageMemoryBarrier{
		        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
		        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
		        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		        .image = dots->image,
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
	// mask slots re-bind input 0's view (same view in several slots
	// is legal; max() over duplicates is neutral). Binding 8 is the
	// dots sidecar (input 0 re-bound when inactive; the shader only
	// taps it when use_dots is set). Ping-pong parity: the other set
	// may still be read by a prior in-flight command buffer (inline
	// fallback while the async CB flies).
	set_parity = !set_parity;
	vk::DescriptorSet set = *descriptor_sets[set_parity ? 1 : 0];
	vk::DescriptorImageInfo infos[max_inputs + 1];
	for (uint32_t i = 0; i < max_inputs; ++i)
	{
		auto & in = inputs[std::min<size_t>(i, inputs.size() - 1)];
		infos[i] = vk::DescriptorImageInfo{
		        .sampler = *sampler,
		        .imageView = in.view_array,
		        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};
	}
	infos[max_inputs] = vk::DescriptorImageInfo{
	        .sampler = *sampler,
	        .imageView = use_dots ? dots->view_array : inputs[0].view_array,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	std::vector<vk::WriteDescriptorSet> writes;
	writes.reserve(max_inputs + 1);
	for (uint32_t i = 0; i < max_inputs + 1; ++i)
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

	auto & tgt = targets[image_index];
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
	        .framebuffer = *tgt.framebuffer,
	        .renderArea = {.offset = {0, 0}, .extent = extent},
	        .clearValueCount = 1,
	        .pClearValues = &clear,
	};
	cmd.beginRenderPass(begin, vk::SubpassContents::eInline);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, set, {});
	// Active mask count + dots flag for the uniform tap early-out
	// (volume_virt FRAG-only precedent): identical for every
	// invocation in the draw, so skipped taps never execute.
	// inputs.size() is 1..8 (checked on entry).
	cmd.pushConstants<push>(*pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, push{.count = (int32_t)inputs.size(), .use_dots = use_dots ? 1 : 0});
	// Fullscreen triangle (blur.vert: 3 verts, no vertex buffers).
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
