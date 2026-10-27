/*
 * WiVRn VR streaming
 * Copyright (C) 2026  Patrick Nicolas <patricknicolas@laposte.net>
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
 * along with this program.  If not, see <https://www.gnu.org/>.
 */

#include "depth_reprojector.h"

#include "utils/wivrn_vk_bundle.h"

#include "math/m_space.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>

namespace
{
// GL-style asymmetric frustum projection matching a standard 0..1
// nonlinear depth buffer. Column-major (GLSL mat4 order):
// v[col * 4 + row]. Returns false on degenerate input (caller falls
// back: no warp recorded).
bool frustum_gl(const xrt_fov & fov, float near_z, float far_z, xrt_matrix_4x4 & out)
{
	// GL internal convention (NDC z in [-1,1], remapped *0.5+0.5 for
	// storage) reproduces exactly the depth values a Vulkan app stores:
	// Vulkan's [0,1] NDC was defined to coincide with the remapped GL
	// values (both are f/(f-n) - nf/((f-n)*d)). So this matrix matches
	// Vulkan-rendered depth buffers; no convention conversion needed.
	if (not(std::isfinite(near_z) and std::isfinite(far_z)) or near_z <= 0 or far_z <= near_z)
		return false;
	float l = near_z * tanf(fov.angle_left);
	float r = near_z * tanf(fov.angle_right);
	float b = near_z * tanf(fov.angle_down);
	float t = near_z * tanf(fov.angle_up);
	if (not(std::isfinite(l) and std::isfinite(r) and std::isfinite(b) and std::isfinite(t)) or not(r > l and t > b))
		return false;
	float rl = 2 * near_z / (r - l);
	float tb = 2 * near_z / (t - b);
	float rl_mid = (r + l) / (r - l);
	float tb_mid = (t + b) / (t - b);
	float fn = -(far_z + near_z) / (far_z - near_z);
	float fn2 = -2 * far_z * near_z / (far_z - near_z);
	// Column 0
	out.v[0] = rl;
	out.v[1] = 0;
	out.v[2] = 0;
	out.v[3] = 0;
	// Column 1
	out.v[4] = 0;
	out.v[5] = tb;
	out.v[6] = 0;
	out.v[7] = 0;
	// Column 2
	out.v[8] = rl_mid;
	out.v[9] = tb_mid;
	out.v[10] = fn;
	out.v[11] = -1;
	// Column 3
	out.v[12] = 0;
	out.v[13] = 0;
	out.v[14] = fn2;
	out.v[15] = 0;
	return true;
}

vk::raii::Sampler make_sampler(wivrn::vk_bundle & vk, vk::Filter filter, const char * name)
{
	vk::raii::Sampler res{
	        vk.device,
	        vk::SamplerCreateInfo{
	                .magFilter = filter,
	                .minFilter = filter,
	                .mipmapMode = vk::SamplerMipmapMode::eNearest,
	                .addressModeU = vk::SamplerAddressMode::eClampToEdge,
	                .addressModeV = vk::SamplerAddressMode::eClampToEdge,
	                .addressModeW = vk::SamplerAddressMode::eClampToEdge,
	                .minLod = 0.0f,
	                .maxLod = 0.0f,
	                .borderColor = vk::BorderColor::eFloatOpaqueBlack,
	        }};
	vk.name(res, name);
	return res;
}

image_allocation make_target(
        wivrn::vk_bundle & vk,
        vk::Extent2D extent,
        vk::Format format,
        vk::ImageUsageFlags usage,
        const char * name)
{
	return image_allocation{
	        vk.device,
	        vk::ImageCreateInfo{
	                .imageType = vk::ImageType::e2D,
	                .format = format,
	                .extent = {extent.width, extent.height, 1},
	                .mipLevels = 1,
	                .arrayLayers = 1,
	                .samples = vk::SampleCountFlagBits::e1,
	                .usage = usage,
	        },
	        VmaAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO},
	        name};
}

vk::raii::ImageView make_view(
        wivrn::vk_bundle & vk,
        vk::Image image,
        vk::Format format,
        vk::ImageUsageFlags usage,
        const char * name)
{
	vk::ImageViewUsageCreateInfo view_usage{
	        .usage = usage,
	};
	vk::raii::ImageView res{
	        vk.device,
	        vk::ImageViewCreateInfo{
	                .pNext = &view_usage,
	                .image = image,
	                .viewType = vk::ImageViewType::e2D,
	                .format = format,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .levelCount = 1,
	                        .layerCount = 1,
	                },
	        },
	};
	vk.name(res, name);
	return res;
}

buffer_allocation make_warp_ubo(vk::raii::Device & device, size_t size, const char * name)
{
	return buffer_allocation{
	        device,
	        vk::BufferCreateInfo{
	                .size = size,
	                .usage = vk::BufferUsageFlagBits::eUniformBuffer,
	        },
	        VmaAllocationCreateInfo{
	                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	        },
	        name};
}

template <typename T>
void upload_ubo(buffer_allocation & ubo, const T & data)
{
	// Maps, copies, flushes if the heap needs it: correct on coherent
	// and non-coherent memory alike.
	vmaCopyMemoryToAllocation(vk_allocator::instance(), &data, ubo, 0, sizeof(data));
}

vk::raii::DescriptorSetLayout make_ds_layout(
        wivrn::vk_bundle & vk,
        std::span<const vk::DescriptorSetLayoutBinding> bindings,
        const char * name)
{
	vk::raii::DescriptorSetLayout res{
	        vk.device,
	        vk::DescriptorSetLayoutCreateInfo{
	                .bindingCount = uint32_t(bindings.size()),
	                .pBindings = bindings.data(),
	        },
	};
	vk.name(*res, name);
	return res;
}

vk::raii::Pipeline make_pipeline(
        wivrn::vk_bundle & vk,
        vk::PipelineLayout layout,
        const char * shader_name,
        const char * name)
{
	auto shader = vk.load_shader(shader_name);
	vk::raii::Pipeline res{
	        vk.device,
	        nullptr,
	        vk::ComputePipelineCreateInfo{
	                .stage = {
	                        .stage = vk::ShaderStageFlagBits::eCompute,
	                        .module = *shader,
	                        .pName = "main",
	                },
	                .layout = layout,
	        },
	};
	vk.name(*res, name);
	return res;
}

} // namespace

namespace wivrn
{

depth_reprojector::depth_reprojector(vk_bundle & vk, vk::Extent2D target_size, bool reverse_z) :
        reverse_z(reverse_z),
        extent(target_size),
        color_sampler(make_sampler(vk, vk::Filter::eLinear, "reproject color sampler")),
        depth_sampler(make_sampler(vk, vk::Filter::eNearest, "reproject depth sampler"))
{
	for (size_t slot = 0; slot < num_slots; ++slot)
	{
		for (int view = 0; view < 2; ++view)
		{
			target[slot][view] = make_target(
			        vk, target_size, vk::Format::eR8G8B8A8Unorm,
			        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
			        std::format("reproject target {}:{}", slot, view).c_str());
			target_view[slot][view] = make_view(
			        vk, target[slot][view], vk::Format::eR8G8B8A8Unorm,
			        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
			        std::format("reproject target view {}:{}", slot, view).c_str());
			splat_depth[slot][view] = make_target(
			        vk, target_size, vk::Format::eR32Uint,
			        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
			        std::format("reproject splat {}:{}", slot, view).c_str());
			splat_view[slot][view] = make_view(
			        vk, splat_depth[slot][view], vk::Format::eR32Uint,
			        vk::ImageUsageFlagBits::eStorage,
			        std::format("reproject splat view {}:{}", slot, view).c_str());
			splat_ubo[slot][view] = make_warp_ubo(vk.device, sizeof(reproject_splat_ubo), std::format("reproject splat UBO {}:{}", slot, view).c_str());
			pull_ubo[slot][view] = make_warp_ubo(vk.device, sizeof(reproject_pull_ubo), std::format("reproject pull UBO {}:{}", slot, view).c_str());
		}
	}
	std::array splat_bindings{
	        vk::DescriptorSetLayoutBinding{
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        },
	        vk::DescriptorSetLayoutBinding{
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        },
	        vk::DescriptorSetLayoutBinding{
	                .binding = 2,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        },
	};
	splat_ds_layout = make_ds_layout(vk, splat_bindings, "reproject splat set layout");
	splat_layout = vk::raii::PipelineLayout{
	        vk.device,
	        vk::PipelineLayoutCreateInfo{
	                .setLayoutCount = 1,
	                .pSetLayouts = &*splat_ds_layout,
	        },
	};
	vk.name(*splat_layout, "reproject splat pipeline layout");
	splat_pipeline = make_pipeline(vk, *splat_layout, "reproject_splat", "reproject splat pipeline");

	std::array pull_bindings{
	        vk::DescriptorSetLayoutBinding{
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        },
	        vk::DescriptorSetLayoutBinding{
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        },
	        vk::DescriptorSetLayoutBinding{
	                .binding = 2,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        },
	        vk::DescriptorSetLayoutBinding{
	                .binding = 3,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        },
	};
	pull_ds_layout = make_ds_layout(vk, pull_bindings, "reproject pull set layout");
	pull_layout = vk::raii::PipelineLayout{
	        vk.device,
	        vk::PipelineLayoutCreateInfo{
	                .setLayoutCount = 1,
	                .pSetLayouts = &*pull_ds_layout,
	        },
	};
	vk.name(*pull_layout, "reproject pull pipeline layout");
	pull_pipeline = make_pipeline(vk, *pull_layout, "reproject_pull", "reproject pull pipeline");

	// Per (slot, eye): splat takes 1 combined + 1 uniform + 1 storage,
	// pull takes 1 combined + 1 uniform + 2 storage; 4 pairs total.
	std::array pool_sizes{
	        vk::DescriptorPoolSize{
	                .type = vk::DescriptorType::eCombinedImageSampler,
	                .descriptorCount = num_slots * 4,
	        },
	        vk::DescriptorPoolSize{
	                .type = vk::DescriptorType::eUniformBuffer,
	                .descriptorCount = num_slots * 4,
	        },
	        vk::DescriptorPoolSize{
	                .type = vk::DescriptorType::eStorageImage,
	                .descriptorCount = num_slots * 6,
	        },
	};
	pool = vk::raii::DescriptorPool{
	        vk.device,
	        vk::DescriptorPoolCreateInfo{
	                .maxSets = num_slots * 4,
	                .poolSizeCount = pool_sizes.size(),
	                .pPoolSizes = pool_sizes.data(),
	        },
	};
	vk.name(*pool, "reproject descriptor pool");

	std::array<vk::DescriptorSetLayout, num_slots * 2> splat_layouts;
	splat_layouts.fill(*splat_ds_layout);
	auto splat_alloc = vk.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
	        .descriptorPool = *pool,
	        .descriptorSetCount = splat_layouts.size(),
	        .pSetLayouts = splat_layouts.data(),
	});
	std::array<vk::DescriptorSetLayout, num_slots * 2> pull_layouts;
	pull_layouts.fill(*pull_ds_layout);
	auto pull_alloc = vk.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
	        .descriptorPool = *pool,
	        .descriptorSetCount = pull_layouts.size(),
	        .pSetLayouts = pull_layouts.data(),
	});
	for (size_t slot = 0; slot < num_slots; ++slot)
	{
		for (int view = 0; view < 2; ++view)
		{
			splat_sets[slot][view] = splat_alloc[slot * 2 + view].release();
			pull_sets[slot][view] = pull_alloc[slot * 2 + view].release();
			vk.name(splat_sets[slot][view], std::format("reproject splat set {}:{}", slot, view));
			vk.name(pull_sets[slot][view], std::format("reproject pull set {}:{}", slot, view));
		}
	}
}

bool depth_reprojector::reproject(
        vk::raii::Device & device,
        vk::raii::CommandBuffer & cmd,
        int slot,
        int view,
        vk::ImageView src_color,
        vk::ImageView src_depth,
        const std::array<float, 4> & color_rect,
        const std::array<float, 4> & depth_rect,
        bool flip_color,
        bool flip_depth,
        const xrt_pose & src_pose,
        const xrt_fov & src_fov,
        float src_near_z,
        float src_far_z,
        const xrt_pose & dst_pose,
        const xrt_fov & dst_fov)
{
	// Fold the matrix products on the CPU and validate first: nothing
	// is recorded on degenerate input.
	xrt_matrix_4x4 src_proj, src_view, dst_proj, dst_view;
	xrt_matrix_4x4 render_inv_p, render_inv_v, dst_inv_p, dst_inv_v;
	if (not frustum_gl(src_fov, src_near_z, src_far_z, src_proj))
		return false;
	math_matrix_4x4_view_from_pose(&src_pose, &src_view);
	math_matrix_4x4_inverse(&src_proj, &render_inv_p);
	math_matrix_4x4_inverse(&src_view, &render_inv_v);
	if (not frustum_gl(dst_fov, src_near_z, src_far_z, dst_proj))
		return false;
	math_matrix_4x4_view_from_pose(&dst_pose, &dst_view);
	math_matrix_4x4_inverse(&dst_proj, &dst_inv_p);
	math_matrix_4x4_inverse(&dst_view, &dst_inv_v);

	reproject_splat_ubo splat_data{};
	math_matrix_4x4_multiply(&render_inv_v, &render_inv_p, &splat_data.render_inv_vp);
	math_matrix_4x4_multiply(&dst_proj, &dst_view, &splat_data.warp_vp);
	std::memcpy(splat_data.depth_rect, depth_rect.data(), sizeof(splat_data.depth_rect));
	splat_data.depth_flip = flip_depth ? 1.f : 0.f;
	splat_data.src_reverse_z = reverse_z ? 1.f : 0.f;
	upload_ubo(splat_ubo[slot][view], splat_data);

	reproject_pull_ubo pull_data{};
	math_matrix_4x4_multiply(&dst_inv_v, &dst_inv_p, &pull_data.dst_inv);
	math_matrix_4x4_multiply(&src_proj, &src_view, &pull_data.src_vp);
	std::memcpy(pull_data.color_rect, color_rect.data(), sizeof(pull_data.color_rect));
	pull_data.color_flip = flip_color ? 1.f : 0.f;
	upload_ubo(pull_ubo[slot][view], pull_data);

	// The layer views change per app frame, so the sets are rewritten
	// per warp. Slot-owned: CPU rewrites are unordered by GPU barriers,
	// and slot i is only touched while image i is held.
	vk::DescriptorImageInfo splat_depth_info{
	        .sampler = *depth_sampler,
	        .imageView = src_depth,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	vk::DescriptorBufferInfo splat_ubo_info{
	        .buffer = splat_ubo[slot][view],
	        .range = vk::WholeSize,
	};
	vk::DescriptorImageInfo splat_target_info{
	        .imageView = *splat_view[slot][view],
	        .imageLayout = vk::ImageLayout::eGeneral,
	};
	vk::DescriptorImageInfo pull_color_info{
	        .sampler = *color_sampler,
	        .imageView = src_color,
	        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};
	vk::DescriptorBufferInfo pull_ubo_info{
	        .buffer = pull_ubo[slot][view],
	        .range = vk::WholeSize,
	};
	vk::DescriptorImageInfo pull_target_info{
	        .imageView = *target_view[slot][view],
	        .imageLayout = vk::ImageLayout::eGeneral,
	};
	std::array writes{
	        vk::WriteDescriptorSet{
	                .dstSet = splat_sets[slot][view],
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &splat_depth_info,
	        },
	        vk::WriteDescriptorSet{
	                .dstSet = splat_sets[slot][view],
	                .dstBinding = 1,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .pBufferInfo = &splat_ubo_info,
	        },
	        vk::WriteDescriptorSet{
	                .dstSet = splat_sets[slot][view],
	                .dstBinding = 2,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .pImageInfo = &splat_target_info,
	        },
	        vk::WriteDescriptorSet{
	                .dstSet = pull_sets[slot][view],
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	                .pImageInfo = &pull_color_info,
	        },
	        vk::WriteDescriptorSet{
	                .dstSet = pull_sets[slot][view],
	                .dstBinding = 1,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eUniformBuffer,
	                .pBufferInfo = &pull_ubo_info,
	        },
	        vk::WriteDescriptorSet{
	                .dstSet = pull_sets[slot][view],
	                .dstBinding = 2,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .pImageInfo = &splat_target_info,
	        },
	        vk::WriteDescriptorSet{
	                .dstSet = pull_sets[slot][view],
	                .dstBinding = 3,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .pImageInfo = &pull_target_info,
	        },
	};
	device.updateDescriptorSets(writes, {});

	// First warp transitions from Undefined; later warps from the
	// steady state (splat GENERAL, target SHADER_READ_ONLY).
	const bool ready = warp_ready[slot][view];
	warp_ready[slot][view] = true;
	const auto reuse_stage = ready ? vk::PipelineStageFlagBits2::eComputeShader
	                               : vk::PipelineStageFlagBits2::eTopOfPipe;
	vk::AccessFlags2 splat_access = ready ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
	                                      : vk::AccessFlags2{};
	vk::AccessFlags2 target_access = ready ? vk::AccessFlagBits2::eShaderRead : vk::AccessFlags2{};
	const auto splat_start = ready ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined;
	const auto target_start = ready ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eUndefined;

	std::array pre_barriers{
	        vk::ImageMemoryBarrier2{
	                .srcStageMask = reuse_stage,
	                .srcAccessMask = splat_access,
	                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
	                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
	                .oldLayout = splat_start,
	                .newLayout = vk::ImageLayout::eTransferDstOptimal,
	                .image = splat_depth[slot][view],
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .levelCount = 1,
	                        .layerCount = 1,
	                },
	        },
	        vk::ImageMemoryBarrier2{
	                .srcStageMask = reuse_stage,
	                .srcAccessMask = target_access,
	                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	                .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
	                .oldLayout = target_start,
	                .newLayout = vk::ImageLayout::eGeneral,
	                .image = target[slot][view],
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .levelCount = 1,
	                        .layerCount = 1,
	                },
	        },
	};
	cmd.pipelineBarrier2({
	        .imageMemoryBarrierCount = pre_barriers.size(),
	        .pImageMemoryBarriers = pre_barriers.data(),
	});

	// Sentinel = "no splat here" (above any encodable depth). Staged
	// through memcpy: C-array union members are not assignable, and
	// char/bytewise staging is the strict-aliasing-safe pattern.
	vk::ClearColorValue splat_empty{};
	const uint32_t empty_bits[4] = {0xffffffffu, 0, 0, 0};
	std::memcpy(splat_empty.uint32, empty_bits, sizeof(empty_bits));
	cmd.clearColorImage(
	        splat_depth[slot][view], vk::ImageLayout::eTransferDstOptimal, splat_empty,
	        vk::ImageSubresourceRange{
	                .aspectMask = vk::ImageAspectFlagBits::eColor,
	                .levelCount = 1,
	                .layerCount = 1,
	        });

	std::array clear_barriers{
	        vk::ImageMemoryBarrier2{
	                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
	                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
	                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	                .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
	                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
	                .newLayout = vk::ImageLayout::eGeneral,
	                .image = splat_depth[slot][view],
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .levelCount = 1,
	                        .layerCount = 1,
	                },
	        },
	};
	cmd.pipelineBarrier2({
	        .imageMemoryBarrierCount = clear_barriers.size(),
	        .pImageMemoryBarriers = clear_barriers.data(),
	});

	const uint32_t groups_x = (extent.width + 7) / 8;
	const uint32_t groups_y = (extent.height + 7) / 8;

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *splat_pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *splat_layout, 0, splat_sets[slot][view], {});
	cmd.dispatch(groups_x, groups_y, 1);

	// Splat atomics must complete before the pull reads.
	std::array mid_barriers{
	        vk::ImageMemoryBarrier2{
	                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
	                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
	                .oldLayout = vk::ImageLayout::eGeneral,
	                .newLayout = vk::ImageLayout::eGeneral,
	                .image = splat_depth[slot][view],
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .levelCount = 1,
	                        .layerCount = 1,
	                },
	        },
	};
	cmd.pipelineBarrier2({
	        .imageMemoryBarrierCount = mid_barriers.size(),
	        .pImageMemoryBarriers = mid_barriers.data(),
	});

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pull_pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pull_layout, 0, pull_sets[slot][view], {});
	cmd.dispatch(groups_x, groups_y, 1);

	// Warp output ends SHADER_READ_ONLY, ready to feed foveation.
	std::array out_barriers{
	        vk::ImageMemoryBarrier2{
	                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
	                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
	                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
	                .oldLayout = vk::ImageLayout::eGeneral,
	                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	                .image = target[slot][view],
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .levelCount = 1,
	                        .layerCount = 1,
	                },
	        },
	};
	cmd.pipelineBarrier2({
	        .imageMemoryBarrierCount = out_barriers.size(),
	        .pImageMemoryBarriers = out_barriers.data(),
	});

	return true;
}

std::array<vk::ImageView, 2> depth_reprojector::get_views(int slot)
{
	return {*target_view[slot][0], *target_view[slot][1]};
}

} // namespace wivrn
