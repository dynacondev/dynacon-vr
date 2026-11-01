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

#include "depth_packer.h"

#include "utils/wivrn_vk_bundle.h"

namespace wivrn
{

depth_packer::depth_packer(vk_bundle & vk)
{
	std::array bindings{
	        vk::DescriptorSetLayoutBinding{
	                .binding = 0,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .descriptorCount = 1,
	                .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        },
	        vk::DescriptorSetLayoutBinding{
	                .binding = 1,
	                .descriptorType = vk::DescriptorType::eStorageImage,
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
	set_layout = vk::raii::DescriptorSetLayout{
	        vk.device,
	        vk::DescriptorSetLayoutCreateInfo{
	                .bindingCount = uint32_t(bindings.size()),
	                .pBindings = bindings.data(),
	        },
	};
	vk.name(*set_layout, "depth pack set layout");

	vk::PushConstantRange push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        .offset = 0,
	        .size = sizeof(push),
	};
	pipe_layout = vk::raii::PipelineLayout{
	        vk.device,
	        vk::PipelineLayoutCreateInfo{
	                .setLayoutCount = 1,
	                .pSetLayouts = &*set_layout,
	                .pushConstantRangeCount = 1,
	                .pPushConstantRanges = &push_range,
	        },
	};
	vk.name(*pipe_layout, "depth pack pipeline layout");

	auto shader = vk.load_shader("depth_pack");
	pipeline = vk::raii::Pipeline{
	        vk.device,
	        nullptr,
	        vk::ComputePipelineCreateInfo{
	                .stage = {
	                        .stage = vk::ShaderStageFlagBits::eCompute,
	                        .module = *shader,
	                        .pName = "main",
	                },
	                .layout = *pipe_layout,
	        },
	};
	vk.name(*pipeline, "depth pack pipeline");

	std::array pool_sizes{
	        vk::DescriptorPoolSize{
	                .type = vk::DescriptorType::eStorageImage,
	                .descriptorCount = uint32_t(num_slots * 3),
	        },
	};
	pool = vk::raii::DescriptorPool{
	        vk.device,
	        vk::DescriptorPoolCreateInfo{
	                .maxSets = uint32_t(num_slots),
	                .poolSizeCount = pool_sizes.size(),
	                .pPoolSizes = pool_sizes.data(),
	        },
	};
	vk.name(*pool, "depth pack descriptor pool");

	std::array<vk::DescriptorSetLayout, num_slots> layouts;
	layouts.fill(*set_layout);
	int i = 0;
	for (auto & s: vk.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
	              .descriptorPool = *pool,
	              .descriptorSetCount = layouts.size(),
	              .pSetLayouts = layouts.data(),
	      }))
		sets[i++] = s.release();
}

bool depth_packer::pack(vk::raii::Device & device,
                        vk::raii::CommandBuffer & cmd,
                        int slot,
                        vk::ImageView splat_l,
                        vk::ImageView splat_r,
                        vk::Extent2D src_extent,
                        float near_l,
                        float far_l,
                        float near_r,
                        float far_r,
                        vk::ImageView dst_y)
{
	if (slot < 0 or slot >= int(num_slots))
		return false;
	if (not(src_extent.width > 0 and src_extent.height > 0))
		return false;
	if (not(near_l > 0 and far_l > near_l and near_r > 0 and far_r > near_r))
		return false;
	if (splat_l == vk::ImageView{} or splat_r == vk::ImageView{} or dst_y == vk::ImageView{})
		return false;

	vk::DescriptorImageInfo splat_l_info{
	        .imageView = splat_l,
	        .imageLayout = vk::ImageLayout::eGeneral,
	};
	vk::DescriptorImageInfo splat_r_info{
	        .imageView = splat_r,
	        .imageLayout = vk::ImageLayout::eGeneral,
	};
	vk::DescriptorImageInfo dst_info{
	        .imageView = dst_y,
	        .imageLayout = vk::ImageLayout::eGeneral,
	};
	std::array writes{
	        vk::WriteDescriptorSet{
	                .dstSet = sets[slot],
	                .dstBinding = 0,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .pImageInfo = &splat_l_info,
	        },
	        vk::WriteDescriptorSet{
	                .dstSet = sets[slot],
	                .dstBinding = 1,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .pImageInfo = &splat_r_info,
	        },
	        vk::WriteDescriptorSet{
	                .dstSet = sets[slot],
	                .dstBinding = 2,
	                .descriptorCount = 1,
	                .descriptorType = vk::DescriptorType::eStorageImage,
	                .pImageInfo = &dst_info,
	        },
	};
	device.updateDescriptorSets(writes, {});

	push pc{
	        .src_w = float(src_extent.width),
	        .src_h = float(src_extent.height),
	        .near_l = near_l,
	        .far_l = far_l,
	        .near_r = near_r,
	        .far_r = far_r,
	};

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipe_layout, 0, sets[slot], {});
	cmd.pushConstants<push>(*pipe_layout, vk::ShaderStageFlagBits::eCompute, 0, pc);
	cmd.dispatch((sbs_width + 7) / 8, (sbs_height + 7) / 8, 1);
	return true;
}

} // namespace wivrn
