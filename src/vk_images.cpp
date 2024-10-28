#include "vk_images.h"
#include "fmt/core.h"
#include "vk_mem_alloc.h"
#include <vulkan/vulkan_core.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

bool load_image_from_file(VulkanEngine& engine, const char* file, AllocatedImage& outImage)
{
	int texWidth, texHeight, texChannels;

	stbi_uc *pixels = stbi_load(file, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	if (!pixels)
	{
		fmt::println("Failed to load texture file {}", file);
		return false;
	}

	VkDeviceSize imageSize = texWidth * texHeight * texChannels;
	assert(texChannels == 4 && "We only support 4 channels right now");

	VkFormat imageFormat = VK_FORMAT_R8G8B8A8_SRGB;

	AllocatedBuffer stagingBuffer = engine.create_buffer(imageSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_ONLY);
	void *data;
	vmaMapMemory(engine._allocator, stagingBuffer._allocation, &data);
	memcpy(data, (void*)pixels, imageSize);
	vmaUnmapMemory(engine._allocator, stagingBuffer._allocation);
	stbi_image_free(pixels);
	pixels = nullptr;

	// create a new image
	VkExtent3D imageExtent {
		.width = static_cast<uint32_t>(texWidth),
		.height = static_cast<uint32_t>(texHeight),
		.depth = 1,
	};
	VkImageCreateInfo imageInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = imageFormat,
		.extent = imageExtent,
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	VmaAllocationCreateInfo imageAllocInfo {
		.usage = VMA_MEMORY_USAGE_GPU_ONLY,
	};
	AllocatedImage newImage;
	vmaCreateImage(engine._allocator,
		&imageInfo,
		&imageAllocInfo,
		&newImage._image,
		&newImage._allocation,
		nullptr);
	vmaSetAllocationName(engine._allocator, newImage._allocation, __FUNCTION__);

	engine.immediate_submit([=](VkCommandBuffer cmd){
		VkImageSubresourceRange range {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		};

		VkImageMemoryBarrier imageBarrierToTransfer {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.image = newImage._image,
			.subresourceRange = range,
		};

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&imageBarrierToTransfer);

		VkBufferImageCopy copyRegion {
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.imageOffset = {0,0,0},
			.imageExtent = imageExtent,
		};
		vkCmdCopyBufferToImage(cmd,
			stagingBuffer._buffer,
			newImage._image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &copyRegion);

		VkImageMemoryBarrier imageBarrierToReadable {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
			.image = newImage._image,
			.subresourceRange = range,
		};
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageBarrierToReadable);
	});

	engine._mainDeleteQueue.push_function([=](){
		vmaDestroyImage(engine._allocator, newImage._image, newImage._allocation);
	});
	vmaDestroyBuffer(engine._allocator, stagingBuffer._buffer, stagingBuffer._allocation);

	fmt::println("Texture loaded successfully {}", file);
	outImage = newImage;

	return true;
}