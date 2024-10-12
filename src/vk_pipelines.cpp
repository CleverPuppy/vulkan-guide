#include "fmt/core.h"
#include <vk_pipelines.h>
#include <vulkan/vulkan_core.h>

namespace vkutil
{

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass)
{
	VkPipelineViewportStateCreateInfo viewportState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = &_viewport,
		.scissorCount = 1,
		.pScissors = &_scissor
	};

	VkPipelineColorBlendStateCreateInfo colorBlendState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.logicOpEnable = false,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &_colorBlendAttachment,
		.blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}
	};

	VkGraphicsPipelineCreateInfo pipelineInfo = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stageCount = static_cast<uint32_t>(_shaderStages.size()),
		.pStages = _shaderStages.data(),
		.pVertexInputState = &_vertexInputInfo,
		.pInputAssemblyState = &_inputAssembly,
		.pTessellationState = nullptr,
		.pViewportState = &viewportState,
		.pRasterizationState = &_rasterizer,
		.pMultisampleState = &_multisampling,
		.pDepthStencilState = &_depthstencil,
		.pColorBlendState = &colorBlendState,
		.pDynamicState = nullptr,
		.layout = _pipelineLayout,
		.renderPass = pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = 0
	};

	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline)
		!= VK_SUCCESS)
	{
		fmt::println("failed to create pipeline");
		return VK_NULL_HANDLE;
	}

	return newPipeline;
}

}

