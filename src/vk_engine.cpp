//> includes
#include "vk_engine.h"
#include "SDL_events.h"
#include "SDL_keycode.h"
#include "SDL_video.h"
#include "VkBootstrap.h"
#include "fmt/core.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/matrix.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "mesh.h"
#include <array>
#include <cstring>
#include <optional>
#include <stddef.h>
#include <utility>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "vk_pipelines.h"
#include <SDL.h>
#include <SDL_vulkan.h>

#include <_types/_uint32_t.h>
#include <fstream>
#include <vector>
#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <thread>
#include <vulkan/vulkan_core.h>

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }
void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    // SDL_Renderer *pSDLRender = SDL_GetRenderer(_window);
    // int rw = 0, rh = 0;
    // SDL_GetRendererOutputSize(pSDLRender, &rw, &rh);
    // if(rw != _windowExtent.width) {
    //     float widthScale = (float)rw / (float) _windowExtent.width;
    //     float heightScale = (float)rh / (float) _windowExtent.height;

    //     if(widthScale != heightScale) {
    //         fprintf(stderr, "WARNING: width scale != height scale\n");
    //     }

    //     SDL_RenderSetScale(pSDLRender, widthScale, heightScale);
    // }
    update_rendersize();

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

	init_default_renderpass();

    init_framebuffers();

    init_pipelines();

    load_meshs();

    init_scene();

    init_camera();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) {

        // make sure the gpu has stopped doint its things
        vkDeviceWaitIdle(_device);

        destroy_depth_image(); // before the _mainDeleteQueue.flush(), since this is allocated by vma
        destroy_framebuffers();
        _mainDeleteQueue.flush();

        for (int i = 0; i < FRAME_OVERLAP; ++i)
        {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);
        }

        // vkQueue and vkPhysicalDevices are not really created resources.
        // so no need to cleanup.

        destroy_swapchain();

        // cleanup render pass and frame buffers
		//destroy the main renderpass
		vkDestroyRenderPass(_device, _renderPass, nullptr);

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debugMessager);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    // wait for GPU finish render for last frame, timeout is 1s.
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    // request the image from the swapchain
    uint32_t swapchainImageIdx;
    VkResult nextImageResult = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIdx);
    if (nextImageResult != VK_SUCCESS)
    {
        // window resize happened.
        if (nextImageResult == VK_ERROR_OUT_OF_DATE_KHR)
        {
            _resize_requested = true;
        }
        return;
    }

    // obtain current cmd buffer and reset it to record again
    VkCommandBuffer cmd = get_current_frame()._commandBuffer;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    // we only use this command buffer once.
    VkCommandBufferBeginInfo cmdBufferBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBufferBeginInfo));

    // do a clear
    VkClearValue clearValue;
    float flash = abs(sin(_frameNumber / 120.f));
    clearValue.color = {.float32{0.0f, 0.0f, flash, 1.0f}};

    VkClearValue clearDepthValue {
        .depthStencil {
            .depth = 1.0,
            .stencil = 0
        }
    };

    std::array<VkClearValue, 2> clearValueArray = {clearValue, clearDepthValue};

    VkRenderPassBeginInfo rpInfo = VkRenderPassBeginInfo {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = _renderPass,
        .framebuffer = _framebuffers[swapchainImageIdx],
        .renderArea = {0, 0, _renderExtent},
        .clearValueCount = clearValueArray.size(),
        .pClearValues = clearValueArray.data(),
    };

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    draw_objects(cmd, _renderables.data(), _renderables.size());
    vkCmdEndRenderPass(cmd);

    // finalize the command buffer
    VK_CHECK(vkEndCommandBuffer(cmd));

    // submit cmd
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo = VkSubmitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &get_current_frame()._swapchainSemaphore,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &get_current_frame()._renderSemaphore
    };
    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submitInfo, get_current_frame()._renderFence));

    // show to the display
    VkPresentInfoKHR presentInfo = vkinit::present_info();
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &_swapchain;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;

    presentInfo.pImageIndices = &swapchainImageIdx;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult != VK_SUCCESS)
    {
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR)
        {
            // window resize happend
            _resize_requested = true;
        }
        return;
    }
    ++_frameNumber;
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESIZED) {
                    fmt::println("Window resized");
                    _resize_requested = true;
                }
            }
            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_SPACE) {
                    _selectShader = (_selectShader + 1) % 2;
                }
            }
        }

        // handle window resize
        if (_resize_requested)
        {
            handle_window_resize();

            _resize_requested = false;
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    auto inst_ret = builder.set_app_name("Example Vulkan Application")
        .request_validation_layers(useValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    _instance = vkb_inst.instance;
    _debugMessager = vkb_inst.debug_messenger;

    // get VkSurface form the SDL window
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    VkPhysicalDeviceVulkan13Features features13 {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features12 {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;


    // use vkBootstrap to select physical GPU
    vkb::PhysicalDeviceSelector selector {vkb_inst};
    vkb::PhysicalDevice vkb_physicalDevice = selector
        .set_minimum_version(1, 2)
        .set_required_features_12(features12)
        // .set_required_features_13(features13)
        .set_surface(_surface)
        .select()
        .value();

    // create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{ vkb_physicalDevice };
    VkPhysicalDeviceShaderDrawParametersFeatures shader_draw_parameters_feature = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
        .pNext = nullptr,
        .shaderDrawParameters = VK_TRUE,
    };
    deviceBuilder.add_pNext(&shader_draw_parameters_feature);
    vkb::Device vkb_device = deviceBuilder.build().value();

    _device = vkb_device.device;
    _chosenGPU = vkb_device.physical_device;
    _gpuProperties = vkb_device.physical_device.properties;
    fmt::println("The GPU has a minimum buffer alignment of {}", _gpuProperties.limits.minUniformBufferOffsetAlignment);

    // use vkb to get a graphics queue
    _graphicsQueue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    // init the memory allocator
    VmaAllocatorCreateInfo allocateInfo = {};
    allocateInfo.physicalDevice = _chosenGPU;
    allocateInfo.device = _device;
    allocateInfo.instance = _instance;
    vmaCreateAllocator(&allocateInfo, &_allocator);
    _mainDeleteQueue.push_function([=](){vmaDestroyAllocator(_allocator);});
}
void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);
    init_depth_image();
}

void VulkanEngine::init_commands()
{
    // create a commandpool for commands submitted to the graphics queue

    // VkCommandPoolCreateInfo commandPoolInfo = VkCommandPoolCreateInfo {
    //     .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    //     .pNext = nullptr,
    //     .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    //     .queueFamilyIndex = _graphicsQueueFamily};

    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
        _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; ++i)
    {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));
        // allocate the command buffer

        // VkCommandBufferAllocateInfo cmdAllocInfo{
        //     .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        //     .pNext = nullptr,
        //     .commandPool = _frames[i]._commandPool,
        //     .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        //     .commandBufferCount = 1
        // };
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._commandBuffer));
    }
}

void VulkanEngine::init_sync_structures()
{
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; ++i)
    {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
    }
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder builder{_chosenGPU, _device, _surface};

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkb_swapchain = builder
        .set_desired_format(VkSurfaceFormatKHR{.format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        // use the vsync mode
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchain = vkb_swapchain.swapchain;
    _swapchainExtent = vkb_swapchain.extent;
    _swapchainImages = vkb_swapchain.get_images().value();
    _swapchainImageViews = vkb_swapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
    // destroy the swapchain will destroy the images it holds internally.
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    // destroy related image views
    for (auto& imageView : _swapchainImageViews)
    {
        vkDestroyImageView(_device, imageView, nullptr);
    }
}

void VulkanEngine::init_default_renderpass()
{
    VkAttachmentDescription color_attachment = VkAttachmentDescription {
        .format = _swapchainImageFormat,
        // no msaa use, so 1 samples
        .samples = VK_SAMPLE_COUNT_1_BIT,
        // clear the color
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        // don't use stencil test
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,

        // don't care the initial layout
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        // the final layout must be the same as the layout ready for display
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };

    VkAttachmentDescription depth_attachment {
        .flags = 0,
        .format = _depthImageFormat,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    VkAttachmentReference color_attachment_ref = VkAttachmentReference {
        //attachment number will index into the pAttachments array in the parent renderpass itself
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkAttachmentReference depth_attachment_ref {
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    // create one subpass
    VkSubpassDescription subpass = VkSubpassDescription {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .pDepthStencilAttachment = &depth_attachment_ref,
    };

    VkAttachmentDescription attachments[2] = {color_attachment, depth_attachment};

    // color attachment dependencies
    VkSubpassDependency color_dependency {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };
    // depth attachment dependencies
    VkSubpassDependency depth_dependency {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = 0,
        .dstAccessMask =VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };

    // UNDEFINED -> RenderPass Begins -> Subpass 0 begins (Transition to
    // Attachment Optimal) -> Subpass 0 renders -> Subpass 0 ends -> Renderpass
    // Ends (Transitions to Present Source)
    std::array<VkSubpassDependency, 2> dependencies = {
        color_dependency,
        depth_dependency
    };

    VkRenderPassCreateInfo render_pass_info = VkRenderPassCreateInfo {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 2,
        .pAttachments = attachments,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = dependencies.size(),
        .pDependencies = dependencies.data(),
    };

    VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));
}
void VulkanEngine::init_framebuffers()
{
    VkFramebufferCreateInfo framebuffer_info = VkFramebufferCreateInfo {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .renderPass = _renderPass,
        .attachmentCount = 1,
        .pAttachments = nullptr,
        .width = _renderExtent.width,
        .height = _renderExtent.height,
        .layers = 1
    };

    const uint32_t swapchain_imagecount = _swapchainImages.size();
    _framebuffers.resize(swapchain_imagecount);
    std::array<VkImageView, 2> imageViews;
    for (int i = 0; i < swapchain_imagecount; ++i)
    {
        imageViews[0] = _swapchainImageViews[i];
        imageViews[1] = _depthImageView;
        framebuffer_info.pAttachments = imageViews.data();
        framebuffer_info.attachmentCount = imageViews.size();
        VK_CHECK(vkCreateFramebuffer(_device, &framebuffer_info, nullptr, &_framebuffers[i]));
    }
}

void VulkanEngine::destroy_framebuffers()
{
    for (const auto& framebuffer : _framebuffers)
    {
        vkDestroyFramebuffer(_device, framebuffer, nullptr);
    }
}

void VulkanEngine::update_rendersize()
{
    int w,h;
    SDL_GetWindowSizeInPixels(_window, &w, &h);
    _renderExtent = {static_cast<uint32_t>(w),static_cast<uint32_t>(h)};
}

bool VulkanEngine::load_shader_module(const char* filepath, VkShaderModule* outShaderModule)
{
    std::ifstream ifs{filepath, std::ios::ate | std::ios::binary};
    if (!ifs.is_open())
    {
        return false;
    }

    //find what the size of the file is by looking up the location of the cursor
    //because the cursor is at the end, it gives the size directly in bytes
    size_t fileSize = (size_t)ifs.tellg();

    //spirv expects the buffer to be on uint32, so make sure to reserve an int vector big enough for the entire file
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    //put file cursor at beginning
    ifs.seekg(0);

    //load the entire file into the buffer
    ifs.read((char*)buffer.data(), fileSize);

    //now that the file is loaded into the buffer, we can close it
    ifs.close();

    VkShaderModuleCreateInfo shaderModuleInfo ={};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.pNext = nullptr;
    shaderModuleInfo.codeSize = buffer.size() * sizeof(uint32_t);
    shaderModuleInfo.pCode = buffer.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(_device, &shaderModuleInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        return false;
    }

    *outShaderModule = shaderModule;
    return true;
}

void VulkanEngine::init_pipelines()
{
    VkShaderModule fragmentShader, vertexShader;
    VkShaderModule coloredFragmentShader, coloredVertexShader;
    VkShaderModule triMeshVertexShader, colorMeshShader;

    if (!load_shader_module("../shaders/triangle.frag.spv", &fragmentShader))
    {
        fmt::println("Error when building the triangle fragment shader module");
    }

    if (!load_shader_module("../shaders/triangle.vert.spv", &vertexShader))
    {
        fmt::println("Error when building the triangle vertex shader module");
    }

    if (!load_shader_module("../shaders/colored_triangle.frag.spv", &coloredFragmentShader))
    {
        fmt::println("Error when building the triangle fragment shader module");
    }

    if (!load_shader_module("../shaders/colored_triangle.vert.spv", &coloredVertexShader))
    {
        fmt::println("Error when building the triangle vertex shader module");
    }

    if (!load_shader_module("../shaders/tri_mesh.vert.spv", &triMeshVertexShader))
    {
        fmt::println("Error when building the triangle mesh vertex shader module");
    }

    if (!load_shader_module("../shaders/default_lit.frag.spv", &colorMeshShader))
    {
        fmt::println("Error when building the default_lit fragment shader module");
    }

    vkutil::PipelineBuilder pipeBuilder;
    VkPipelineShaderStageCreateInfo&& vertexStageInfo = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vertexShader);
    VkPipelineShaderStageCreateInfo&& fragmentStageInfo = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader);
    pipeBuilder._shaderStages.emplace_back(vertexStageInfo);
    pipeBuilder._shaderStages.emplace_back(fragmentStageInfo);

    // have no attributes yet.
    pipeBuilder._vertexInputInfo = VkPipelineVertexInputStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0,
    };

    pipeBuilder._inputAssembly = VkPipelineInputAssemblyStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE
    };

    pipeBuilder._viewport = VkViewport {
        .x = 0,
        .y = 0,
        .width = static_cast<float>(_renderExtent.width),
        .height = static_cast<float>(_renderExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    pipeBuilder._scissor = VkRect2D {
        .offset {0, 0},
        .extent = _renderExtent
    };

    pipeBuilder._rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE, // no cull yet
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f
    };

    // don't enable blender yet
    pipeBuilder._colorBlendAttachment = VkPipelineColorBlendAttachmentState{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    };

    pipeBuilder._multisampling = VkPipelineMultisampleStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE
    };

    // enable depth test
    pipeBuilder._depthstencil = VkPipelineDepthStencilStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = true,
        .depthWriteEnable = true,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = false,
        .stencilTestEnable = VK_FALSE,
    };

    // pipeline layout
    VkPipelineLayoutCreateInfo pipeLayoutInfo = VkPipelineLayoutCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr
    };

    if (vkCreatePipelineLayout(_device, &pipeLayoutInfo, nullptr, &_trianglePipelineLayout) != VK_SUCCESS)
    {
        fmt::println("vkCreatePipelineLayout _trianglePipelineLayout failed.");
        return;
    }
    _mainDeleteQueue.push_function([=](){vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);});

    pipeBuilder._pipelineLayout = _trianglePipelineLayout;
    _trianglePipeline = pipeBuilder.build_pipeline(_device, _renderPass);
    _mainDeleteQueue.push_function([=](){vkDestroyPipeline(_device, _trianglePipeline, nullptr);});

    // create a second pipeline, only change the shaders
    pipeBuilder._shaderStages.clear();
    pipeBuilder._shaderStages.emplace_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, coloredVertexShader)
    );
    pipeBuilder._shaderStages.emplace_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, coloredFragmentShader)
    );
    _coloredTrianglePipeline = pipeBuilder.build_pipeline(_device, _renderPass);
    _mainDeleteQueue.push_function([=](){vkDestroyPipeline(_device, _coloredTrianglePipeline, nullptr);});

    // create a mesh pipeline layout to load constants
    VkPushConstantRange pushConstantRange = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(MeshPushConstants),
    };
    std::array aSetLayouts = {_globalSetLayout, _objectSetLayout};
    pipeLayoutInfo = VkPipelineLayoutCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = aSetLayouts.size(),
        .pSetLayouts = aSetLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeLayoutInfo, nullptr, &_meshPipelineLayout));
    _mainDeleteQueue.push_function([=](){vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);});

    // create a mesh pipeline
    pipeBuilder._shaderStages.clear();
    pipeBuilder._shaderStages.emplace_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, triMeshVertexShader)
    );
    pipeBuilder._shaderStages.emplace_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, colorMeshShader)
    );
    auto vertexInputDescription = Vertex::get_vertex_description();
    pipeBuilder._vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = vertexInputDescription.flags,
        .vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputDescription.bindings.size()),
        .pVertexBindingDescriptions = vertexInputDescription.bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputDescription.attributes.size()),
        .pVertexAttributeDescriptions = vertexInputDescription.attributes.data()
    };
    pipeBuilder._pipelineLayout = _meshPipelineLayout;
    _meshPipeline = pipeBuilder.build_pipeline(_device, _renderPass);

    create_material(_meshPipeline, _meshPipelineLayout, "defaultmesh");
    _mainDeleteQueue.push_function([=](){vkDestroyPipeline(_device, _meshPipeline, nullptr);});

    vkDestroyShaderModule(_device, vertexShader, nullptr);
    vkDestroyShaderModule(_device, fragmentShader, nullptr);
    vkDestroyShaderModule(_device, coloredVertexShader, nullptr);
    vkDestroyShaderModule(_device, coloredFragmentShader, nullptr);
    vkDestroyShaderModule(_device, triMeshVertexShader, nullptr);
    vkDestroyShaderModule(_device, colorMeshShader, nullptr);
}

void VulkanEngine::load_meshs ()
{
    _triangleMesh._vertices.resize(3);

	//vertex positions
	_triangleMesh._vertices[0].position = { 1.f, 1.f, 0.0f };
	_triangleMesh._vertices[1].position = {-1.f, 1.f, 0.0f };
	_triangleMesh._vertices[2].position = { 0.f,-1.f, 0.0f };

	//vertex colors, all green
	_triangleMesh._vertices[0].color = { 0.f, 1.f, 0.0f }; //pure green
	_triangleMesh._vertices[1].color = { 0.f, 1.f, 0.0f }; //pure green
	_triangleMesh._vertices[2].color = { 0.f, 1.f, 0.0f }; //pure green
	//we don't care about the vertex normals

    // load obj model
    _monkeyMesh.load_from_obj("../assets/suzanne.obj");

	upload_mesh(_triangleMesh);
    upload_mesh(_monkeyMesh);

    _meshes["monkey"] = _monkeyMesh;
    _meshes["triangle"] = _triangleMesh;
}

void VulkanEngine::upload_mesh(Mesh& mesh)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext= nullptr;
    bufferInfo.size = mesh._vertices.size() * sizeof(decltype(mesh._vertices)::value_type);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo,
		&mesh._vertexBuffer._buffer,
		&mesh._vertexBuffer._allocation,
		nullptr));

    _mainDeleteQueue.push_function([=](){
        vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
    });

    void *data;
    vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);
    memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));
    vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);

}

void VulkanEngine::init_depth_image()
{
    _depthImageFormat = VK_FORMAT_D32_SFLOAT;
    VkExtent3D imageExtent {
        _renderExtent.width,
        _renderExtent.height,
        1
    };

    VkImageCreateInfo imageInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = _depthImageFormat,
        .extent = imageExtent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = VK_NULL_HANDLE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo allocInfo {
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    vmaCreateImage(_allocator, &imageInfo, &allocInfo, &_depthImage._image, &_depthImage._allocation, nullptr);

    VkImageViewCreateInfo imageViewInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = _depthImage._image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = _depthImageFormat,
        .components = VkComponentMapping {
            VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B,
            VK_COMPONENT_SWIZZLE_A},
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
    };
    VK_CHECK(vkCreateImageView(_device, &imageViewInfo, nullptr, &_depthImageView));
}

void VulkanEngine::destroy_depth_image()
{
    vkDestroyImageView(_device, _depthImageView, nullptr);
    vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
}

//create material and add it to the map
Material* VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout layout,const std::string& name)
{
    auto insetRet = _materials.insert(
        std::make_pair(name, Material{pipeline, layout})
    );

    if (!insetRet.second)
    {
        fmt::println("Already have Material {}", name);
    }

    return &_materials[name];
}

//returns nullptr if it can't be found
Material* VulkanEngine::get_material(const std::string& name)
{
    auto it = _materials.find(name);
    if (it == _materials.end()) return nullptr;

    return &it->second;
}

//returns nullptr if it can't be found
Mesh* VulkanEngine::get_mesh(const std::string& name)
{
    auto it = _meshes.find(name);
    if (it == _meshes.end()) return nullptr;

    return &it->second;
}

//our draw function
void VulkanEngine::draw_objects(VkCommandBuffer cmd,RenderObject* first, int count)
{
    const Material* lastMaterial = nullptr;
    const Mesh* lastMesh = nullptr;

    // update camera data
    GPUCameraData cameraData = _camera.GetGPUData();
    cameraData.proj[1][1] *= -1;
    cameraData.projview = cameraData.proj * cameraData.view;
    void* data;
    vmaMapMemory(_allocator, get_current_frame()._cameraBuffer._allocation, &data);
    memcpy(data, &cameraData, sizeof(GPUCameraData));
    vmaUnmapMemory(_allocator, get_current_frame()._cameraBuffer._allocation);

    // update scene data
    float framed = (_frameNumber / 480.0f);
    _sceneParameters.ambientColor = {sin(framed), 0, cos(framed), 1.0};
    void* sceneData;
    vmaMapMemory(_allocator, _sceneParameterBuffer._allocation, &sceneData);
    memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData));
    vmaUnmapMemory(_allocator, _sceneParameterBuffer._allocation);

    void* objectData;
    vmaMapMemory(_allocator, get_current_frame()._objectBuffer._allocation, &objectData);
    GPUObjectData* objectSSBO = static_cast<GPUObjectData*>(objectData);
    for (int i = 0; i < count; ++i)
    {
        objectSSBO[i].modelMatrix = (first + i)->transformMatrix;
    }
    vmaUnmapMemory(_allocator, get_current_frame()._objectBuffer._allocation);
    for (int i = 0; i < count; ++i)
    {
        /* updaate pipeline if needed. */
        RenderObject *target = first + i;
        if (target->material != lastMaterial)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, target->material->pipeline);
            lastMaterial= target->material;

            // bind descriptor sets when change the pipeline
            // vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, target->material->pipelineLayout,
            //     0, 1, &get_current_frame()._globalDescriptor, 0, nullptr);

            // dynamic descriptor sets can change the offset
            uint32_t uniform_offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * i;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                target->material->pipelineLayout, 0,
                1, &get_current_frame()._globalDescriptor,
                1, &uniform_offset);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                target->material->pipelineLayout, 1,
                1, &get_current_frame()._objectDescriptor,
                0, VK_NULL_HANDLE);
        }

        /* update const push */
        MeshPushConstants constants {
            .render_matrix = target->transformMatrix,
        };
        vkCmdPushConstants(cmd, target->material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

        /* bind vertex buffers if needed */
        if (target->mesh != lastMesh)
        {
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &target->mesh->_vertexBuffer._buffer, &offset);
            lastMesh = target->mesh;
        }

        // vkCmdDraw(cmd, target->mesh->_vertices.size(), 1, 0, 0);
        vkCmdDraw(cmd, target->mesh->_vertices.size(), 1, 0, i);
    }
}

void VulkanEngine::init_scene()
{
    RenderObject monkey;
    monkey.mesh = get_mesh("monkey");
    monkey.material = get_material("defaultmesh");
    monkey.transformMatrix = glm::mat4(1.0f);

    _renderables.emplace_back(monkey);

	for (int x = -20; x <= 20; x++) {
		for (int y = -20; y <= 20; y++) {

			RenderObject tri;
			tri.mesh = get_mesh("triangle");
			tri.material = get_material("defaultmesh");
			glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x, 0, y));
			glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(0.2, 0.2, 0.2));
			tri.transformMatrix = translation * scale;

			_renderables.push_back(tri);
		}
	}
}

void VulkanEngine::init_descriptors()
{
    { // create _globalSetLayout
        VkDescriptorSetLayoutBinding cameraDescriptorBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
        VkDescriptorSetLayoutBinding sceneDescriptorBindig = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            1);
        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {cameraDescriptorBinding, sceneDescriptorBindig};
        VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(bindings.data(), bindings.size());
        VK_CHECK(vkCreateDescriptorSetLayout(_device, &setinfo, nullptr, &_globalSetLayout));
    }

    { // create _objectSetLayout
        VkDescriptorSetLayoutBinding objectDescriptorBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
        VkDescriptorSetLayoutCreateInfo setinfo = vkinit::descriptorset_layout_create_info(&objectDescriptorBinding, 1);
        VK_CHECK(vkCreateDescriptorSetLayout(
            _device, &setinfo, nullptr, &_objectSetLayout));
    }

    // create a pool holds 10 uniform buffers
    std::vector<VkDescriptorPoolSize> vecPoolSize {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
    };
    VkDescriptorPoolCreateInfo poolInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = 10,
        .poolSizeCount = (uint32_t) vecPoolSize.size(),
        .pPoolSizes = vecPoolSize.data()
    };
    VK_CHECK(vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptorPool));

    _mainDeleteQueue.push_function([=](){
        vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _objectSetLayout, nullptr);
        vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
    });

    // dynamic descriptor sets buffer alloc
    const size_t sceneParameterBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));
    _sceneParameterBuffer = create_buffer(sceneParameterBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    _mainDeleteQueue.push_function([=](){
        vmaDestroyBuffer(_allocator, _sceneParameterBuffer._buffer, _sceneParameterBuffer._allocation);
    });

    constexpr int MAX_OBJECTS = 10000;
    for (int i = 0; i < FRAME_OVERLAP; ++i)
    {
        // create uniform buffer and setup free functions
        _frames[i]._cameraBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        _frames[i]._objectBuffer = create_buffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        _mainDeleteQueue.push_function([=](){
            vmaDestroyBuffer(_allocator, _frames[i]._cameraBuffer._buffer, _frames[i]._cameraBuffer._allocation);
            vmaDestroyBuffer(_allocator, _frames[i]._objectBuffer._buffer, _frames[i]._objectBuffer._allocation);
        });

        // alloc one descriptor set from the descriptor pool per frame
        VkDescriptorSetAllocateInfo descriptorAllocInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = _descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &_globalSetLayout,
        };
        VK_CHECK(vkAllocateDescriptorSets(_device, &descriptorAllocInfo, &_frames[i]._globalDescriptor));

        // allocate _objectDescriptorSet
        descriptorAllocInfo.pSetLayouts = &_objectSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(_device, &descriptorAllocInfo, &_frames[i]._objectDescriptor));

        VkDescriptorBufferInfo cameraBufferInfo {
            .buffer = _frames[i]._cameraBuffer._buffer,
            .offset = 0,
            .range = sizeof(GPUCameraData),
        };
        VkDescriptorBufferInfo sceneBufferInfo {
            .buffer = _sceneParameterBuffer._buffer,
            // .offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * i,
            .offset = 0,
            .range = sizeof(GPUSceneData)
        };
        VkDescriptorBufferInfo objectBufferInfo {
            .buffer = _frames[i]._objectBuffer._buffer,
            .offset = 0,
            .range = sizeof(GPUObjectData) * MAX_OBJECTS,
        };

        VkWriteDescriptorSet&& writeCameraBuffer = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            _frames[i]._globalDescriptor, &cameraBufferInfo, 0);
        VkWriteDescriptorSet&& writeSceneBuffer = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            _frames[i]._globalDescriptor, &sceneBufferInfo, 1);
        VkWriteDescriptorSet&& writeObjectBuffer = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            _frames[i]._objectDescriptor, &objectBufferInfo, 0);
        std::array writes {
            writeCameraBuffer,
            writeSceneBuffer,
            writeObjectBuffer,
        };

        vkUpdateDescriptorSets(_device, writes.size(), writes.data(), 0, VK_NULL_HANDLE);
    }

}

void VulkanEngine::handle_window_resize()
{
    vkDeviceWaitIdle(_device);

    destroy_depth_image();
    destroy_framebuffers();
    destroy_swapchain();

    update_rendersize();

    create_swapchain(_renderExtent.width, _renderExtent.height);
    init_depth_image();
    init_framebuffers();
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = allocSize,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = VK_NULL_HANDLE
    };

    VmaAllocationCreateInfo allocInfo {
        .usage = memoryUsage,
    };

    AllocatedBuffer newBuffer;

    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo,
        &allocInfo, &newBuffer._buffer,
        &newBuffer._allocation, nullptr));

    return newBuffer;
}

void VulkanEngine::init_camera(void)
{
    /* setting up cameras */
    glm::vec3 cameraPos = {6.0f, 6.0f, -6.0f};
    glm::vec3 up {0.0, 1.0, 0.0};
    glm::vec3 center {0.0, 0.0, 0.0};
    _camera = Camera(cameraPos, up, center);
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t origSize)
{
    size_t align = _gpuProperties.limits.minUniformBufferOffsetAlignment;

    return (origSize + align - 1) & ~(align - 1);
}