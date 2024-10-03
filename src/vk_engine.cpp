//> includes
#include "vk_engine.h"
#include "SDL_video.h"
#include "VkBootstrap.h"
#include "fmt/core.h"
#include <SDL.h>
#include <SDL_vulkan.h>

#include <_types/_uint32_t.h>
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

	init_default_renderpass();

    init_framebuffers();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) {

        // make sure the gpu has stopped doint its things
        vkDeviceWaitIdle(_device);

        for (int i = 0; i < FRAME_OVERLAP; ++i)
        {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
        }

        // vkQueue and vkPhysicalDevices are not really created resources.
        // so no need to cleanup.

        destroy_swapchain();

        // cleanup render pass and frame buffers
		//destroy the main renderpass
		vkDestroyRenderPass(_device, _renderPass, nullptr);
		for (int i = 0; i < _framebuffers.size(); i++) {
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
		}

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
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIdx));

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

    VkRenderPassBeginInfo rpInfo = VkRenderPassBeginInfo {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = _renderPass,
        .framebuffer = _framebuffers[swapchainImageIdx],
        .renderArea = {0, 0, _renderExtent},
        .clearValueCount = 1,
        .pClearValues = &clearValue
    };

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

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

    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));
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

                }
            }
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
    vkb::Device vkb_device = deviceBuilder.build().value();

    _device = vkb_device.device;
    _chosenGPU = vkb_device.physical_device;

    // use vkb to get a graphics queue
    _graphicsQueue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

}
void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);
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

    VkAttachmentReference color_attachment_ref = VkAttachmentReference {
        //attachment number will index into the pAttachments array in the parent renderpass itself
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    // create one subpass
    VkSubpassDescription subpass = VkSubpassDescription {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref
    };

    // UNDEFINED -> RenderPass Begins -> Subpass 0 begins (Transition to
    // Attachment Optimal) -> Subpass 0 renders -> Subpass 0 ends -> Renderpass
    // Ends (Transitions to Present Source)


    VkRenderPassCreateInfo render_pass_info = VkRenderPassCreateInfo {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr
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
    for (int i = 0; i < swapchain_imagecount; ++i)
    {
        framebuffer_info.pAttachments = &_swapchainImageViews[i];
        VK_CHECK(vkCreateFramebuffer(_device, &framebuffer_info, nullptr, &_framebuffers[i]));
    }
}

void VulkanEngine::update_rendersize()
{
    int w,h;
    SDL_GetWindowSizeInPixels(_window, &w, &h);
    _renderExtent = {static_cast<uint32_t>(w),static_cast<uint32_t>(h)};
}
