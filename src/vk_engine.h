// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <_types/_uint32_t.h>
#include <vector>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

struct FrameData {
	VkCommandPool _commandPool;
	VkCommandBuffer _commandBuffer;
	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;
};
constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	bool useValidationLayers {true};
	VkExtent2D _windowExtent{ 1700 , 900 };
	VkExtent2D _renderExtent;

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debugMessager;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	// swapchain related
	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	// command buffer related
	FrameData _frames[FRAME_OVERLAP];
	FrameData& get_current_frame() {return _frames[_frameNumber % FRAME_OVERLAP];}
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	// render pass related
	VkRenderPass _renderPass;
	std::vector<VkFramebuffer> _framebuffers;

	struct SDL_Window* _window{ nullptr };

	static VulkanEngine& Get();

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();

	void init_default_renderpass();
	void init_framebuffers();

	void update_rendersize();
};
