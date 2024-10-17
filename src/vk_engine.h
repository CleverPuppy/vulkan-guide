// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <_types/_uint32_t.h>
#include <deque>
#include <functional>
#include <stddef.h>
#include <sys/sysctl.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "vk_mem_alloc.h"
#include "vk_types.h"
#include <vulkan/vulkan_core.h>

#include "mesh.h"
#include "camera.h"
#include "glm/glm.hpp"

struct FrameData {
	VkCommandPool _commandPool;
	VkCommandBuffer _commandBuffer;
	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;
	AllocatedBuffer _cameraBuffer;
	AllocatedBuffer _objectBuffer;
	VkDescriptorSet _globalDescriptor;
	VkDescriptorSet _objectDescriptor;
};
constexpr unsigned int FRAME_OVERLAP = 2;

struct GPUSceneData {
	glm::vec4 fogColor;
	glm::vec4 forDistance;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;
};

struct GPUObjectData {
	glm::mat4 modelMatrix;
};

class DeleteQueue {
public:
	template<typename T>
	void push_function(T&& function)
	{
		_deletors.emplace_back(std::forward<T>(function));
	}

	void flush()
	{
		for (auto it = _deletors.rbegin(); it != _deletors.rend(); ++it)
		{
			(*it)();
		}

		_deletors.clear();
	}
private:
	std::deque<std::function<void()>> _deletors;
};

struct MeshPushConstants {
	glm::vec4 data;
	glm::mat4 render_matrix;
};

struct Material {
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject {
	Mesh* mesh;
	Material* material;
	glm::mat4 transformMatrix;
};

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
	VkPhysicalDeviceProperties _gpuProperties;
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

	// pipelines
	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;
	VkPipeline _coloredTrianglePipeline;

	// vma allocator
	VmaAllocator _allocator;

	// mesh related
	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;
	Mesh _triangleMesh;
	Mesh _monkeyMesh;

	// depth image
	VkFormat _depthImageFormat;
	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	// resource delete queue
	DeleteQueue _mainDeleteQueue;

	// render objects
	std::vector<RenderObject> _renderables;
	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;

	// descriptor sets related
	VkDescriptorPool _descriptorPool;
	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
	// camera related
	Camera _camera;

	// scene related data
	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;

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

	int _selectShader {0};
	bool _resize_requested {false};

	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_depth_image();
	void destroy_depth_image();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();

	void init_default_renderpass();
	void init_framebuffers();
	void destroy_framebuffers();

	void update_rendersize();
	void handle_window_resize();

	bool load_shader_module(const char* filepath, VkShaderModule* outShaderModule);
	void init_pipelines();

	void on_window_resize(int width, int height);

	void load_meshs();
	void upload_mesh(Mesh& mesh);

	void init_scene();
	void init_camera();

	void init_descriptors();

	//create material and add it to the map
	Material* create_material(VkPipeline pipeline, VkPipelineLayout layout,const std::string& name);

	//returns nullptr if it can't be found
	Material* get_material(const std::string& name);

	//returns nullptr if it can't be found
	Mesh *get_mesh(const std::string& name);

	//our draw function
	void draw_objects(VkCommandBuffer cmd,RenderObject* first, int count);

	// helper functions
	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	size_t pad_uniform_buffer_size(size_t origSize);
};
