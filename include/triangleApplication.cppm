module;
#define VMA_IMPLEMENTATION
#define VMA_VULKAN_VERSION 1004000

#ifndef NDEBUG
#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#define VMA_DEBUG_DETECT_CORRUPTION 1
#endif

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#include <vulkan/vulkan_core.h>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <vector>

export module triangleApplication;

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr const char* TEXTURE_LOCATION = "textures/sick-EP.jpg"; // lagtrain.jpg sick-EP.jpg texture.jpg

std::vector validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};
std::vector<const char*> requiredDeviceExtensions = {
	vk::KHRSwapchainExtensionName,
	vk::KHRSpirv14ExtensionName,
	vk::KHRSynchronization2ExtensionName,
	vk::KHRCreateRenderpass2ExtensionName,

	// needed for m1mac asahi linux
	vk::KHRShaderDrawParametersExtensionName
};
#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription();
	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions();
};
// alignment spec: https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-resources-layout
struct UniformBufferObject {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

// shader data
const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f},  {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f},   {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f},  {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};
const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0
};

struct VMABuffer {
	VkBuffer buffer;
	VmaAllocation allocation;

	operator VkBuffer() const {return buffer;}
};
struct VMAImage {
	VkImage image;
	VmaAllocation allocation;

	operator VkImage() const {return image;}
};

export class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window                                = nullptr;
	vk::raii::Context context;

	// VkInstance instance = nullptr;
	vk::raii::Instance instance                      = nullptr;
	// VkDebugUtilsMessengerEXT debugMesseger = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger   = nullptr;
	vk::raii::PhysicalDevice physicalDevice           = nullptr;
	vk::raii::Device device                           = nullptr;
	VmaAllocator allocator                            = nullptr;
	uint32_t graphicsIndex                            = ~0;
	uint32_t currentFrame                             = 0;
	uint32_t semaphoreIndex                           = 0;
	bool frameBufferResized                           = false;
	vk::raii::Queue graphicsQueue                     = nullptr;
	vk::raii::Queue presentQueue                      = nullptr;
	vk::raii::SurfaceKHR surface                      = nullptr;

	vk::raii::SwapchainKHR swapChain                  = nullptr;
	std::vector<VMAImage> swapChainImages;
	vk::Format swapChainImageFormat                   = vk::Format::eUndefined;
	vk::Extent2D swapChainExtent                      = vk::Extent2D::NativeType();
	std::vector<vk::raii::ImageView> swapChainImageViews;

	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout pipelineLayout           = nullptr;
	vk::raii::Pipeline graphicsPipeline               = nullptr;

	vk::raii::CommandBuffer setupCommandBuffer = nullptr;
	vk::raii::DescriptorPool descriptorPool           = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;
	vk::raii::CommandPool commandPool                 = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence> inFlightFences;

	std::vector<VMABuffer> uniformBuffers;
	std::vector<void*> uniformBuffersMapped;
	VMABuffer vertexBuffer;
	VMABuffer indexBuffer;

	VMAImage textureImage;
	vk::raii::ImageView textureImageView = nullptr;
	vk::raii::Sampler textureSampler = nullptr;

	void drawFrame();

	void initWindow();
	void initVulkan();
	void createInstance();
	void setupDebugMessenger();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void initVMA();
	void createSurface();
	void createSwapChain();
	void cleanupSwapChain();
	void recreateSwapChain();
	void createImageViews();
	void createDescriptorSetLayout();
	void createGraphicsPipeline();
	void createCommandPool();
	void createTextureImage();
	void createTextureImageView();
	void createTextureSampler();
	vk::raii::ImageView createImageView(VMAImage& image, vk::Format format);
	void createImage(
		uint32_t width,
		uint32_t height,
		vk::Format format,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		VMAImage& image
	);
	void createVertexBuffer();
	void createIndexBuffer();
	void createUniformBuffers();
	void createDescriptorPool();
	void createDescriptorSets();
	void createBuffer(
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		VMABuffer& buffer
	);
	void createCommandBuffer();
	void createSyncObjects();
	void recordCommandBuffer(uint32_t imageIndex);
	void transition_image_layout(
		uint32_t imageIndex,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::AccessFlags2 srcAccessMask,
		vk::AccessFlags2 dstAccessMask,
		vk::PipelineStageFlags2 srcStageMask,
		vk::PipelineStageFlags2 dstStageMask
	);
	void transitionImageLayout(const VMAImage& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
	void copyBuffer(VMABuffer& srcBuffer, VMABuffer& dstBuffer, vk::DeviceSize size);
	void copyBufferToImage(const VMABuffer& buffer, VMAImage& image, uint32_t width, uint32_t height);
	void updateUniformBuffers(uint32_t currentImage);
	vk::raii::CommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer);
	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;
	static vk::Format chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
	uint32_t findQueueFamilies(VkPhysicalDevice device);
	std::vector<const char*> getRequiredExtensions();
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
	void mainLoop();
	void cleanup();


	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
	static std::vector<char> readFile(const std::string& filename);

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
		vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
		vk::DebugUtilsMessageTypeFlagsEXT type,
		const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void*
	);
};
