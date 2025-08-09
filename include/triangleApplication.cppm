module;
// #define VMA_IMPLEMENTATION
// #define VMA_VULKAN_VERSION 1003000

#include <GLFW/glfw3.h>
#include <cstdint>
#include <vector>
#include <fstream>
#include <memory>
#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#include <vk_mem_alloc.h>

export module triangleApplication;

export constexpr uint32_t WIDTH  = 800;
export constexpr uint32_t HEIGHT = 600;
export constexpr int MAX_FRAMES_IN_FLIGHT = 2;

export std::vector validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};
export std::vector<const char*> requiredDeviceExtensions = {
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

	static vk::VertexInputBindingDescription getBindingDescription();
	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions();
};
struct UniformBufferObject {
	glm::mat4 mode;
	glm::mat4 view;
	glm::mat4 proj;
};

// shader data
const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};
const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0
};
// export const std::vector<Vertex> vertices = {
//     {{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
//     {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
//     {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
// };
// export const std::vector<Vertex> vertices = {
//     {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
//     {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
//     {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
// };

export class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window                              = nullptr;
	vk::raii::Context context;

	vk::raii::Instance instance                     = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::PhysicalDevice physicalDevice         = nullptr;
	vk::raii::Device device                         = nullptr;
	uint32_t graphicsIndex                          = ~0;
	uint32_t currentFrame                           = 0;
	uint32_t semaphoreIndex                         = 0;
	bool frameBufferResized                         = false;
	vk::raii::Queue graphicsQueue                   = nullptr;
	vk::raii::Queue presentQueue                    = nullptr;
	vk::raii::SurfaceKHR surface                    = nullptr;

	vk::raii::SwapchainKHR swapChain                = nullptr;
	std::vector<vk::Image> swapChainImages;
	vk::Format swapChainImageFormat                 = vk::Format::eUndefined;
	vk::Extent2D swapChainExtent                    = vk::Extent2D::NativeType();
	std::vector<vk::raii::ImageView> swapChainImageViews;

	vk::raii::PipelineLayout pipelineLayout         = nullptr;
	vk::raii::Pipeline graphicsPipeline             = nullptr;

	vk::raii::CommandPool commandPool               = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence> inFlightFences;


	vk::raii::Buffer vertexBuffer                   = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory       = nullptr;
	vk::raii::Buffer indexBuffer                    = nullptr;
	vk::raii::DeviceMemory indexBufferMemory        = nullptr;

	void drawFrame();

	void initWindow();
	void initVulkan();
	void createInstance();
	void setupDebugMessenger();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void createSurface();
	void createSwapChain();
	void cleanupSwapChain();
	void recreateSwapChain();
	void createImageViews();
	void createDescriptorSetLayout();
	void createGraphicsPipeline();
	void createCommandPool();
	void createVertexBuffer();
	void createIndexBuffer();
	void createBuffer(
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		vk::raii::Buffer& buffer,
		vk::raii::DeviceMemory& deviceMemory
	);
	void createCommandBuffer();
	void createSyncObjects();
	void recordCommandBuffer(uint32_t imageIndex);
	void transitionImageLayout(
		uint32_t imageIndex,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::AccessFlags2 srcAccessMask,
		vk::AccessFlags2 dstAccessMask,
		vk::PipelineStageFlags2 srcStageMask,
		vk::PipelineStageFlags2 dstStageMask
	);
	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);
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

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
	                                                      vk::DebugUtilsMessageTypeFlagsEXT type,
	                                                      const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
	                                                      void*);
};
