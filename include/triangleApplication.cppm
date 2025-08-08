module;
#include <GLFW/glfw3.h>
#include <cstdint>
#include <vector>
#include <fstream>
#include <memory>
#include <vulkan/vulkan_raii.hpp>

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

	void drawFrame();

	void initVulkan();
	void initWindow();
	void createInstance();
	void setupDebugMessenger();
	void createSurface();
	void createSwapChain();
	void cleanupSwapChain();
	void recreateSwapChain();
	void createImageViews();
	void createGraphicsPipeline();
	void createCommandPool();
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
	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;
	static vk::Format chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
	void pickPhysicalDevice();
	void createLogicalDevice();
	uint32_t findQueueFamilies(VkPhysicalDevice device);
	std::vector<const char*> getRequiredExtensions();
	void mainLoop();
	void cleanup();


	static std::vector<char> readFile(const std::string& filename);

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
	                                                      vk::DebugUtilsMessageTypeFlagsEXT type,
	                                                      const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
	                                                      void*);
};
