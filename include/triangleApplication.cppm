module;
#include <GLFW/glfw3.h>
#include <cstdint>
#include <vector>
#include <memory>
#include <vulkan/vulkan_raii.hpp>

export module triangleApplication;

export constexpr uint32_t WIDTH  = 800;
export constexpr uint32_t HEIGHT = 600;

export std::vector validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};
export std::vector<const char*> requiredDeviceExtensions = {
	vk::KHRSwapchainExtensionName,
	vk::KHRSpirv14ExtensionName,
	vk::KHRSynchronization2ExtensionName,
	vk::KHRCreateRenderpass2ExtensionName,
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
	GLFWwindow*        window                       = nullptr;
	vk::raii::Context  context;
	vk::raii::Instance instance                     = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::PhysicalDevice physicalDevice         = nullptr;
	vk::raii::Device device                         = nullptr;
	vk::raii::Queue graphicsQueue                   = nullptr;
	vk::raii::Queue presentQueue                    = nullptr;
	vk::raii::SurfaceKHR surface                    = nullptr;
	vk::raii::SwapchainKHR swapChain                = nullptr;
	std::vector<vk::Image> swapChainImages;
	vk::Format swapChainImageFormat                 = vk::Format::eUndefined;
	vk::Extent2D swapChainExtent                    = vk::Extent2D::NativeType();

	void initVulkan();
	void initWindow();
	void createInstance();
	void setupDebugMessenger();
	void createSurface();
	void createSwapChain();
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
	void pickPhysicalDevice();
	void createLogicalDevice();
	uint32_t findQueueFamilies(VkPhysicalDevice device);
	std::vector<const char*> getRequiredExtensions();
	void mainLoop();
	void cleanup();

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
	                                                      vk::DebugUtilsMessageTypeFlagsEXT type,
	                                                      const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
	                                                      void*);
};
