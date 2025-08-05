#include <vulkan/vulkan_structs.hpp>
module;
#define GLFW_INCLUDE_VULKAN
// #define VULKAN_HPP_NO_EXCEPTIONS

#include <print>
#include <map>
#include <vector>
#include <iostream>
#include <exception>
#include <vulkan/vulkan_raii.hpp>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

module triangleApplication;

void HelloTriangleApplication::initVulkan() {
	createInstance();
	setupDebugMessenger();
	pickPhysicalDevice();
}

void HelloTriangleApplication::initWindow() {
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	// trying to fix wayland transparent windows when no surface
	glfwWindowHint(GLFW_ALPHA_BITS, 0);
	glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
}

void HelloTriangleApplication::createInstance() {
	// list available extensions
	auto availableExtensions = context.enumerateInstanceExtensionProperties();
	std::println("Available extensions:");
	for (const auto& extension: availableExtensions)
		std::cout << "\t" << extension.extensionName << "\n";

	// app Information
	constexpr vk::ApplicationInfo appInfo{
			.pApplicationName	  = "Hello Triangle",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName	      = "No Engine",
			.engineVersion	    = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion	        = vk::ApiVersion14
	};

	// get required layers
	std::vector<char const*> requiredLayers;
	if (enableValidationLayers) {
		requiredLayers.assign(validationLayers.begin(), validationLayers.end());
	}
	// check if required layers are supported by the Vulkan implementation
	auto layerProperties = context.enumerateInstanceLayerProperties();
	if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
							return std::ranges::none_of(layerProperties, [requiredLayer](auto const& layerProperty)
							{ return strcmp(layerProperty.layerName, requiredLayer) == 0;});
    }))
	{
		throw std::runtime_error("One or more required layers are not supported!");
	}

	auto extensions = getRequiredExtensions();
	auto extensionProperties = context.enumerateInstanceExtensionProperties();
	for (auto const& extension: extensions) {
		if (std::ranges::none_of(extensionProperties, [extension](auto const& extensionProperty)
			{return strcmp(extensionProperty.extensionName, extension) == 0;}))
		{
			throw std::runtime_error("Required extension not supported: " + std::string(extension));
		}
	}
	for (uint32_t i = 0; i < extensions.size(); ++i)
	{
		println("{}", std::string(extensions[i]));
		if (std::ranges::none_of(extensionProperties,
		                         [extension= extensions[i]](auto const& extensionProperty)
			                     { return strcmp(extensionProperty.extensionName, extension) == 0;}))
		{
			throw std::runtime_error("Required GLFW extension not supported: " + std::string(extensions[i]));
		}
	}
	vk::InstanceCreateInfo createInfo {
		.pApplicationInfo        = &appInfo,
		.enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
		.ppEnabledLayerNames     = requiredLayers.data(),
		.enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
		.ppEnabledExtensionNames = extensions.data()
	};
	instance = vk::raii::Instance(context, createInfo);
}

void HelloTriangleApplication::setupDebugMessenger() {
	if (!enableValidationLayers)
		return;

	vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
	vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
	vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
		.messageSeverity = severityFlags,
		.messageType = messageTypeFlags,
		.pfnUserCallback = &debugCallback
	};
	debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void HelloTriangleApplication::pickPhysicalDevice() {
	auto devices = instance.enumeratePhysicalDevices();
	if (devices.empty())
		throw std::runtime_error("failed to find GPUs with Vulkan Support!");

	const auto devIter = std::ranges::find_if(devices,
		[&](vk::raii::PhysicalDevice& device) {
			auto queueFamilies = device.getQueueFamilyProperties();
			bool isSuitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;
			const auto qfpIter = std::ranges::find_if(queueFamilies,
				[](vk::QueueFamilyProperties const& qfp)
				{
					return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
				});
			isSuitable = isSuitable && (qfpIter != queueFamilies.end());
			auto extensions = device.enumerateDeviceExtensionProperties();
			bool found = true;
			for (auto const& extension: deviceExtensions) {
				auto extensionIter = std::ranges::find_if(extensions, [extension](auto const& ext) {return strcmp(ext.extensionName, extension) == 0;});
				found = found && extensionIter != extensions.end();
			}
			isSuitable = isSuitable && found;
			std::println("");
			if (isSuitable)
				physicalDevice = device;
			return isSuitable;
		}
	);
	if (devIter == devices.end()) {
		throw std::runtime_error("Failed to find a suitable GPU");
	}

	// -- Select best based on score. For now keeping tutorial version
	// std::multimap<int, vk::raii::PhysicalDevice> candidates;
	// for (const auto& device : devices) {
	// 	auto deviceProperties = device.getProperties();
	// 	auto deviceFeatures = device.getFeatures();
	// 	uint32_t score = 0;
	// 	if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
	// 		score += 1000;
	// 	score += deviceProperties.limits.maxImageDimension2D;
	// 	if (!deviceFeatures.geometryShader)
	// 		continue;
	// 	candidates.insert(std::make_pair(score, device));
	// }
	// if (candidates.rbegin()->first > 0) {
	// 	physicalDevice = candidates.rbegin()->second;
	// } else {
	// 	throw std::runtime_error("Failed to find a suitable GPU!");
	// }
}

void HelloTriangleApplication::createLogicalDevice() {
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

	float queuePriority = 0.0f;
	vk::PhysicalDeviceFeatures deviceFeatures;
	vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
		{},
		{.dynamicRendering = true},
		{.extendedDynamicState = true}
	};
	vk::DeviceQueueCreateInfo { .queueFamilyIndex = graphicsIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};

}

uint32_t HelloTriangleApplication::findQueueFamilies(VkPhysicalDevice device) {
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
	auto graphicsQueueFamilyProperty = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
		[](vk::QueueFamilyProperties const& qfp) {return qfp.queueFlags & vk::QueueFlagBits::eGraphics;});
	return static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
}

std::vector<const char *> HelloTriangleApplication::getRequiredExtensions() {
	uint32_t glfwExtensionCount = 0;
	auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
	if (enableValidationLayers) {
		extensions.push_back(vk::EXTDebugUtilsExtensionName);
	}

	return extensions;
}

void HelloTriangleApplication::mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
	}
}
void HelloTriangleApplication::cleanup() {
	glfwDestroyWindow(window);
	glfwTerminate();
}


// -- Debug Callback --
VKAPI_ATTR vk::Bool32 VKAPI_CALL HelloTriangleApplication::debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
	   vk::DebugUtilsMessageTypeFlagsEXT type,
	   const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
	   void*)
{
	std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
	return vk::False;
}
