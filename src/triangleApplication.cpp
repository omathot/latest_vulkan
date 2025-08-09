module;
#define GLFW_INCLUDE_VULKAN
// #define VULKAN_HPP_NO_EXCEPTIONS

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <print>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <chrono>

module triangleApplication;

void HelloTriangleApplication::initVulkan() {
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createImageViews();
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createCommandPool();
	createVertexBuffer();
	createIndexBuffer();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffer();
	createSyncObjects();
}

void HelloTriangleApplication::initWindow() {
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
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


void HelloTriangleApplication::createSurface() {
	VkSurfaceKHR _surface;
	if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
		throw std::runtime_error("Failed to create window surface!");
	}
	surface = vk::raii::SurfaceKHR(instance, _surface);
}

void HelloTriangleApplication::createSwapChain() {
	auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	swapChainImageFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
	swapChainExtent = chooseSwapExtent(surfaceCapabilities);
	auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
	minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount)
					? surfaceCapabilities.maxImageCount
					: minImageCount;

	uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
	if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
		imageCount = surfaceCapabilities.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR swapChainCreateInfo {
		.flags = vk::SwapchainCreateFlagsKHR(),
		.surface = surface,
		.minImageCount = minImageCount,
		.imageFormat = swapChainImageFormat,
		.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
		.imageExtent = swapChainExtent,
		.imageArrayLayers = 1,
		.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
		.imageSharingMode = vk::SharingMode::eExclusive,
		.preTransform = surfaceCapabilities.currentTransform,
		.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
		.presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)),
		.clipped = true,
		.oldSwapchain = nullptr
	};
	swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
	swapChainImages = swapChain.getImages();
}

void HelloTriangleApplication::recreateSwapChain() {
	int width, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 && height == 0) {
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}
	device.waitIdle();
	cleanupSwapChain();

	createSwapChain();
	createImageViews();
}

void HelloTriangleApplication::cleanupSwapChain() {
	swapChainImageViews.clear();
	swapChain = nullptr;
}

void HelloTriangleApplication::createImageViews() {
	swapChainImageViews.clear();

	vk::ImageViewCreateInfo imageViewCreateInfo{
		.viewType = vk::ImageViewType::e2D,
		.format = swapChainImageFormat,
		.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};

	for (auto image: swapChainImages) {
		imageViewCreateInfo.image = image;
		swapChainImageViews.emplace_back(device, imageViewCreateInfo);
	}
}

void HelloTriangleApplication::createDescriptorSetLayout() {
	vk::DescriptorSetLayoutBinding uboLayoutBinding = {
		.binding = 0,
		.descriptorType = vk::DescriptorType::eUniformBuffer,
		.descriptorCount = 1,
		.stageFlags = vk::ShaderStageFlagBits::eVertex,
		.pImmutableSamplers = nullptr,
	};
	vk::DescriptorSetLayoutCreateInfo layoutInfo = {
		.bindingCount = 1,
		.pBindings = &uboLayoutBinding,
	};
	descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void HelloTriangleApplication::createGraphicsPipeline() {
	vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));
	vk::PipelineShaderStageCreateInfo vertShaderStageInfo {
		.stage = vk::ShaderStageFlagBits::eVertex,
		.module = shaderModule,
		.pName = "vertMain",
	};
	vk::PipelineShaderStageCreateInfo fragShaderStageInfo {
		.stage = vk::ShaderStageFlagBits::eFragment,
		.module = shaderModule,
		.pName = "fragMain",
	};
	vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescriptions();
	vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &bindingDescription,
		.vertexAttributeDescriptionCount = attributeDescriptions.size(),
		.pVertexAttributeDescriptions = attributeDescriptions.data(),
	};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly {.topology = vk::PrimitiveTopology::eTriangleList};
	std::vector dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor,
	};
	vk::PipelineDynamicStateCreateInfo dynamicState{
		.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data()
	};
	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = vk::False,
		.depthBiasClamp = vk::False,
		.depthBiasSlopeFactor = 1.0f,
		.lineWidth = 1.0f,
	};
	vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

	vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		.blendEnable = false,
		.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
		.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		.colorBlendOp = vk::BlendOp::eAdd,
		.srcAlphaBlendFactor = vk::BlendFactor::eOne,
		.dstAlphaBlendFactor = vk::BlendFactor::eZero,
		.alphaBlendOp = vk::BlendOp::eAdd,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |vk::ColorComponentFlagBits::eB |vk::ColorComponentFlagBits::eA,
	};
	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable = vk::False,
		.logicOp = vk::LogicOp::eCopy,
		.attachmentCount = 1,
		.pAttachments = &colorBlendAttachment
	};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount = 1,
		.pSetLayouts = &*descriptorSetLayout,
		.pushConstantRangeCount = 0
	};
	pipelineLayout = vk::raii::PipelineLayout{device, pipelineLayoutInfo};

	vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainImageFormat};
	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.pNext = &pipelineRenderingCreateInfo,
		.stageCount = 2,
		.pStages = shaderStages,
		.pVertexInputState = &vertexInputInfo,
		.pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pColorBlendState = &colorBlending,
		.pDynamicState = &dynamicState,
		.layout = pipelineLayout,
		.renderPass = nullptr, // dynamic rendering
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1,
	};
	graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}

void HelloTriangleApplication::createSyncObjects() {
	presentCompleteSemaphores.clear();
	renderFinishedSemaphores.clear();
	inFlightFences.clear();
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		inFlightFences.emplace_back(device, vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
	}
	for (size_t i = 0; i < swapChainImages.size(); i++) {
		presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
		renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
	}
}

void HelloTriangleApplication::createCommandPool() {
	vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = graphicsIndex};
	commandPool = vk::raii::CommandPool(device, poolInfo);
}

void HelloTriangleApplication::createVertexBuffer() {
	vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

	// only use "host" visible buffer as temporary buffer (staging) - perf boost
	vk::raii::DeviceMemory stagingBufferMemory = nullptr;
	vk::raii::Buffer stagingBuffer = nullptr;
	createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer,
		stagingBufferMemory
	);

	void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, vertices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vertexBuffer,
		vertexBufferMemory
	);
	copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
}

void HelloTriangleApplication::createIndexBuffer() {
	vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

	vk::raii::DeviceMemory stagingBufferMemory = nullptr;
	vk::raii::Buffer stagingBuffer = nullptr;
	createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer,
		stagingBufferMemory
	);

	void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, indices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		indexBuffer,
		indexBufferMemory
	);
	copyBuffer(stagingBuffer, indexBuffer, bufferSize);
}

void HelloTriangleApplication::createUniformBuffers() {
	uniformBuffers.clear();
	uniformBuffersMemory.clear();
	uniformBuffersMapped.clear();

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
		vk::raii::Buffer buffer = nullptr;
		vk::raii::DeviceMemory bufferMemory = nullptr;
		createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			buffer,
			bufferMemory
		);
		uniformBuffers.emplace_back(std::move(buffer));
		uniformBuffersMemory.emplace_back(std::move(bufferMemory));
		uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
	}
}


void HelloTriangleApplication::createDescriptorPool() {
	vk::DescriptorPoolSize poolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT);
	vk::DescriptorPoolCreateInfo poolInfo = {
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = MAX_FRAMES_IN_FLIGHT,
		.poolSizeCount = 1,
		.pPoolSizes = &poolSize
	};
	descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void HelloTriangleApplication::createDescriptorSets() {
	descriptorSets.clear();
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo = {
		.descriptorPool = descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};
	descriptorSets = device.allocateDescriptorSets(allocInfo);
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::DescriptorBufferInfo bufferInfo = {
			.buffer = uniformBuffers[i],
			.offset = 0,
			.range = sizeof(UniformBufferObject)
		};
		vk::WriteDescriptorSet descriptorWrite = {
			.dstSet = descriptorSets[i],
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eUniformBuffer,
			.pBufferInfo = &bufferInfo,
		};
		device.updateDescriptorSets(descriptorWrite, {});
	}
}

void HelloTriangleApplication::createBuffer(
	vk::DeviceSize size,
	vk::BufferUsageFlags usage,
	vk::MemoryPropertyFlags properties,
	vk::raii::Buffer& buffer,
	vk::raii::DeviceMemory& bufferMemory
) {
	vk::BufferCreateInfo bufferInfo{
		.size = size,
		.usage = usage,
		.sharingMode = vk::SharingMode::eExclusive,
	};
	buffer = vk::raii::Buffer(device, bufferInfo);
	vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
	vk::MemoryAllocateInfo memoryAllocateInfo = {
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
	};
	bufferMemory = vk::raii::DeviceMemory(device, memoryAllocateInfo);
	buffer.bindMemory(bufferMemory, 0);
}

void HelloTriangleApplication::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
	vk::CommandBufferAllocateInfo allocInfo = {
		.commandPool = commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1,
	};
	vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
	commandCopyBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
	commandCopyBuffer.end();
	graphicsQueue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
	graphicsQueue.waitIdle();
}

void HelloTriangleApplication::updateUniformBuffers(uint32_t currentImage) {
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	UniformBufferObject ubo{};
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(
		glm::radians(45.0f),
		static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
		0.1f,
		10.f
	);
	ubo.proj[1][1] *= -1;
	memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void HelloTriangleApplication::createCommandBuffer() {
	commandBuffers.clear();
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = MAX_FRAMES_IN_FLIGHT,
	};
	commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void HelloTriangleApplication::recordCommandBuffer(uint32_t imageIndex) {
	commandBuffers[currentFrame].begin({});
	transitionImageLayout(
		imageIndex,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput
	);

	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	vk::RenderingAttachmentInfo attachmentInfo = {
		.imageView = swapChainImageViews[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor
	};
	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo
	};
	commandBuffers[currentFrame].beginRendering(renderingInfo);
	commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
	commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, {0});
	commandBuffers[currentFrame].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
	commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);

	// set dynamic states
	commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

	commandBuffers[currentFrame].drawIndexed(indices.size(), 1, 0, 0, 0);
	// commandBuffers[currentFrame].draw(vertices.size(), 1, 0, 0);
	commandBuffers[currentFrame].endRendering();
	transitionImageLayout(
		imageIndex,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		{},
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eBottomOfPipe
	);
	commandBuffers[currentFrame].end();
}

void HelloTriangleApplication::transitionImageLayout(
		uint32_t imageIndex,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::AccessFlags2 srcAccessMask,
		vk::AccessFlags2 dstAccessMask,
		vk::PipelineStageFlags2 srcStageMask,
		vk::PipelineStageFlags2 dstStageMask
) {
	vk::ImageMemoryBarrier2 barrier = {
		.srcStageMask = srcStageMask,
		.srcAccessMask = srcAccessMask,
		.dstStageMask = dstStageMask,
		.dstAccessMask = dstAccessMask,
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = swapChainImages[imageIndex],
		.subresourceRange = {
			.aspectMask = vk::ImageAspectFlagBits::eColor,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	vk::DependencyInfo dependencyInfo = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier,
	};
	commandBuffers[currentFrame].pipelineBarrier2(dependencyInfo);
}

vk::raii::ShaderModule HelloTriangleApplication::createShaderModule(const std::vector<char>& code) const {
	vk::ShaderModuleCreateInfo shaderModuleCreateInfo {
		.codeSize = code.size() * sizeof(char),
		.pCode = reinterpret_cast<const uint32_t*>(code.data()),
	};
	vk::raii::ShaderModule shaderModule{device, shaderModuleCreateInfo};
	return shaderModule;
}

vk::Format HelloTriangleApplication::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return availableFormat.format;
		}
	}
	// could find "next best" but for now whatevs
	return availableFormats[0].format;
}

vk::PresentModeKHR HelloTriangleApplication::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
	for (const auto& availablePresentMode : availablePresentModes) {
		if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
			return availablePresentMode;
		}
	}
	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D HelloTriangleApplication::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilities.currentExtent;
	}
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	return {
		std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
	};
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
			for (auto const& extension: requiredDeviceExtensions) {
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

	auto graphicsQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties, [](auto const& qfp)
													{return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);});
	graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
	auto presentIndex = physicalDevice.getSurfaceSupportKHR(graphicsIndex, *surface)
										? graphicsIndex
										: static_cast<uint32_t>(queueFamilyProperties.size());
	if (presentIndex == queueFamilyProperties.size()) {
		// graphicsIndex doesn't support present -> look for one that supports both graphics and present
		for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
			if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) && physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface)) {
				graphicsIndex = static_cast<uint32_t>(i);
				presentIndex = graphicsIndex;
				break;
			}
		}
	}
	if (presentIndex == queueFamilyProperties.size()) {
		// nothing like a single family index that supports both grahpics and present
		// -> look for another family that supports present
		for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
			if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface)) {
				presentIndex = static_cast<uint32_t>(i);
				break;
			}
		}
	}
	if ((graphicsIndex == queueFamilyProperties.size()) || (presentIndex == queueFamilyProperties.size())) {
		throw std::runtime_error("Could not find a queue for graphics or present -> terminating");
	}

	// query Vulkan 1.3 features
	vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
		{},                                                     // vk::PhysicalDeviceFeatures2
		{.synchronization2 = true, .dynamicRendering = true },  // vk::PhysicalDeviceVulkan13Features
		{.extendedDynamicState = true }                         // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
	};

	// create device
	float queuePriority = 0.0f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = graphicsIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
	vk::DeviceCreateInfo deviceCreateInfo {
		.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &deviceQueueCreateInfo,
		.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size()),
		.ppEnabledExtensionNames = requiredDeviceExtensions.data(),
	};
	device = vk::raii::Device(physicalDevice, deviceCreateInfo);
	graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
	presentQueue = vk::raii::Queue(device, presentIndex, 0);
	std::println("graphicsIndex = {}, presentIndex = {}", graphicsIndex, presentIndex);
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

uint32_t HelloTriangleApplication::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
	vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	throw std::runtime_error("Failed to find suitable memory type!");
}

void HelloTriangleApplication::mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		drawFrame();
	}
	device.waitIdle();
}
void HelloTriangleApplication::cleanup() {
	cleanupSwapChain();

	glfwDestroyWindow(window);
	glfwTerminate();
}

void HelloTriangleApplication::drawFrame() {
	while (vk::Result::eTimeout == device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX)); // you can do that??
	auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[semaphoreIndex], nullptr);
	if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
		frameBufferResized = false;
		recreateSwapChain();
		return;
	}
	if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
		throw std::runtime_error("failed to  acquire swap chain image!");
	}
	device.resetFences(*inFlightFences[currentFrame]);

	commandBuffers[currentFrame].reset();
	recordCommandBuffer(imageIndex);

	vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
	updateUniformBuffers(currentFrame);
	const vk::SubmitInfo submitInfo{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*presentCompleteSemaphores[semaphoreIndex],
		.pWaitDstStageMask = &waitDestinationStageMask,
		.commandBufferCount = 1,
		.pCommandBuffers = &*commandBuffers[currentFrame],
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &*renderFinishedSemaphores[imageIndex],
	};
	graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);
	const vk::PresentInfoKHR presentInfoKHR{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
		.swapchainCount = 1,
		.pSwapchains = &*swapChain,
		.pImageIndices = &imageIndex,
		.pResults = nullptr, // optional
	};
	result = presentQueue.presentKHR(presentInfoKHR);
	if (result == vk::Result::eErrorOutOfDateKHR) {
		frameBufferResized = false;
		recreateSwapChain();
	}
	if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
		throw std::runtime_error("failed to  acquire swap chain image!");
	}
	semaphoreIndex = (semaphoreIndex + 1) % presentCompleteSemaphores.size();
	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}


std::vector<char> HelloTriangleApplication::readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file!");
	}
	std::vector<char> buffer(file.tellg());
	file.seekg(0, std::ios::beg);
	file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
	file.close();
	return buffer;
}

void HelloTriangleApplication::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
	app->frameBufferResized = true;
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


// shader
vk::VertexInputBindingDescription Vertex::getBindingDescription() {
	return {.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};
}

std::array<vk::VertexInputAttributeDescription, 2> Vertex::getAttributeDescriptions() {
	return {
		vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
		vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
	};
}
