# My project to learn Vulkan

Project to learn Vulkan >= 1.4. First by following the [latest_vulkan tutorial](https://docs.vulkan.org/tutorial/latest/00_Introduction.html) and then implementing their recommended improvements on top.

I try to keep as many Vk objects raii as possible (vk::raii), but I integrate [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) for memory allocations. Meaning this project creates a thin wrapper around VkBuffer and VmaAllocation to make a VMABuffer that can be used to leverage VMA's features.

