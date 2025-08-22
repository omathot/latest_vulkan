module;
#include <stdexcept>
#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
module vmaRaii;

namespace vma {

	Allocator::Allocator(const VmaAllocatorCreateInfo& createInfo)
	{
		VkResult result = vmaCreateAllocator(&createInfo, &_allocator);
		if (result != VK_SUCCESS)
			throw std::runtime_error("Failed to create vmaAllocator");
	}
	Allocator::~Allocator()
	{
		if (_allocator != nullptr)
			vmaDestroyAllocator(_allocator);
	}
	Allocator::Allocator(Allocator&& other) noexcept : _allocator(other._allocator)
	{
		other._allocator = nullptr;
	}
	Allocator& Allocator::operator=(Allocator&& other) noexcept
	{
		if (this != &other)
		{
			if (_allocator != nullptr)
				vmaDestroyAllocator(_allocator);
			_allocator = other._allocator;
			other._allocator = nullptr;
		}
		return *this;
	}
	VmaAllocator Allocator::operator*() const
	{
		return _allocator;
	}

	Buffer::Buffer(VmaAllocator alloc, const VkBufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocInfo) : _alloc(alloc)
	{
		VkResult result = vmaCreateBuffer(_alloc, &bufferInfo, &allocInfo, &_buff, &_allocation, nullptr);
		if (result != VK_SUCCESS)
			throw std::runtime_error("Failed to create vmaBuffer");
	}
	Buffer::~Buffer()
	{
		if (_buff != VK_NULL_HANDLE)
			vmaDestroyBuffer(_alloc, _buff, _allocation);
	}

} // namespace vma
