module;
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_raii.hpp>

export module vmaRaii;

export namespace vma {

	class Allocator
	{
	public:
		Allocator() = default;
		Allocator(const VmaAllocatorCreateInfo& allocInfo);
		~Allocator();
		Allocator(Allocator&& other) noexcept;
		Allocator& operator=(Allocator&& other) noexcept;
		VmaAllocator operator*() const;

	private:
		VmaAllocator _allocator = nullptr;
	};

	class Buffer
	{
	public:
		Buffer() = default;
		Buffer(VmaAllocator alloc, const VkBufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocInfo);
		~Buffer();
		Buffer(Buffer&& other) noexcept;
		Buffer& operator=(Buffer&& other) noexcept;
		VkBuffer operator*() const;

	private:
		VmaAllocator _alloc       = nullptr;
		VkBuffer _buff            = VK_NULL_HANDLE;
		VmaAllocation _allocation = nullptr;
	};

	class Image
	{
	public:
		Image() = default;
		Image(VmaAllocator alloc, const VkImageCreateInfo& imgInfo, const VmaAllocationCreateInfo& allocInfo);
		Image(Image&& other) noexcept;
		Image& operator=(Image&& other) noexcept;
		VkImage operator*() const;
		~Image();

	private:
		VmaAllocator _alloc       = nullptr;
		VkImage _img              = VK_NULL_HANDLE;
		VmaAllocation _allocation = nullptr;
	};

} // namespace vma
