#include <exception>
#include <print>

import vulkan_hpp;
import triangleApplication;
import vmaRaii;


int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::println("{}", e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
