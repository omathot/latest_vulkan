set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

cmake_minimum_required(VERSION 3.31)
project(learnvulkan VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)

find_package(glm REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE MODULE_SOURCES "include/*.cppm")

find_program(SLANGC_EXECUTABLE
    NAMES slangc
    PATHS
        /usr/local/bin
        /opt/slang/bin
        /usr/bin
    DOC "Slang shader compiler"
    REQUIRED
)
set(SHADER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)
# slang compilation function
function (add_slang_shader_target TARGET)
  cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})
  set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
  set (ENTRY_POINTS -entry vertMain -entry fragMain)
  add_custom_command (
          OUTPUT ${SHADERS_DIR}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )
  add_custom_command (
          OUTPUT  ${SHADERS_DIR}/slang.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
          WORKING_DIRECTORY ${SHADERS_DIR}
          DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )
  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()
add_slang_shader_target(${PROJECT_NAME}_shaders SOURCES ${SHADER_SOURCES})

# set up Vulkan C++ module
add_library(VulkanCppModule)
add_library(Vulkan::cppm ALIAS VulkanCppModule)

target_compile_definitions(VulkanCppModule PUBLIC
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)
target_include_directories(VulkanCppModule
        PRIVATE
        "${Vulkan_INCLUDE_DIR}"
)
target_link_libraries(VulkanCppModule
        PUBLIC
        Vulkan::Vulkan
)

set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 23)

target_sources(VulkanCppModule
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS
        "${Vulkan_INCLUDE_DIR}"
        FILES
        "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
)


# Download CPM
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
CPMAddPackage(
    NAME VulkanMemoryAllocator
    GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v3.3.0
    DOWNLOAD_ONLY YES
)
if(VulkanMemoryAllocator_ADDED)
        add_library(VulkanMemoryAllocator INTERFACE)
        target_include_directories(VulkanMemoryAllocator INTERFACE ${VulkanMemoryAllocator_SOURCE_DIR}/include)
        target_compile_definitions(VulkanMemoryAllocator INTERFACE
                VMA_STATIC_VULKAN_FUNCTIONS=0
                VMA_DYNAMIC_VULKAN_FUNCTIONS=1
        )
        target_link_libraries(VulkanMemoryAllocator INTERFACE Vulkan::Vulkan)
endif()

add_library(TriangleAppModule)
add_library(TriangleApp::module ALIAS TriangleAppModule)
set_target_properties(TriangleAppModule PROPERTIES CXX_STANDARD 23)
target_sources(TriangleAppModule
  PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES
  BASE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  FILES
  ${MODULE_SOURCES}
)
target_link_libraries(TriangleAppModule PUBLIC VulkanMemoryAllocator)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GLFW3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
        TriangleApp::module
        Vulkan::cppm
        ${GLFW3_LIBRARIES}
        glm::glm
        VulkanMemoryAllocator
)

target_compile_options(${PROJECT_NAME} PRIVATE ${GLFW3_CFLAGS_OTHER})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_shaders)
