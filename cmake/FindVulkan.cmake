# FindVulkan.cmake - Find Vulkan SDK headers and libraries
#
# This module defines:
#  Vulkan_FOUND - True if Vulkan SDK is found
#  Vulkan_INCLUDE_DIRS - Include directories for Vulkan headers
#  Vulkan_LIBRARIES - List of libraries when using Vulkan
#  Vulkan_LIBRARY - Vulkan library
#  Vulkan_SDK_ROOT - Root directory of Vulkan SDK
#  Vulkan_VERSION - Version of Vulkan SDK
#  Vulkan_API_VERSION - Vulkan API version
#  Vulkan_GLSLC_EXECUTABLE - Path to glslc compiler
#  Vulkan_GLSLANGVALIDATOR_EXECUTABLE - Path to glslangValidator
#  Vulkan_SPIRV_TOOLS_FOUND - True if SPIRV-Tools are available
#  Vulkan_COMPUTE_SUPPORT - True if compute shader support is available

cmake_minimum_required(VERSION 3.18)

# Set default Vulkan SDK root directory candidates
set(Vulkan_SDK_ROOT_CANDIDATES
    $ENV{VULKAN_SDK}
    $ENV{VK_SDK_PATH}
    $ENV{VULKAN_ROOT}
    /usr/local
    /usr
    /opt/vulkan
    /Library/Frameworks/vulkan.framework
    "C:/VulkanSDK/1.3.268.0"
    "C:/VulkanSDK/1.3.261.1"
    "C:/VulkanSDK/1.3.250.1"
    "C:/VulkanSDK/1.3.243.0"
    "C:/VulkanSDK/1.3.236.0"
    "C:/VulkanSDK/1.3.231.1"
    "C:/VulkanSDK/1.3.224.1"
    "C:/VulkanSDK/1.3.216.0"
    "C:/VulkanSDK/1.3.211.0"
    "C:/VulkanSDK/1.3.204.1"
    "C:/VulkanSDK/1.2.198.1"
    "C:/VulkanSDK/1.2.189.2"
    "C:/VulkanSDK/1.2.182.0"
    "C:/VulkanSDK/1.2.176.1"
    "C:/VulkanSDK/1.2.162.1"
    "C:/VulkanSDK/1.2.154.1"
    "C:/VulkanSDK/1.2.148.1"
    "C:/VulkanSDK/1.2.141.2"
    "C:/VulkanSDK/1.2.135.0"
    "C:/VulkanSDK/1.2.131.2"
)

# Platform-specific library and include paths
if(WIN32)
    set(Vulkan_LIB_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/Lib
        ${Vulkan_SDK_ROOT_CANDIDATES}/Lib32
        ${Vulkan_SDK_ROOT_CANDIDATES}/Bin
        ${Vulkan_SDK_ROOT_CANDIDATES}/Bin32
    )
    set(Vulkan_INC_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/Include
    )
    set(Vulkan_BIN_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/Bin
        ${Vulkan_SDK_ROOT_CANDIDATES}/Bin32
    )
elseif(APPLE)
    set(Vulkan_LIB_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/lib
        ${Vulkan_SDK_ROOT_CANDIDATES}/macOS/lib
        /usr/local/lib
        /Library/Frameworks
    )
    set(Vulkan_INC_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/include
        ${Vulkan_SDK_ROOT_CANDIDATES}/macOS/include
        /usr/local/include
        /Library/Frameworks/vulkan.framework/Headers
    )
    set(Vulkan_BIN_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/bin
        ${Vulkan_SDK_ROOT_CANDIDATES}/macOS/bin
        /usr/local/bin
    )
else()
    set(Vulkan_LIB_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/lib
        ${Vulkan_SDK_ROOT_CANDIDATES}/lib64
        ${Vulkan_SDK_ROOT_CANDIDATES}/x86_64/lib
        /usr/lib64
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib64
        /usr/local/lib
    )
    set(Vulkan_INC_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/include
        ${Vulkan_SDK_ROOT_CANDIDATES}/x86_64/include
        /usr/include
        /usr/local/include
    )
    set(Vulkan_BIN_SEARCH_PATHS
        ${Vulkan_SDK_ROOT_CANDIDATES}/bin
        ${Vulkan_SDK_ROOT_CANDIDATES}/x86_64/bin
        /usr/bin
        /usr/local/bin
    )
endif()

# Find Vulkan SDK root directory
find_path(Vulkan_SDK_ROOT
    NAMES include/vulkan/vulkan.h
    PATHS ${Vulkan_SDK_ROOT_CANDIDATES}
    DOC "Vulkan SDK root directory"
)

# Find Vulkan include directory
find_path(Vulkan_INCLUDE_DIR
    NAMES vulkan/vulkan.h
    PATHS ${Vulkan_INC_SEARCH_PATHS}
    DOC "Vulkan include directory"
)

# Find Vulkan library
if(APPLE)
    find_library(Vulkan_LIBRARY
        NAMES 
            vulkan
            MoltenVK
        PATHS ${Vulkan_LIB_SEARCH_PATHS}
        DOC "Vulkan library"
    )
else()
    find_library(Vulkan_LIBRARY
        NAMES 
            vulkan-1
            vulkan
        PATHS ${Vulkan_LIB_SEARCH_PATHS}
        DOC "Vulkan library"
    )
endif()

# Find GLSL compiler
find_program(Vulkan_GLSLC_EXECUTABLE
    NAMES glslc
    PATHS ${Vulkan_BIN_SEARCH_PATHS}
    DOC "Vulkan GLSL compiler (glslc)"
)

# Find GLSL validator
find_program(Vulkan_GLSLANGVALIDATOR_EXECUTABLE
    NAMES glslangValidator
    PATHS ${Vulkan_BIN_SEARCH_PATHS}
    DOC "Vulkan GLSL validator (glslangValidator)"
)

# Determine Vulkan version from SDK
if(Vulkan_SDK_ROOT AND EXISTS "${Vulkan_SDK_ROOT}/include/vulkan/vulkan.h")
    file(READ "${Vulkan_SDK_ROOT}/include/vulkan/vulkan.h" Vulkan_HEADER_CONTENT)
    
    # Extract Vulkan version from header
    string(REGEX MATCH "#define VK_HEADER_VERSION_COMPLETE VK_MAKE_API_VERSION\\(0, ([0-9]+), ([0-9]+), VK_HEADER_VERSION\\)" 
           Vulkan_VERSION_MATCH ${Vulkan_HEADER_CONTENT})
    
    if(Vulkan_VERSION_MATCH)
        set(Vulkan_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(Vulkan_VERSION_MINOR ${CMAKE_MATCH_2})
        set(Vulkan_VERSION "${Vulkan_VERSION_MAJOR}.${Vulkan_VERSION_MINOR}")
    else()
        # Try alternative version detection
        string(REGEX MATCH "#define VK_API_VERSION_1_([0-9]+)" Vulkan_API_MATCH ${Vulkan_HEADER_CONTENT})
        if(Vulkan_API_MATCH)
            set(Vulkan_VERSION_MAJOR 1)
            set(Vulkan_VERSION_MINOR ${CMAKE_MATCH_1})
            set(Vulkan_VERSION "${Vulkan_VERSION_MAJOR}.${Vulkan_VERSION_MINOR}")
        endif()
    endif()
    
    # Extract API version
    string(REGEX MATCH "#define VK_API_VERSION VK_MAKE_API_VERSION\\(0, ([0-9]+), ([0-9]+), 0\\)" 
           Vulkan_API_MATCH ${Vulkan_HEADER_CONTENT})
    
    if(Vulkan_API_MATCH)
        set(Vulkan_API_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    else()
        set(Vulkan_API_VERSION ${Vulkan_VERSION})
    endif()
    
    # Check for compute shader support
    string(FIND ${Vulkan_HEADER_CONTENT} "VK_SHADER_STAGE_COMPUTE_BIT" Vulkan_COMPUTE_FOUND)
    if(Vulkan_COMPUTE_FOUND GREATER -1)
        set(Vulkan_COMPUTE_SUPPORT TRUE)
    else()
        set(Vulkan_COMPUTE_SUPPORT FALSE)
    endif()
endif()

# Check for SPIRV-Tools availability
if(Vulkan_SDK_ROOT)
    find_program(Vulkan_SPIRV_AS
        NAMES spirv-as
        PATHS ${Vulkan_BIN_SEARCH_PATHS}
    )
    
    find_program(Vulkan_SPIRV_DIS
        NAMES spirv-dis
        PATHS ${Vulkan_BIN_SEARCH_PATHS}
    )
    
    find_program(Vulkan_SPIRV_OPT
        NAMES spirv-opt
        PATHS ${Vulkan_BIN_SEARCH_PATHS}
    )
    
    if(Vulkan_SPIRV_AS AND Vulkan_SPIRV_DIS)
        set(Vulkan_SPIRV_TOOLS_FOUND TRUE)
    else()
        set(Vulkan_SPIRV_TOOLS_FOUND FALSE)
    endif()
else()
    set(Vulkan_SPIRV_TOOLS_FOUND FALSE)
endif()

# Create validation test
if(Vulkan_INCLUDE_DIR AND Vulkan_LIBRARY)
    set(Vulkan_TEST_FILE "${CMAKE_BINARY_DIR}/CMakeFiles/vulkan_test.c")
    
    file(WRITE ${Vulkan_TEST_FILE}
        "#include <vulkan/vulkan.h>\n"
        "int main() {\n"
        "    VkApplicationInfo appInfo = {};\n"
        "    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;\n"
        "    appInfo.apiVersion = VK_API_VERSION_1_0;\n"
        "    VkInstanceCreateInfo createInfo = {};\n"
        "    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;\n"
        "    createInfo.pApplicationInfo = &appInfo;\n"
        "    return 0;\n"
        "}\n"
    )
    
    try_compile(Vulkan_COMPILE_TEST_RESULT
        ${CMAKE_BINARY_DIR}
        ${Vulkan_TEST_FILE}
        CMAKE_FLAGS 
            "-DINCLUDE_DIRECTORIES:STRING=${Vulkan_INCLUDE_DIR}"
            "-DLINK_LIBRARIES:STRING=${Vulkan_LIBRARY}"
    )
    
    file(REMOVE ${Vulkan_TEST_FILE})
else()
    set(Vulkan_COMPILE_TEST_RESULT FALSE)
endif()

# Set up variables
if(Vulkan_INCLUDE_DIR AND Vulkan_LIBRARY)
    set(Vulkan_INCLUDE_DIRS ${Vulkan_INCLUDE_DIR})
    set(Vulkan_LIBRARIES ${Vulkan_LIBRARY})
endif()

# Handle standard CMake find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vulkan
    REQUIRED_VARS Vulkan_LIBRARY Vulkan_INCLUDE_DIR
    VERSION_VAR Vulkan_VERSION
)

# Create imported targets
if(Vulkan_FOUND AND NOT TARGET Vulkan::Vulkan)
    add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan::Vulkan PROPERTIES
        IMPORTED_LOCATION ${Vulkan_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${Vulkan_INCLUDE_DIRS}
    )
    
    # Platform-specific properties
    if(WIN32)
        set_target_properties(Vulkan::Vulkan PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "VK_USE_PLATFORM_WIN32_KHR"
        )
    elseif(APPLE)
        set_target_properties(Vulkan::Vulkan PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "VK_USE_PLATFORM_MACOS_MVK"
        )
    else()
        set_target_properties(Vulkan::Vulkan PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "VK_USE_PLATFORM_XLIB_KHR"
        )
    endif()
endif()

# Create GLSL compiler target
if(Vulkan_GLSLC_EXECUTABLE AND NOT TARGET Vulkan::glslc)
    add_executable(Vulkan::glslc IMPORTED)
    set_target_properties(Vulkan::glslc PROPERTIES
        IMPORTED_LOCATION ${Vulkan_GLSLC_EXECUTABLE}
    )
endif()

# Create GLSL validator target
if(Vulkan_GLSLANGVALIDATOR_EXECUTABLE AND NOT TARGET Vulkan::glslangValidator)
    add_executable(Vulkan::glslangValidator IMPORTED)
    set_target_properties(Vulkan::glslangValidator PROPERTIES
        IMPORTED_LOCATION ${Vulkan_GLSLANGVALIDATOR_EXECUTABLE}
    )
endif()

# Set Vulkan availability macros and compiler definitions
if(Vulkan_FOUND)
    add_definitions(-DHAVE_VULKAN=1)
    
    # Add version definitions
    if(Vulkan_VERSION_MAJOR AND Vulkan_VERSION_MINOR)
        add_definitions(-DVULKAN_VERSION_MAJOR=${Vulkan_VERSION_MAJOR})
        add_definitions(-DVULKAN_VERSION_MINOR=${Vulkan_VERSION_MINOR})
    endif()
    
    # Add API version definition
    if(Vulkan_API_VERSION)
        string(REPLACE "." "_" Vulkan_API_VERSION_DEFINE ${Vulkan_API_VERSION})
        add_definitions(-DVULKAN_API_VERSION=${Vulkan_API_VERSION_DEFINE})
    endif()
    
    # Add compute support definition
    if(Vulkan_COMPUTE_SUPPORT)
        add_definitions(-DVULKAN_COMPUTE_SUPPORT=1)
    endif()
    
    # Add SPIRV-Tools support definition
    if(Vulkan_SPIRV_TOOLS_FOUND)
        add_definitions(-DVULKAN_SPIRV_TOOLS_FOUND=1)
    endif()
    
    # Add compiler availability definitions
    if(Vulkan_GLSLC_EXECUTABLE)
        add_definitions(-DVULKAN_GLSLC_FOUND=1)
    endif()
    
    if(Vulkan_GLSLANGVALIDATOR_EXECUTABLE)
        add_definitions(-DVULKAN_GLSLANG_VALIDATOR_FOUND=1)
    endif()
endif()

# Function to compile GLSL shaders to SPIR-V
function(vulkan_compile_shaders target_name)
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "glslc not found, cannot compile shaders")
    endif()
    
    set(SHADER_SOURCE_FILES ${ARGN})
    set(SHADER_BINARY_FILES "")
    
    foreach(SHADER_SOURCE ${SHADER_SOURCE_FILES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        set(SHADER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv")
        
        add_custom_command(
            OUTPUT ${SHADER_BINARY}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER_SOURCE} -o ${SHADER_BINARY}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling shader ${SHADER_NAME}"
        )
        
        list(APPEND SHADER_BINARY_FILES ${SHADER_BINARY})
    endforeach()
    
    add_custom_target(${target_name} DEPENDS ${SHADER_BINARY_FILES})
endfunction()

# Mark variables as advanced
mark_as_advanced(
    Vulkan_SDK_ROOT
    Vulkan_INCLUDE_DIR
    Vulkan_LIBRARY
    Vulkan_GLSLC_EXECUTABLE
    Vulkan_GLSLANGVALIDATOR_EXECUTABLE
    Vulkan_SPIRV_AS
    Vulkan_SPIRV_DIS
    Vulkan_SPIRV_OPT
)

# Debug information
if(Vulkan_FOUND)
    message(STATUS "Found Vulkan: ${Vulkan_LIBRARY}")
    message(STATUS "Vulkan version: ${Vulkan_VERSION}")
    message(STATUS "Vulkan API version: ${Vulkan_API_VERSION}")
    message(STATUS "Vulkan include dir: ${Vulkan_INCLUDE_DIR}")
    message(STATUS "Vulkan SDK root: ${Vulkan_SDK_ROOT}")
    message(STATUS "Vulkan compute support: ${Vulkan_COMPUTE_SUPPORT}")
    message(STATUS "SPIRV-Tools found: ${Vulkan_SPIRV_TOOLS_FOUND}")
    if(Vulkan_GLSLC_EXECUTABLE)
        message(STATUS "glslc found: ${Vulkan_GLSLC_EXECUTABLE}")
    endif()
    if(Vulkan_GLSLANGVALIDATOR_EXECUTABLE)
        message(STATUS "glslangValidator found: ${Vulkan_GLSLANGVALIDATOR_EXECUTABLE}")
    endif()
else()
    message(STATUS "Vulkan not found")
endif()
