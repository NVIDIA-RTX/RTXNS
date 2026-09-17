#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.24)

project(
    RtxNeuralShading
    DESCRIPTION "RTX Neural Shading"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(ENABLE_DX12_COOP_VECTOR_PREVIEW "" OFF)
option(DONUT_WITH_DX11 "Not supported in this SDK" OFF)
option(DONUT_WITH_DX12 "DX12 is only supported with DX12_COOP_VECTOR_PREVIEW ON" OFF)

# Vulkan support requires the Vulkan SDK (see README): its DXC is the compiler for the SPIR-V shaders.
# Without the SDK, the Vulkan backend and the SPIR-V shaders are not built rather than compiling
# SPIR-V with whichever dxc happens to be found (the Windows SDK ships a dxc.exe without SPIR-V code
# generation). The SDK is detected from the VULKAN_SDK environment variable set by its installer, or
# from system-wide headers on Linux. Pass -DDONUT_WITH_VULKAN=OFF to build without Vulkan even when
# the SDK is installed; turning it ON without the SDK is a configuration error.
find_path(RTXNS_VULKAN_SDK_INCLUDE_DIR NAMES vulkan/vulkan.h HINTS ENV VULKAN_SDK PATH_SUFFIXES Include include
	DOC "Vulkan SDK include directory; used to detect whether the Vulkan SDK is installed")
mark_as_advanced(RTXNS_VULKAN_SDK_INCLUDE_DIR)
if(RTXNS_VULKAN_SDK_INCLUDE_DIR)
	set(_rtxns_vulkan_default ON)
else()
	set(_rtxns_vulkan_default OFF)
endif()
option(DONUT_WITH_VULKAN "Build the Vulkan version of the samples and their SPIR-V shaders (requires the Vulkan SDK)" ${_rtxns_vulkan_default})
if(DONUT_WITH_VULKAN AND RTXNS_VULKAN_SDK_INCLUDE_DIR)
	message(STATUS "RTXNS: Vulkan SDK found (${RTXNS_VULKAN_SDK_INCLUDE_DIR}); the Vulkan backend and SPIR-V shaders are enabled")
elseif(DONUT_WITH_VULKAN)
	message(FATAL_ERROR "RTXNS: DONUT_WITH_VULKAN is ON but the Vulkan SDK was not found (VULKAN_SDK is not set). "
		"Install the Vulkan SDK (see README) and re-run CMake, or configure with -DDONUT_WITH_VULKAN=OFF.")
else()
	message(STATUS "RTXNS: Vulkan SDK not found (VULKAN_SDK is not set); the Vulkan backend and SPIR-V shaders are disabled. Install the Vulkan SDK to enable them. After installing it, configure in a fresh build directory or remove RTXNS_VULKAN_SDK_INCLUDE_DIR and DONUT_WITH_VULKAN from the CMake cache so they are detected again.")
endif()
option(DONUT_WITH_STATIC_SHADERS "" ON)

set(DONUT_DIR "${CMAKE_SOURCE_DIR}/external/donut"
    CACHE PATH "Path to donut framework")
list(APPEND CMAKE_MODULE_PATH "${DONUT_DIR}")

# Register our path for CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/support/cmake")

# Detect the target and host CPU architectures (RTXNS_TARGET_ARCH / RTXNS_HOST_ARCH). These select
# the architecture-specific packages fetched below: NVAPI and Agility SDK binaries follow the target,
# the DXC and Slang compilers follow the host they run on.
include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/TargetArch.cmake")

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=1")
endif()

option(DONUT_WITH_ASSIMP "" OFF)

if(WIN32)
	set(RTXNS_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin/windows-${RTXNS_TARGET_ARCH}" CACHE PATH "Output directory for the RTXNS build")
else()
	set(RTXNS_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin/linux-${RTXNS_TARGET_ARCH}" CACHE PATH "Output directory for the RTXNS build")
endif()

# Executables are written to a per-configuration folder, bin/<platform>/<Config>/, together with the
# shaders and the D3D12 Agility runtime they load relative to the executable. Single-config
# generators (Ninja, Makefiles) need CMAKE_BUILD_TYPE for that folder name.
get_property(_rtxns_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (NOT _rtxns_multi_config AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()
set(RTXNS_OUTPUT_DIR "${RTXNS_BINARY_DIR}/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${RTXNS_OUTPUT_DIR}")

# ShaderMake appends the configuration itself, so this yields bin/<platform>/<Config>/ShaderMake.exe.
# FORCE: the layout is owned by the project, so build directories configured earlier do not keep a stale value.
set(SHADERMAKE_BIN_OUTPUT_PATH "${RTXNS_BINARY_DIR}" CACHE STRING "Output directory for the ShaderMake executable" FORCE)

set(D3D_AGILITY_SDK_PREVIEW_VERSION "1.721.3-preview" CACHE STRING "D3D12 Agility SDK preview version used for DX12 Cooperative Vector support" FORCE)
# Agility SDK 1.721.3 has no matching DirectX-Headers tag. The 1.721.2
# headers already expose the linear algebra APIs used by this project.
set(RTXNS_DIRECTX_HEADERS_VERSION "1.721.2-preview")
set(SHADERMAKE_DXC_VERSION "v1.10.2605.24" CACHE STRING "DXC version override" FORCE)
set(SHADERMAKE_DXC_DATE "preview_2026_05_22" CACHE STRING "DXC date override" FORCE)
set(SHADERMAKE_SLANG_VERSION "2026.10" CACHE STRING "Slang version override" FORCE)
set(RTXNS_DX12_SHADER_MODEL "6_10")

# Passed to Donut's Agility SDK fetch as the extraction directory; created/populated during configure.
set(RTXNS_D3D_AGILITY_SDK_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dx12-agility-sdk-${D3D_AGILITY_SDK_PREVIEW_VERSION}")

# Download the DXC and Slang compilers for the build host's architecture. This also turns off
# ShaderMake's own download, which does not recognise Windows on Arm hosts.
include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchShaderCompilers.cmake")

# When cross-compiling, build ShaderMake (which runs at build time) for the host instead of letting
# Donut build it with the target toolchain. No-op for native builds.
include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/HostShaderMake.cmake")

# Only used when DONUT_WITH_STATIC_SHADERS is OFF; the samples load the framework shaders from
# <executable dir>/shaders/framework/<api>, so this has to follow the per-configuration output folder.
set(DONUT_SHADERS_OUTPUT_DIR "${RTXNS_OUTPUT_DIR}/shaders/framework")

if (WIN32)
	if (ENABLE_DX12_COOP_VECTOR_PREVIEW)
		set(DONUT_WITH_DX12 ON)
		set(NVRHI_WITH_DX12 ON)
	else()
		# DX12 is only supported with DX12_COOP_VECTOR_PREVIEW
		set(DONUT_WITH_DX12 OFF)
		set(NVRHI_WITH_DX12 OFF)
	endif()
endif()

if(NOT DONUT_WITH_DX12 AND NOT DONUT_WITH_VULKAN)
	message(FATAL_ERROR "RTXNS: no graphics API is enabled. Vulkan is off (the Vulkan SDK was not found, or DONUT_WITH_VULKAN=OFF) "
		"and DX12 is off (Windows only; requires ENABLE_DX12_COOP_VECTOR_PREVIEW=ON).")
endif()

if (DONUT_WITH_DX12)
	# Get D3D Agility SDK Preview for Coop Vector support
	# The SDK is extracted into a versioned folder in the source tree that is shared by all build
	# directories. Donut's fetch deletes and re-extracts the folder on the first configure of every
	# new build directory, so once it is present, point Donut at the existing path instead of the URL.
	if (EXISTS "${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}/build/native/include/d3d12.h")
		set(DONUT_D3D_AGILITY_SDK_PATH "${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}")
		unset(DONUT_D3D_AGILITY_SDK_URL)
		message(STATUS "Agility SDK: using existing download in ${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}")
	else()
		set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${D3D_AGILITY_SDK_PREVIEW_VERSION}")
	endif()
	set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}" CACHE STRING "" FORCE)
	unset(_d3d_agility_include CACHE)
	unset(DONUT_D3D_AGILITY_SDK_CORE_DLL CACHE)
	unset(DONUT_D3D_AGILITY_SDK_LAYERS_DLL CACHE)
	include("${CMAKE_CURRENT_SOURCE_DIR}/external/donut/cmake/FetchAgilitySDK.cmake")
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/ConfigureAgilitySDK.cmake")

	# Get NVAPI
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchNVAPI.cmake")
endif()

# Provide Vulkan-Headers and DirectX-Headers to NVRHI from tag archives shared by all build directories
# instead of the git clones NVRHI would otherwise make in each one (see the module for details).
include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchGraphicsHeaders.cmake")

add_subdirectory(external/donut)

# DXC headers (dx/linalg.h) are at <DXC root>/inc/hlsl; must run after ShaderMake sets SHADERMAKE_DXC_PATH in Donut.
if(DONUT_WITH_DX12)
	cmake_path(GET SHADERMAKE_DXC_PATH PARENT_PATH _DXC_EXE_DIR)
	cmake_path(SET DXC_HLSL_INCLUDE_DIR NORMALIZE "${_DXC_EXE_DIR}/../../inc/hlsl")
endif()
add_subdirectory(src)
add_subdirectory(samples)
set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SimpleInferencing)

if (NOT EXISTS "${SHADERMAKE_SLANG_PATH}")
	message(FATAL_ERROR "SHADERMAKE_SLANG_PATH not set!")
endif()

# Copy the DXC DLLs next to Slang because the -dxc-path option doesn't seem to work.
if (WIN32)
	cmake_path(REMOVE_FILENAME SHADERMAKE_SLANG_PATH OUTPUT_VARIABLE SLANG_DIRECTORY)
	cmake_path(REPLACE_FILENAME SHADERMAKE_DXC_PATH "dxcompiler.dll" OUTPUT_VARIABLE DXCOMPILER_DLL)
	cmake_path(REPLACE_FILENAME SHADERMAKE_DXC_PATH "dxil.dll" OUTPUT_VARIABLE DXIL_DLL)
	file(COPY "${DXCOMPILER_DLL}" "${DXIL_DLL}" DESTINATION "${SLANG_DIRECTORY}/")
endif()

# Convenience launcher for the Slang compiler used by the build; the SlangPy training sample
# (samples/SlangpyTraining/Helpers.py) passes it to ShaderMake. A batch file on Windows, a shell
# script on other hosts. It points at the shared download in external/, so it is the same for all
# build directories of a given toolchain.
if (WIN32)
	file(WRITE "${CMAKE_SOURCE_DIR}/bin/slangc.bat" "${SHADERMAKE_SLANG_PATH} %*")
else()
	file(WRITE "${CMAKE_SOURCE_DIR}/bin/slangc.sh" "#!/bin/sh\nexec \"${SHADERMAKE_SLANG_PATH}\" \"$@\"\n")
	file(CHMOD "${CMAKE_SOURCE_DIR}/bin/slangc.sh" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

# Get implot. 
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchImplot.cmake")
    message(FATAL_ERROR "Missing FetchImplot.cmake file")
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchImplot.cmake")
set_target_properties(implot PROPERTIES FOLDER "Third-Party Libraries")

    
