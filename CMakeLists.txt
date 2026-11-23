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

# Whether this SDK is the top-level CMake project. When it is instead consumed
# via add_subdirectory() -- e.g. by the Neural Material SDK or the Neural Texture
# Compression SDK, which vendor this repo under external/NeuralShadingSDK -- the
# parent project owns the samples, the IDE startup project, the compile_commands.json
# mirror, the git hooks, the build type and output directories, and the shader
# toolchain (donut/NVRHI/ShaderMake and the DXC/Slang downloads), so we build only
# the libraries against the targets the parent already created.
if (PROJECT_IS_TOP_LEVEL)
    set(RTXNS_IS_TOP_LEVEL ON)
else()
    set(RTXNS_IS_TOP_LEVEL OFF)
endif()

option(RTXNS_BUILD_SAMPLES "Build the RTXNS sample applications" ${RTXNS_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(DONUT_WITH_DX11 "Not supported in this SDK" OFF)
option(DONUT_WITH_DX12 "Build the DX12 backend (Windows only)" ${WIN32})
if(DONUT_WITH_DX12 AND NOT WIN32)
	message(FATAL_ERROR "RTXNS: DONUT_WITH_DX12 is only supported on Windows.")
endif()

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
elseif(DONUT_WITH_VULKAN AND NOT RTXNS_IS_TOP_LEVEL)
	message(STATUS "RTXNS: Vulkan SDK not found (VULKAN_SDK is not set); the embedding project provides the Vulkan toolchain")
elseif(DONUT_WITH_VULKAN)
	message(FATAL_ERROR "RTXNS: DONUT_WITH_VULKAN is ON but the Vulkan SDK was not found (VULKAN_SDK is not set). "
		"Install the Vulkan SDK (see README) and re-run CMake, or configure with -DDONUT_WITH_VULKAN=OFF.")
else()
	message(STATUS "RTXNS: Vulkan SDK not found (VULKAN_SDK is not set); the Vulkan backend and SPIR-V shaders are disabled. Install the Vulkan SDK to enable them. After installing it, configure in a fresh build directory or remove RTXNS_VULKAN_SDK_INCLUDE_DIR and DONUT_WITH_VULKAN from the CMake cache so they are detected again.")
endif()
option(DONUT_WITH_STATIC_SHADERS "" ON)

# CMAKE_CURRENT_SOURCE_DIR, not CMAKE_SOURCE_DIR: donut is a submodule of *this*
# repo, so when this SDK is embedded in another project the path must still
# resolve inside our own tree.
set(DONUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/donut"
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
set(RTXNS_OUTPUT_DIR "${RTXNS_BINARY_DIR}/$<CONFIG>")
if (RTXNS_IS_TOP_LEVEL)
	get_property(_rtxns_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
	if (NOT _rtxns_multi_config AND NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
		set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
	endif()
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${RTXNS_OUTPUT_DIR}")

	# ShaderMake appends the configuration itself, so this yields bin/<platform>/<Config>/ShaderMake.exe.
	# FORCE: the layout is owned by the project, so build directories configured earlier do not keep a stale value.
	set(SHADERMAKE_BIN_OUTPUT_PATH "${RTXNS_BINARY_DIR}" CACHE STRING "Output directory for the ShaderMake executable" FORCE)
endif()

# Shader toolchain versions. A top-level build owns them (FORCE, so build directories configured
# before a version bump follow it); an embedded build takes the parent project's values and only
# provides defaults.
if (RTXNS_IS_TOP_LEVEL)
	set(_rtxns_pin_force FORCE)
else()
	set(_rtxns_pin_force "")
endif()
set(D3D_AGILITY_SDK_PREVIEW_VERSION "1.721.3-preview" CACHE STRING "D3D12 Agility SDK preview version used for DX12 Cooperative Vector support" ${_rtxns_pin_force})
# Agility SDK 1.721.3 has no matching DirectX-Headers tag. The 1.721.2
# headers already expose the linear algebra APIs used by this project.
set(RTXNS_DIRECTX_HEADERS_VERSION "1.721.2-preview")
set(SHADERMAKE_DXC_VERSION "v1.10.2605.24" CACHE STRING "DXC version override" ${_rtxns_pin_force})
set(SHADERMAKE_DXC_DATE "preview_2026_05_22" CACHE STRING "DXC date override" ${_rtxns_pin_force})
set(SHADERMAKE_SLANG_VERSION "2026.20" CACHE STRING "Slang version override" ${_rtxns_pin_force})
if (SHADERMAKE_SLANG_VERSION VERSION_LESS "2026.20")
	message(WARNING "RTXNS: the shader library requires Slang 2026.20 or newer (native CoopVec autodiff, derivatives of the CoopVec operators on HLSL), "
		"but SHADERMAKE_SLANG_VERSION is ${SHADERMAKE_SLANG_VERSION}")
endif()
set(RTXNS_DX12_SHADER_MODEL "6_10")

# Passed to Donut's Agility SDK fetch as the extraction directory; created/populated during configure.
set(RTXNS_D3D_AGILITY_SDK_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dx12-agility-sdk-${D3D_AGILITY_SDK_PREVIEW_VERSION}")

# Download the DXC and Slang compilers for the build host's architecture. This also turns off
# ShaderMake's own download, which does not recognise Windows on Arm hosts.
# When embedded in a project that has already configured ShaderMake, its compilers
# (SHADERMAKE_DXC_PATH / SHADERMAKE_SLANG_PATH) are used instead.
if (NOT TARGET ShaderMake)
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchShaderCompilers.cmake")

	# When cross-compiling, build ShaderMake (which runs at build time) for the host instead of letting
	# Donut build it with the target toolchain. No-op for native builds.
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/HostShaderMake.cmake")
endif()

# Only used when DONUT_WITH_STATIC_SHADERS is OFF; the samples load the framework shaders from
# <executable dir>/shaders/framework/<api>, so this has to follow the per-configuration output folder.
set(DONUT_SHADERS_OUTPUT_DIR "${RTXNS_OUTPUT_DIR}/shaders/framework")

set(NVRHI_WITH_DX12 ${DONUT_WITH_DX12})

if(NOT DONUT_WITH_DX12 AND NOT DONUT_WITH_VULKAN)
	message(FATAL_ERROR "RTXNS: no graphics API is enabled. Vulkan is off (the Vulkan SDK was not found, or DONUT_WITH_VULKAN=OFF) "
		"and DX12 is off (Windows only; enable it with DONUT_WITH_DX12=ON).")
endif()

if (DONUT_WITH_DX12)
	if (TARGET dx12-agility-sdk)
		# The parent owns deployment and must expose the version exported by DeviceUtils.cpp.
		if (NOT DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION)
			message(FATAL_ERROR "RTXNS: the existing dx12-agility-sdk target requires the parent to provide "
				"DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION.")
		endif()
		message(STATUS "RTXNS: using the existing dx12-agility-sdk target")
	else()
		# An embedded build may supply the SDK location without a deployment target.
		if (RTXNS_IS_TOP_LEVEL OR (NOT DONUT_D3D_AGILITY_SDK_PATH AND NOT DONUT_D3D_AGILITY_SDK_URL))
			# Reuse the shared download instead of letting FetchContent extract it again.
			if (EXISTS "${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}/build/native/include/d3d12.h")
				set(DONUT_D3D_AGILITY_SDK_PATH "${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}")
				unset(DONUT_D3D_AGILITY_SDK_URL)
				message(STATUS "Agility SDK: using existing download in ${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}")
			else()
				set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${D3D_AGILITY_SDK_PREVIEW_VERSION}")
			endif()
			set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}" CACHE STRING "" FORCE)
		endif()
		# Refresh discovery for the selected SDK, including a changed parent-supplied path.
		unset(_d3d_agility_include CACHE)
		unset(DONUT_D3D_AGILITY_SDK_CORE_DLL CACHE)
		unset(DONUT_D3D_AGILITY_SDK_LAYERS_DLL CACHE)
		include("${CMAKE_CURRENT_SOURCE_DIR}/external/donut/cmake/FetchAgilitySDK.cmake")
		include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/ConfigureAgilitySDK.cmake")
	endif()

	# Resolve shader includes even when the parent already provides the NVAPI target.
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchNVAPI.cmake")
endif()

# Provide Vulkan-Headers and DirectX-Headers to NVRHI from tag archives shared by all build directories
# instead of the git clones NVRHI would otherwise make in each one (see the module for details).
# When embedded in a project that already provides donut (and therefore NVRHI and ShaderMake),
# reuse the parent's targets instead of adding a second copy.
if (NOT TARGET donut_core)
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchGraphicsHeaders.cmake")
	add_subdirectory(external/donut)
endif()

# DXC headers (dx/linalg.h) are at <DXC root>/inc/hlsl; must run after ShaderMake sets SHADERMAKE_DXC_PATH in Donut.
if(DONUT_WITH_DX12)
	cmake_path(GET SHADERMAKE_DXC_PATH PARENT_PATH _DXC_EXE_DIR)
	cmake_path(SET DXC_HLSL_INCLUDE_DIR NORMALIZE "${_DXC_EXE_DIR}/../../inc/hlsl")
endif()

add_subdirectory(src)

if (RTXNS_BUILD_SAMPLES)
	add_subdirectory(samples)
	# VS_STARTUP_PROJECT is only honoured on the top-level directory; when this
	# SDK is embedded the parent project picks its own startup target.
	set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SimpleInferencing)
endif()

if (NOT EXISTS "${SHADERMAKE_SLANG_PATH}")
	message(FATAL_ERROR "SHADERMAKE_SLANG_PATH not set!")
endif()

# Only standalone builds create the convenience launcher in bin/.
if (RTXNS_IS_TOP_LEVEL)
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
endif()

# Get implot. 
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchImplot.cmake")
    message(FATAL_ERROR "Missing FetchImplot.cmake file")
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchImplot.cmake")
set_target_properties(implot PROPERTIES FOLDER "Third-Party Libraries")

    
