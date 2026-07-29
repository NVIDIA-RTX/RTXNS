#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.10)

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
option(DONUT_WITH_VULKAN "" ON)
option(DONUT_WITH_STATIC_SHADERS "" ON)

set(DONUT_DIR "${CMAKE_SOURCE_DIR}/external/donut"
    CACHE PATH "Path to donut framework")
list(APPEND CMAKE_MODULE_PATH "${DONUT_DIR}")

# Register our path for CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/support/cmake")

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=1")
endif()

option(DONUT_WITH_ASSIMP "" OFF)

if(WIN32)
	set(RTXNS_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin/windows-x64" CACHE PATH "Output directory for the RTXNS build")
else()
	set(RTXNS_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin/linux-x64" CACHE PATH "Output directory for the RTXNS build")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${RTXNS_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${RTXNS_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${RTXNS_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${RTXNS_BINARY_DIR}")

set(SHADERMAKE_BIN_OUTPUT_PATH "${RTXNS_BINARY_DIR}/bin" CACHE STRING "Output directory for the ShaderMake executable")

# 717 is retained as a temporary compatibility path for the SM 6.9 preview driver/toolchain.
# Prefer 721 for new development; The 717 path will be removed once support for the legacy preview driver is dropped.
set(RTXNS_DX12_COOP_VECTOR_PREVIEW "721" CACHE STRING "DX12 Cooperative Vector preview toolchain to use (721 or 717). Requires a driver compatible with the selected preview; the same driver is not expected to support both.")
set_property(CACHE RTXNS_DX12_COOP_VECTOR_PREVIEW PROPERTY STRINGS "721" "717")

if("${RTXNS_DX12_COOP_VECTOR_PREVIEW}" STREQUAL "717")
	set(D3D_AGILITY_SDK_PREVIEW_VERSION "1.717.0-preview" CACHE STRING "D3D12 Agility SDK preview version used for DX12 Cooperative Vector support" FORCE)
	set(SHADERMAKE_DXC_VERSION "v1.8.2505" CACHE STRING "DXC version override" FORCE)
	set(SHADERMAKE_DXC_DATE "2025_05_24" CACHE STRING "DXC date override" FORCE)
	set(SHADERMAKE_SLANG_VERSION "2025.23.1" CACHE STRING "Slang version override" FORCE)
	set(RTXNS_DX12_SHADER_MODEL "6_9")
elseif("${RTXNS_DX12_COOP_VECTOR_PREVIEW}" STREQUAL "721")
	set(D3D_AGILITY_SDK_PREVIEW_VERSION "1.721.2-preview" CACHE STRING "D3D12 Agility SDK preview version used for DX12 Cooperative Vector support" FORCE)
	set(SHADERMAKE_DXC_VERSION "v1.10.2605.24" CACHE STRING "DXC version override" FORCE)
	set(SHADERMAKE_DXC_DATE "preview_2026_05_22" CACHE STRING "DXC date override" FORCE)
	set(SHADERMAKE_SLANG_VERSION "2026.10" CACHE STRING "Slang version override" FORCE)
	set(RTXNS_DX12_SHADER_MODEL "6_10")
else()
	message(FATAL_ERROR "Unsupported RTXNS_DX12_COOP_VECTOR_PREVIEW='${RTXNS_DX12_COOP_VECTOR_PREVIEW}'. Expected '721' or '717'.")
endif()

# These paths are passed to FetchContent as SOURCE_DIR and are created/populated during configure.
set(RTXNS_D3D_AGILITY_SDK_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dx12-agility-sdk-${D3D_AGILITY_SDK_PREVIEW_VERSION}")
set(RTXNS_DIRECTX_HEADERS_FETCH_DIR "${CMAKE_BINARY_DIR}/_deps/directx_headers-${D3D_AGILITY_SDK_PREVIEW_VERSION}")

option(SHADERMAKE_FIND_SLANG "" ON)

set(DONUT_SHADERS_OUTPUT_DIR "${RTXNS_BINARY_DIR}/bin/shaders/framework")

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

if (DONUT_WITH_DX12)
	# Get D3D Agility SDK Preview for Coop Vector support
	set(RTXNS_DIRECTX_HEADERS_GIT_TAG "v${D3D_AGILITY_SDK_PREVIEW_VERSION}" CACHE STRING "DirectX-Headers git tag or commit used by NVRHI" FORCE)
	set(NVRHI_DIRECTX_HEADERS_GIT_TAG "${RTXNS_DIRECTX_HEADERS_GIT_TAG}" CACHE STRING "DirectX-Headers git tag or commit used by NVRHI" FORCE)
	set(NVRHI_DIRECTX_HEADERS_FETCH_DIR "${RTXNS_DIRECTX_HEADERS_FETCH_DIR}" CACHE STRING "Source directory for fetched DirectX-Headers" FORCE)
	set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${D3D_AGILITY_SDK_PREVIEW_VERSION}")
	set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "${RTXNS_D3D_AGILITY_SDK_FETCH_DIR}" CACHE STRING "" FORCE)
	unset(_d3d_agility_include CACHE)
	unset(DONUT_D3D_AGILITY_SDK_CORE_DLL CACHE)
	unset(DONUT_D3D_AGILITY_SDK_LAYERS_DLL CACHE)
	include("${CMAKE_CURRENT_SOURCE_DIR}/external/donut/cmake/FetchAgilitySDK.cmake")
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/ConfigureAgilitySDK.cmake")

	# Get NVAPI
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchNVAPI.cmake")
endif()

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

file(WRITE "${CMAKE_SOURCE_DIR}/bin/slangc.bat" "${SHADERMAKE_SLANG_PATH} %*")

# Get implot. 
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchImplot.cmake")
    message(FATAL_ERROR "Missing FetchImplot.cmake file")
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchImplot.cmake")
set_target_properties(implot PROPERTIES FOLDER "Third-Party Libraries")

    
