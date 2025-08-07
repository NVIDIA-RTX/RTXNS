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

set(SHADERMAKE_DXC_VERSION "v1.8.2505" CACHE STRING "")
set(SHADERMAKE_DXC_DATE "2025_05_24" CACHE STRING "")

option(SHADERMAKE_FIND_SLANG "" ON)
set(SHADERMAKE_SLANG_VERSION "2025.10" CACHE STRING "")

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
	set(D3D_AGILITY_SDK_PREVIEW_VERSION "1.717.0-preview")
	set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${D3D_AGILITY_SDK_PREVIEW_VERSION}")
	set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dx12-agility-sdk" CACHE STRING "" FORCE)
	include("${CMAKE_CURRENT_SOURCE_DIR}/external/donut/cmake/FetchAgilitySDK.cmake")
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/ConfigureAgilitySDK.cmake")
endif()

add_subdirectory(external/donut)
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
    
