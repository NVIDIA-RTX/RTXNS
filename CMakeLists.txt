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
set(DONUT_SHADERS_OUTPUT_DIR "${RTXNS_BINARY_DIR}/bin/shaders/framework")

# Get Slang
set(SLANG_VERSION "2025.10")
set(SLANG_URL_BASE "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}")
if(WIN32)
    set(SLANG_URL "${SLANG_URL_BASE}/slang-${SLANG_VERSION}-windows-x86_64.zip")
else()
    set(SLANG_URL "${SLANG_URL_BASE}/slang-${SLANG_VERSION}-linux-x86_64-glibc-2.17.tar.gz")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchPrebuildBinary.cmake")
download_package(slang ${SLANG_URL})

if (WIN32)
	set(SLANGC_PATH "${slang_SOURCE_DIR}/bin/slangc.exe")
	if (ENABLE_DX12_COOP_VECTOR_PREVIEW)
		set(DONUT_WITH_DX12 ON)
		set(NVRHI WITH_DX12 ON)
	else()
		# DX12 is only supported with DX12_COOP_VECTOR_PREVIEW
		set(DONUT_WITH_DX12 OFF)
		set(NVRHI WITH_DX12 OFF)
	endif()
else()
	set(SLANGC_PATH "${slang_SOURCE_DIR}/bin/slangc")
endif()

if (NOT SLANGC_PATH)
    message(FATAL_ERROR "Slang compiler not found - this is required for CoopVec support.")
else()
    message("Slang compiler found: ${SLANGC_PATH}")
endif()

if (DONUT_WITH_DX12)
	# Get D3D Agility SDK Preview for Coop Vector support
	set(D3D_AGILITY_SDK_PREVIEW_VERSION "1.717.0-preview")
	set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${D3D_AGILITY_SDK_PREVIEW_VERSION}")
	set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dx12-agility-sdk" CACHE STRING "" FORCE)
	include("${CMAKE_CURRENT_SOURCE_DIR}/external/donut/cmake/FetchAgilitySDK.cmake")
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/ConfigureAgilitySDK.cmake")
	
	# Get DXC preview for SM6.9 support
	set(DXC_PREVIEW_VERSION "1.8.2505.28")
	set(DXC_PREVIEW_PATH "" CACHE STRING "Directory to fetch the DXC to, empty uses build directory default")
	include("${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/FetchDXCPreview.cmake")
	
	set(DXC_PATH "${DXC_PREVIEW_PATH}")
	
	# copy dxc to Slang
	foreach(file_name IN ITEMS dxc.exe dxcompiler.dll dxil.dll)
		set(src "${DXC_PREVIEW_BIN_PATH}/${file_name}")
		set(dst "${slang_SOURCE_DIR}/bin/")
		if(EXISTS "${src}")
			configure_file("${src}" "${dst}" COPYONLY)
		else()
			message(WARNING "DXC binary not found: ${src}")
		endif()
	endforeach()
endif()

add_subdirectory(external/donut)
add_subdirectory(src)
add_subdirectory(samples)
set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SimpleInferencing)

file(WRITE "${CMAKE_SOURCE_DIR}/bin/slangc.bat" "${SLANGC_PATH} %*")
    
