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

option(DONUT_WITH_DX11 "" OFF)
option(DONUT_WITH_DX12 "" OFF)
option(DONUT_WITH_VULKAN "" ON)
option(DONUT_WITH_STATIC_SHADERS "" ON)

# Get Slang
set(SLANG_VERSION "2025.3.3")
set(SLANG_URL_BASE "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}")
if(WIN32)
    set(SLANG_URL "${SLANG_URL_BASE}/slang-${SLANG_VERSION}-windows-x86_64.zip")
else()
    set(SLANG_URL "${SLANG_URL_BASE}/slang-${SLANG_VERSION}-linux-x86_64-glibc-2.17.tar.gz")
endif()

include(FetchContent)
# This is used for downloading prebuilt external binaries.
macro(download_package name url)
    FetchContent_Declare(
        ${name}
        URL ${url}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(${name})
    if(NOT ${name}_POPULATED)
        message(STATUS "Populating ${name} ...")
        FetchContent_Populate(${name})
    endif()
endmacro()

download_package(slang ${SLANG_URL})

if (WIN32)
    set(SLANGC_PATH "${slang_SOURCE_DIR}/bin/slangc.exe")
else()
    set(SLANGC_PATH "${slang_SOURCE_DIR}/bin/slangc")
endif()

if (NOT SLANGC_PATH)
    message(FATAL_ERROR "Slang compiler not found - this is required for CoopVec support.")
else()
    message("Slang compiler found: ${SLANGC_PATH}")
    set(RTXNS_COOPVEC ON)
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=1")
endif()

option(DONUT_WITH_ASSIMP "" OFF)

set(RTXNS_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin/$<CONFIG>" CACHE PATH "Output directory for the RTXNS build")
set(SHADERMAKE_BIN_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin" CACHE STRING "Output directory for the ShaderMake executable")
set(DONUT_SHADERS_OUTPUT_DIR "${RTXNS_BINARY_DIR}/shaders/framework")

add_subdirectory(external/donut)
add_subdirectory(src)
add_subdirectory(samples)
set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SimpleInferencing)

file(WRITE "${CMAKE_SOURCE_DIR}/bin/slangc.bat" "${SLANGC_PATH} %*")
    
