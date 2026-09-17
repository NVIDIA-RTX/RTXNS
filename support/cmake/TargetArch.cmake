#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Determines the CPU architecture of the build target and of the build host so that
# architecture-specific packages (NVAPI, Agility SDK, DXC, Slang) are selected consistently.
#
#   RTXNS_TARGET_ARCH  Architecture the samples are compiled for: "x64", "arm64" or "arm64ec".
#                      Auto-detected by default; override with -DRTXNS_TARGET_ARCH=<arch>.
#   RTXNS_HOST_ARCH    Architecture of the machine running the build: "x64" or "arm64". Used for
#                      tools that execute at build time (DXC, Slang, ShaderMake).
#
# The target is detected from the first of these that is set: the compiler architecture id
# (MSVC / clang-cl; covers both "-A ARM64" and Ninja inside an ARM64 vcvars environment), the
# Visual Studio generator platform, then CMAKE_SYSTEM_PROCESSOR. All comparisons are
# case-insensitive because Windows reports "ARM64" while Linux reports "aarch64".

function(rtxns_normalize_arch input output_var)
    string(TOLOWER "${input}" _arch)
    if(_arch MATCHES "^(x64|amd64|x86_64)$")
        set(${output_var} "x64" PARENT_SCOPE)
    elseif(_arch STREQUAL "arm64ec")
        set(${output_var} "arm64ec" PARENT_SCOPE)
    elseif(_arch MATCHES "^(arm64|aarch64)$")
        set(${output_var} "arm64" PARENT_SCOPE)
    else()
        set(${output_var} "" PARENT_SCOPE)
    endif()
endfunction()

set(RTXNS_TARGET_ARCH "" CACHE STRING "Target CPU architecture: x64, arm64 or arm64ec. Leave empty to auto-detect from the compiler.")
if(NOT CMAKE_SCRIPT_MODE_FILE)
    set_property(CACHE RTXNS_TARGET_ARCH PROPERTY STRINGS "" "x64" "arm64" "arm64ec")
endif()

if(RTXNS_TARGET_ARCH)
    set(_rtxns_arch_source "RTXNS_TARGET_ARCH")
    set(_rtxns_arch_value "${RTXNS_TARGET_ARCH}")
elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
    set(_rtxns_arch_source "CMAKE_CXX_COMPILER_ARCHITECTURE_ID")
    set(_rtxns_arch_value "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
elseif(CMAKE_GENERATOR_PLATFORM)
    set(_rtxns_arch_source "CMAKE_GENERATOR_PLATFORM")
    set(_rtxns_arch_value "${CMAKE_GENERATOR_PLATFORM}")
elseif(CMAKE_VS_PLATFORM_NAME)
    set(_rtxns_arch_source "CMAKE_VS_PLATFORM_NAME")
    set(_rtxns_arch_value "${CMAKE_VS_PLATFORM_NAME}")
else()
    set(_rtxns_arch_source "CMAKE_SYSTEM_PROCESSOR")
    set(_rtxns_arch_value "${CMAKE_SYSTEM_PROCESSOR}")
endif()

rtxns_normalize_arch("${_rtxns_arch_value}" _rtxns_target_arch)
if(NOT _rtxns_target_arch)
    message(FATAL_ERROR
        "Unsupported target architecture '${_rtxns_arch_value}' (from ${_rtxns_arch_source}). "
        "RTXNS supports x64 and arm64 targets; pass -DRTXNS_TARGET_ARCH=<x64|arm64|arm64ec> to override detection.")
endif()

# An explicit RTXNS_TARGET_ARCH (e.g. from a CMakePresets.json preset) must agree with the compiler
# that is actually in use; otherwise architecture-specific packages would not match the binaries.
# The usual cause is running a preset from a developer prompt for the wrong architecture.
if(RTXNS_TARGET_ARCH AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
    rtxns_normalize_arch("${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" _rtxns_compiler_arch)
    if(_rtxns_compiler_arch AND NOT _rtxns_compiler_arch STREQUAL _rtxns_target_arch)
        message(FATAL_ERROR
            "RTXNS_TARGET_ARCH is '${_rtxns_target_arch}' but the C++ compiler targets '${_rtxns_compiler_arch}' "
            "(${CMAKE_CXX_COMPILER}). Configure from the Developer Command Prompt for the target architecture "
            "(e.g. 'vcvarsall x64_arm64' for ARM64 on an x64 host), or pick the preset that matches your environment.")
    endif()
endif()

# Expose the resolved value under the same name; this shadows the (possibly empty) cache entry
# for the rest of the configure run.
set(RTXNS_TARGET_ARCH "${_rtxns_target_arch}")

rtxns_normalize_arch("${CMAKE_HOST_SYSTEM_PROCESSOR}" RTXNS_HOST_ARCH)
if(NOT RTXNS_HOST_ARCH OR RTXNS_HOST_ARCH STREQUAL "arm64ec")
    message(WARNING "Unrecognised host processor '${CMAKE_HOST_SYSTEM_PROCESSOR}'; assuming an x64 build host for downloaded tools.")
    set(RTXNS_HOST_ARCH "x64")
endif()

message(STATUS "RTXNS: target architecture '${RTXNS_TARGET_ARCH}' (from ${_rtxns_arch_source}='${_rtxns_arch_value}'), host architecture '${RTXNS_HOST_ARCH}'")
