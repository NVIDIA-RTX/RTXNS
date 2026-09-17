#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Toolchain file: Windows x64 target with MSVC.
#
# Used by the CMakePresets.json presets (Ninja generator; Visual Studio and VS Code set up the developer
# environment for the target from the preset's 'architecture' field). With single-config generators the
# compiler comes from that environment: from a plain shell use the Developer Command Prompt for this
# target (vcvarsall x64 on an x64 host, or vcvarsall arm64_x64 on an ARM64 host).
#
# It also works with a Visual Studio generator chosen on the command line (-G): CMAKE_GENERATOR_PLATFORM
# may be initialised by a toolchain file, so no -A is needed. A mismatch between this file and the
# compiler in use is rejected by support/cmake/TargetArch.cmake.

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # A platform given explicitly (-A) must agree with this file; refuse to override it silently.
    if(CMAKE_GENERATOR_PLATFORM)
        string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" _rtxns_platform)
        if(NOT _rtxns_platform STREQUAL "x64")
            message(FATAL_ERROR
                "This toolchain file targets x64, but the generator platform is '${CMAKE_GENERATOR_PLATFORM}'. "
                "Remove the -A option (the RTXNS presets do not need it) or use the toolchain file / preset that matches it.")
        endif()
    endif()
    set(CMAKE_GENERATOR_PLATFORM x64)
    # Also record it in the cache: 'cmake --build' (and therefore the IDEs) read the platform from there
    # to pass /p:Platform to MSBuild; a -A argument would have written this entry itself.
    set(CMAKE_GENERATOR_PLATFORM x64 CACHE INTERNAL "Generator platform (initialized by the toolchain file)")
elseif(NOT _rtxns_toolchain_env_checked)
    # vcvarsall.bat (and the IDEs, which run it) export the target architecture of the developer
    # environment. Catch a mismatching or missing environment here, before compiler detection fails
    # with a less specific message. The toolchain file is read several times per configure; check once.
    set(_rtxns_toolchain_env_checked TRUE)
    if(DEFINED ENV{VSCMD_ARG_TGT_ARCH})
        string(TOLOWER "$ENV{VSCMD_ARG_TGT_ARCH}" _rtxns_env_target)
        if(NOT _rtxns_env_target STREQUAL "x64")
            message(FATAL_ERROR
                "This toolchain file targets x64, but the Visual Studio developer environment targets "
                "'$ENV{VSCMD_ARG_TGT_ARCH}'. Configure from the Developer Command Prompt for x64 "
                "(vcvarsall x64 on an x64 host, or vcvarsall arm64_x64 on an ARM64 host), or pick the preset that matches this environment.")
        endif()
    elseif(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
        message(WARNING
            "No Visual Studio developer environment detected (VSCMD_ARG_TGT_ARCH is not set). With the "
            "${CMAKE_GENERATOR} generator the compiler comes from the environment: configure from the "
            "Developer Command Prompt for x64 (vcvarsall x64 on an x64 host, or vcvarsall arm64_x64 on an ARM64 host), or open the folder in Visual Studio or "
            "VS Code, which set it up from the preset.")
    endif()
endif()

set(RTXNS_TARGET_ARCH "x64" CACHE STRING "Target CPU architecture (set by the toolchain file)" FORCE)
