#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Builds ShaderMake for the build *host* when cross-compiling.
#
# ShaderMake runs during the build to compile shaders, so it must be a host executable. Donut
# normally builds it as part of the main project, i.e. with the target toolchain, which produces an
# unusable binary when the target architecture differs from the host. Donut only adds its own
# ShaderMake subdirectory when no `ShaderMake` target exists yet (external/donut/CMakeLists.txt),
# so this module builds ShaderMake with the host toolchain at configure time, defines that target as
# a placeholder, and provides everything Donut's compileshaders.cmake expects from the ShaderMake
# subdirectory: SHADERMAKE_PATH, SHADERMAKE_DXC_VK_PATH and the ShaderMakeBlob library.
#
# Why configure time and not an ExternalProject: with Visual Studio generators a single-target build
# ('cmake --build --target X', which is also what IDEs run) invokes MSBuild on X.vcxproj directly. The
# dependency of a shader target on a *utility* target such as an ExternalProject only exists at the
# solution level, so it would not be built before the shader PRE_BUILD steps that need ShaderMake.exe.
# Building here guarantees the executable exists before any build step; the build is incremental and
# takes a few seconds once the host build directory exists.
#
# For native builds (host == target) nothing is done and Donut builds ShaderMake as usual.
# RTXNS_HOST_SHADERMAKE can point at a prebuilt host ShaderMake to skip the host build entirely.
#
# Expects: RTXNS_HOST_ARCH / RTXNS_TARGET_ARCH (TargetArch.cmake), DONUT_DIR,
#          SHADERMAKE_DXC_PATH (FetchShaderCompilers.cmake). Include before add_subdirectory(donut).

if(NOT RTXNS_HOST_ARCH OR NOT RTXNS_TARGET_ARCH)
    message(FATAL_ERROR "HostShaderMake.cmake requires RTXNS_HOST_ARCH and RTXNS_TARGET_ARCH (include support/cmake/TargetArch.cmake first)")
endif()

if(RTXNS_HOST_ARCH STREQUAL RTXNS_TARGET_ARCH)
    return()
endif()

set(RTXNS_HOST_SHADERMAKE "" CACHE FILEPATH "Optional: prebuilt ShaderMake executable for the build host. When set, the configure-time host build of ShaderMake is skipped.")

set(_shadermake_src "${DONUT_DIR}/ShaderMake/ShaderMake")
set(_shadermake_bin "${CMAKE_BINARY_DIR}/host-tools/ShaderMake")

message(STATUS "RTXNS: cross-compiling (${RTXNS_HOST_ARCH} host -> ${RTXNS_TARGET_ARCH} target); ShaderMake must be a host executable")

if(RTXNS_HOST_SHADERMAKE)
    if(NOT EXISTS "${RTXNS_HOST_SHADERMAKE}")
        message(FATAL_ERROR "RTXNS_HOST_SHADERMAKE='${RTXNS_HOST_SHADERMAKE}' does not exist")
    endif()
    set(_shadermake_exe "${RTXNS_HOST_SHADERMAKE}")
    message(STATUS "RTXNS: using prebuilt host ShaderMake ${_shadermake_exe}")
else()
    # Command prefix used to run CMake for the host build: plain cmake, a batch wrapper that first
    # switches to the host toolchain environment, or cmake through '-E env' with CC/CXX cleared.
    set(_shadermake_cmake "${CMAKE_COMMAND}")
    set(_host_platform "")

    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        # Visual Studio sets up the toolchain environment per platform itself: build for the host platform.
        if(RTXNS_HOST_ARCH STREQUAL "arm64")
            set(_host_platform "ARM64")
        else()
            set(_host_platform "x64")
        endif()
    elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Single-config generator (e.g. Ninja) running inside a cross-compiling vcvars environment. Wrap
        # cmake in a batch file that switches to the host toolchain environment (vcvarsall <host>) before
        # configuring and building ShaderMake. vcvarsall.bat is located from the compiler path:
        #   <VC>/Tools/MSVC/<version>/bin/Host<host>/<target>/cl.exe
        # The wrapper also clears CC/CXX: CMake exports the compilers it detected into its own environment,
        # which child processes inherit, and that would make the host build pick the target C++ compiler.
        set(_vc_dir "${CMAKE_CXX_COMPILER}")
        foreach(_i RANGE 1 7)
            cmake_path(GET _vc_dir PARENT_PATH _vc_dir)
        endforeach()
        set(_vcvarsall "${_vc_dir}/Auxiliary/Build/vcvarsall.bat")
        string(REGEX MATCH "/MSVC/([0-9.]+)/" _unused "${CMAKE_CXX_COMPILER}")
        set(_vc_toolset_version "${CMAKE_MATCH_1}")
        if(NOT EXISTS "${_vcvarsall}" OR NOT _vc_toolset_version)
            message(FATAL_ERROR
                "Cross-compiling for ${RTXNS_TARGET_ARCH} on a ${RTXNS_HOST_ARCH} host requires a host build of ShaderMake, "
                "but vcvarsall.bat could not be located from CMAKE_CXX_COMPILER='${CMAKE_CXX_COMPILER}'. "
                "Use a Visual Studio generator, or set RTXNS_HOST_SHADERMAKE to a host ShaderMake executable.")
        endif()

        file(TO_NATIVE_PATH "${_vcvarsall}" _vcvarsall_native)
        file(TO_NATIVE_PATH "${CMAKE_COMMAND}" _cmake_native)
        set(_shadermake_wrapper "${CMAKE_BINARY_DIR}/host-tools/cmake-host-env.bat")
        set(_shadermake_cmake "${_shadermake_wrapper}")
        file(WRITE "${_shadermake_wrapper}"
"@echo off
set \"INCLUDE=\"
set \"LIB=\"
set \"LIBPATH=\"
set \"VSCMD_VER=\"
set \"CC=\"
set \"CXX=\"
call \"${_vcvarsall_native}\" ${RTXNS_HOST_ARCH} -vcvars_ver=${_vc_toolset_version} >nul 2>&1
\"${_cmake_native}\" %*
exit /b %ERRORLEVEL%
")
    else()
        # Non-MSVC cross toolchain (e.g. Linux). CMake exports the compilers it detected as CC/CXX to
        # child processes, which would make the host build use the target cross-compiler; clear them so
        # the host build picks up the default host compilers from PATH.
        set(_shadermake_cmake "${CMAKE_COMMAND}" -E env --unset=CC --unset=CXX "${CMAKE_COMMAND}")
        message(STATUS
            "RTXNS: non-MSVC toolchain; ShaderMake is built with the default host compilers from PATH. "
            "Set RTXNS_HOST_SHADERMAKE to a prebuilt host ShaderMake if that fails.")
    endif()

    # Multi-config generators place the executable in a per-configuration subdirectory.
    if(CMAKE_GENERATOR MATCHES "Visual Studio|Xcode|Multi-Config")
        set(_shadermake_exe "${_shadermake_bin}/Release/ShaderMake${CMAKE_EXECUTABLE_SUFFIX}")
    else()
        set(_shadermake_exe "${_shadermake_bin}/ShaderMake${CMAKE_EXECUTABLE_SUFFIX}")
    endif()

    message(STATUS "RTXNS: building ShaderMake for the ${RTXNS_HOST_ARCH} host in ${_shadermake_bin}")

    # Configure the host build once (same generator as the main build; Visual Studio additionally gets the
    # host platform), then build it. The build is incremental, so re-running it on every configure is cheap.
    if(NOT EXISTS "${_shadermake_bin}/CMakeCache.txt")
        set(_shadermake_cfg_args -S "${_shadermake_src}" -B "${_shadermake_bin}" -G "${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=Release)
        if(_host_platform)
            list(APPEND _shadermake_cfg_args -A "${_host_platform}")
        endif()
        if(CMAKE_GENERATOR_INSTANCE)
            list(APPEND _shadermake_cfg_args "-DCMAKE_GENERATOR_INSTANCE=${CMAKE_GENERATOR_INSTANCE}")
        endif()
        execute_process(
            COMMAND ${_shadermake_cmake} ${_shadermake_cfg_args}
            RESULT_VARIABLE _shadermake_result
            OUTPUT_VARIABLE _shadermake_output
            ERROR_VARIABLE _shadermake_output)
        if(NOT _shadermake_result EQUAL 0)
            file(REMOVE_RECURSE "${_shadermake_bin}")
            message(FATAL_ERROR "Configuring the host build of ShaderMake failed:\n${_shadermake_output}")
        endif()
    endif()

    execute_process(
        COMMAND ${_shadermake_cmake} --build "${_shadermake_bin}" --target ShaderMake --config Release
        RESULT_VARIABLE _shadermake_result
        OUTPUT_VARIABLE _shadermake_output
        ERROR_VARIABLE _shadermake_output)
    if(NOT _shadermake_result EQUAL 0)
        message(FATAL_ERROR "Building ShaderMake for the ${RTXNS_HOST_ARCH} host failed:\n${_shadermake_output}")
    elseif(NOT EXISTS "${_shadermake_exe}")
        message(FATAL_ERROR "ShaderMake was built for the ${RTXNS_HOST_ARCH} host but was not found at '${_shadermake_exe}'. "
                            "Delete '${_shadermake_bin}' and re-run CMake.\n${_shadermake_output}")
    endif()
    message(STATUS "RTXNS: host ShaderMake ready: ${_shadermake_exe}")
endif()

# Donut's compileshaders.cmake makes every shader target depend on a target named ShaderMake.
add_custom_target(ShaderMake)
set_target_properties(ShaderMake PROPERTIES FOLDER "ShaderMake")
set(SHADERMAKE_PATH "${_shadermake_exe}" CACHE INTERNAL "")

# Library that Donut's engine uses to read shader blobs; built for the target like the rest of Donut.
if(NOT TARGET ShaderMakeBlob)
    add_library(ShaderMakeBlob STATIC
        "${_shadermake_src}/ShaderBlob.h"
        "${_shadermake_src}/ShaderBlob.cpp"
    )
    target_include_directories(ShaderMakeBlob PUBLIC "${DONUT_DIR}/ShaderMake")
    set_target_properties(ShaderMakeBlob PROPERTIES FOLDER "ShaderMake" POSITION_INDEPENDENT_CODE ON)
endif()

# DXC for SPIR-V, normally located by the ShaderMake subdirectory: prefer the Vulkan SDK's dxc,
# fall back to the downloaded DXC (both host executables). On Windows only the Vulkan SDK is
# searched, never PATH: a developer command prompt puts the Windows SDK's bin folder on PATH, and
# the dxc.exe shipped there has no SPIR-V code generation.
if(NOT SHADERMAKE_DXC_VK_PATH)
    if(CMAKE_HOST_WIN32)
        if(DEFINED ENV{VULKAN_SDK})
            find_program(SHADERMAKE_DXC_VK_PATH NAMES dxc PATHS "$ENV{VULKAN_SDK}/Bin" NO_DEFAULT_PATH)
        endif()
    else()
        find_program(SHADERMAKE_DXC_VK_PATH NAMES dxc)
    endif()
    if(NOT SHADERMAKE_DXC_VK_PATH AND SHADERMAKE_DXC_PATH)
        set(SHADERMAKE_DXC_VK_PATH "${SHADERMAKE_DXC_PATH}" CACHE INTERNAL "")
    endif()
endif()
