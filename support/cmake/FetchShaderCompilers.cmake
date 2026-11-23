#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Downloads the DXC and Slang compilers that ShaderMake / Donut invoke at build time.
#
# ShaderMake can download these itself (SHADERMAKE_FIND_DXC / SHADERMAKE_FIND_SLANG), but it picks
# the package from CMAKE_SYSTEM_PROCESSOR with a case-sensitive match that does not recognise
# Windows on Arm ("ARM64"), and it uses the target rather than the host architecture. Both
# compilers execute on the build host, so they are selected from RTXNS_HOST_ARCH here and
# ShaderMake's own download is disabled.
#
# The packages are extracted into versioned folders under external/ that all build directories
# share (see FetchArchive.cmake): switching between x64/arm64 or Debug/Release build directories
# does not download them again, and bin/slangc.bat (used by the SlangPy sample) keeps pointing at
# the same location.
#
# Expects: RTXNS_HOST_ARCH (TargetArch.cmake), SHADERMAKE_DXC_VERSION, SHADERMAKE_DXC_DATE,
#          SHADERMAKE_SLANG_VERSION.

include("${CMAKE_CURRENT_LIST_DIR}/FetchArchive.cmake")

if(NOT RTXNS_HOST_ARCH)
    message(FATAL_ERROR "FetchShaderCompilers.cmake requires RTXNS_HOST_ARCH (include support/cmake/TargetArch.cmake first)")
endif()

set(SHADERMAKE_FIND_DXC OFF CACHE BOOL "Disabled: DXC is downloaded by support/cmake/FetchShaderCompilers.cmake" FORCE)
set(SHADERMAKE_FIND_SLANG OFF CACHE BOOL "Disabled: Slang is downloaded by support/cmake/FetchShaderCompilers.cmake" FORCE)

# This SDK's own external/ directory, also when the SDK is embedded in another project.
cmake_path(SET _rtxns_tools_dir NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../../external")

# Older builds cached SDK-selected paths as INTERNAL entries. Rediscover those
# defaults after a version change, while preserving explicit compiler overrides.
foreach(_compiler DXC SLANG)
    set(_path_var "SHADERMAKE_${_compiler}_PATH")
    get_property(_path_type CACHE "${_path_var}" PROPERTY TYPE)
    get_property(_cached_path CACHE "${_path_var}" PROPERTY VALUE)
    cmake_path(IS_PREFIX _rtxns_tools_dir "${_cached_path}" NORMALIZE _sdk_path)
    if(_path_type STREQUAL "INTERNAL" AND _sdk_path AND "${${_path_var}}" STREQUAL "${_cached_path}")
        unset(${_path_var} CACHE)
    endif()
    if(${_path_var})
        if(NOT EXISTS "${${_path_var}}" OR IS_DIRECTORY "${${_path_var}}")
            message(FATAL_ERROR "RTXNS: ${_path_var}='${${_path_var}}' is not a compiler executable.")
        endif()
        message(STATUS "RTXNS: using supplied ${_compiler} compiler ${${_path_var}}")
    endif()
endforeach()

# ---- DXC (github.com/microsoft/DirectXShaderCompiler) ----
# The Windows release archive contains x64, x86 and arm64 binaries; the Linux archive is x86_64 only.
set(_dxc_dir "")
if(CMAKE_HOST_WIN32)
    set(_dxc_dir "${_rtxns_tools_dir}/dxc-${SHADERMAKE_DXC_VERSION}-windows")
    set(_dxc_archive "dxc_${SHADERMAKE_DXC_DATE}.zip")
    set(_dxc_exe "bin/${RTXNS_HOST_ARCH}/dxc.exe")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND RTXNS_HOST_ARCH STREQUAL "x64")
    set(_dxc_dir "${_rtxns_tools_dir}/dxc-${SHADERMAKE_DXC_VERSION}-linux-x86_64")
    set(_dxc_archive "linux_dxc_${SHADERMAKE_DXC_DATE}.x86_64.tar.gz")
    set(_dxc_exe "bin/dxc")
endif()

if(NOT SHADERMAKE_DXC_PATH)
    if(_dxc_dir)
        rtxns_fetch_archive(DXC
            "https://github.com/microsoft/DirectXShaderCompiler/releases/download/${SHADERMAKE_DXC_VERSION}/${_dxc_archive}"
            "${_dxc_dir}" "${_dxc_exe}")
        set(SHADERMAKE_DXC_PATH "${_dxc_dir}/${_dxc_exe}")
    else()
        # No prebuilt DXC on GitHub for this host; ShaderMake falls back to the Vulkan SDK's dxc.
        message(STATUS "RTXNS: no prebuilt DXC release for ${CMAKE_HOST_SYSTEM_NAME}/${RTXNS_HOST_ARCH}; using the Vulkan SDK's dxc if available")
    endif()
endif()

# ---- Slang (github.com/shader-slang/slang) ----
if(CMAKE_HOST_WIN32)
    set(_slang_os "windows")
    set(_slang_exe "bin/slangc.exe")
elseif(CMAKE_HOST_APPLE)
    set(_slang_os "macos")
    set(_slang_exe "bin/slangc")
else()
    set(_slang_os "linux")
    set(_slang_exe "bin/slangc")
endif()
if(RTXNS_HOST_ARCH STREQUAL "arm64")
    set(_slang_cpu "aarch64")
else()
    set(_slang_cpu "x86_64")
endif()
set(_slang_dir "${_rtxns_tools_dir}/slang-${SHADERMAKE_SLANG_VERSION}-${_slang_os}-${_slang_cpu}")

if(NOT SHADERMAKE_SLANG_PATH)
    rtxns_fetch_archive(Slang
        "https://github.com/shader-slang/slang/releases/download/v${SHADERMAKE_SLANG_VERSION}/slang-${SHADERMAKE_SLANG_VERSION}-${_slang_os}-${_slang_cpu}.zip"
        "${_slang_dir}" "${_slang_exe}")
    set(SHADERMAKE_SLANG_PATH "${_slang_dir}/${_slang_exe}")

    # Slang's -dxc-path does not reliably locate DXC. Prepare SDK-managed Slang
    # packages here, including when this SDK is embedded. External Slang is untouched.
    if(CMAKE_HOST_WIN32 AND SHADERMAKE_DXC_PATH)
        cmake_path(GET SHADERMAKE_SLANG_PATH PARENT_PATH _slang_bin_dir)
        foreach(_dll dxcompiler.dll dxil.dll)
            cmake_path(REPLACE_FILENAME SHADERMAKE_DXC_PATH "${_dll}" OUTPUT_VARIABLE _dxc_dll)
            if(NOT EXISTS "${_dxc_dll}")
                message(FATAL_ERROR "RTXNS: DXC runtime '${_dxc_dll}' is required by SDK-managed Slang. "
                    "Provide a complete DXC package or a prepared SHADERMAKE_SLANG_PATH.")
            endif()
            file(COPY_FILE "${_dxc_dll}" "${_slang_bin_dir}/${_dll}" ONLY_IF_DIFFERENT)
        endforeach()
    endif()
endif()
