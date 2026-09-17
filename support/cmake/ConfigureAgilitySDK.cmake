# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(target_path "${RTXNS_OUTPUT_DIR}/d3d12/")

# Donut's FetchAgilitySDK.cmake selects the DLL folder from CMAKE_SYSTEM_PROCESSOR / the Visual Studio
# generator platform. Re-resolve it against the project's own target architecture so that an explicit
# RTXNS_TARGET_ARCH, or a toolchain that only sets the compiler, picks the matching binaries.
# The NuGet package ships build/native/bin/{x64,win32,arm64}.
if(RTXNS_TARGET_ARCH STREQUAL "x64")
    set(_agility_arch "x64")
else()
    set(_agility_arch "arm64")
endif()
set(_agility_bin_dir "${DONUT_D3D_AGILITY_SDK_PATH}/build/native/bin/${_agility_arch}")
if(EXISTS "${_agility_bin_dir}/D3D12Core.dll" AND EXISTS "${_agility_bin_dir}/d3d12SDKLayers.dll")
    set(DONUT_D3D_AGILITY_SDK_CORE_DLL "${_agility_bin_dir}/D3D12Core.dll" CACHE FILEPATH "D3D12 Agility SDK core DLL" FORCE)
    set(DONUT_D3D_AGILITY_SDK_LAYERS_DLL "${_agility_bin_dir}/d3d12SDKLayers.dll" CACHE FILEPATH "D3D12 Agility SDK debug layer DLL" FORCE)
    set(DONUT_D3D_AGILITY_SDK_LIBRARIES "${DONUT_D3D_AGILITY_SDK_CORE_DLL}" "${DONUT_D3D_AGILITY_SDK_LAYERS_DLL}")
    message(STATUS "Agility SDK: using ${_agility_arch} binaries for target '${RTXNS_TARGET_ARCH}' from ${_agility_bin_dir}")
else()
    message(SEND_ERROR "Agility SDK binaries for target architecture '${RTXNS_TARGET_ARCH}' were not found in '${_agility_bin_dir}'")
endif()

# Find the Agility Preview SDK version number
if(_d3d_agility_include)
    
    #set(DONUT_D3D_AGILITY_SDK_INCLUDE_DIR "${_d3d_agility_include}")
    
    # find the SDK version number
    file(READ "${_d3d_agility_include}/d3d12.idl" _d3d12_idl)
    string(REGEX MATCH "const UINT D3D12_PREVIEW_SDK_VERSION = ([0-9]+)" _match ${_d3d12_idl})
    if(_match AND CMAKE_MATCH_1)
        set(DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION ${CMAKE_MATCH_1})
        message(STATUS "Found D3D12 Agility Preview SDK: ${DONUT_D3D_AGILITY_SDK_INCLUDE_DIR} (version ${DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION})")
    else()
        message(FATAL_ERROR "Cannot resolve D3D12 Agility Preview SDK version number")
    endif()    
endif()

if (NOT DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION OR NOT DONUT_D3D_AGILITY_SDK_LIBRARIES)
    message(SEND_ERROR "Agility SDK variables were not configured, please re-configure the project to download it.")
endif()

add_custom_target(dx12-agility-sdk)
set_property (TARGET dx12-agility-sdk PROPERTY FOLDER "Third-Party Libraries")

add_custom_command(TARGET dx12-agility-sdk POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${target_path}")

foreach (filename ${DONUT_D3D_AGILITY_SDK_LIBRARIES})
    add_custom_command(TARGET dx12-agility-sdk POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${filename}" "${target_path}")

endforeach()
