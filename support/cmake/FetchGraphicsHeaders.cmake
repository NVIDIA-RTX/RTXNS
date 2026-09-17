#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Provides the Vulkan-Headers and DirectX-Headers sources that NVRHI's Vulkan and D3D12 backends compile
# against, without modifying Donut or NVRHI.
#
# NVRHI fetches both with FetchContent from git (external/donut/nvrhi/CMakeLists.txt), which clones the
# full history (about 130 MB for Vulkan-Headers and 24 MB for DirectX-Headers) into every build
# directory. This module instead downloads the tag archives (a few MB) once into versioned folders
# under external/ shared by all build directories, and sets FETCHCONTENT_SOURCE_DIR_<NAME> for NVRHI's
# FetchContent dependency names. FetchContent honours that variable by skipping the download and adding
# the given directory, so NVRHI's own FetchContent_Declare()/FetchContent_MakeAvailable() calls run
# unchanged and create the usual targets. NVRHI still prints its "Fetching ... from git" status line;
# no fetch happens.
#
#   RTXNS_VULKAN_HEADERS_ROOT / RTXNS_DIRECTX_HEADERS_ROOT   Optional existing checkouts to use
#                                                             instead of the downloads.
#
# Expects: DONUT_WITH_VULKAN, DONUT_WITH_DX12, RTXNS_DIRECTX_HEADERS_VERSION (top-level CMakeLists.txt)
#          and DONUT_DIR. Include before add_subdirectory(donut).

include("${CMAKE_CURRENT_LIST_DIR}/FetchArchive.cmake")

# Build directories configured before this module existed may still carry the git fetch directories
# that used to be forced into NVRHI's cache; they are unused now. NVRHI re-declares them empty.
unset(NVRHI_VULKAN_HEADERS_FETCH_DIR CACHE)
unset(NVRHI_DIRECTX_HEADERS_FETCH_DIR CACHE)

# Resolves one dependency: uses <root_override> when set, otherwise downloads <url> into
# external/<folder> (checked with <probe>), then points NVRHI's FetchContent name <fc_name> at it.
function(_rtxns_provide_headers display fc_name root_override url folder probe)
    if(root_override)
        set(_dir "${root_override}")
        if(NOT EXISTS "${_dir}/${probe}")
            message(FATAL_ERROR "${display}: '${_dir}' does not contain ${probe}")
        endif()
        message(STATUS "${display}: using ${_dir}")
    else()
        set(_dir "${CMAKE_SOURCE_DIR}/external/${folder}")
        rtxns_fetch_archive("${display}" "${url}" "${_dir}" "${probe}")
    endif()
    if(NOT EXISTS "${_dir}/CMakeLists.txt")
        message(FATAL_ERROR "${display}: '${_dir}' has no CMakeLists.txt; NVRHI adds this directory with add_subdirectory()")
    endif()

    string(TOUPPER "${fc_name}" _fc_upper)
    # INTERNAL implies FORCE: the value follows the pinned version, so a build directory configured before
    # a version bump must not keep the old folder. FetchContent's own non-forcing set() of this variable
    # then leaves it alone. Use the *_ROOT variables above to point somewhere else.
    set(FETCHCONTENT_SOURCE_DIR_${_fc_upper} "${_dir}" CACHE INTERNAL "Pre-populated source for NVRHI's ${fc_name} dependency (managed by RTXNS)")
endfunction()

# ---- Vulkan-Headers (github.com/KhronosGroup/Vulkan-Headers) ----
if(DONUT_WITH_VULKAN)
    # Must match the version NVRHI expects (default of NVRHI_VULKAN_HEADERS_GIT_TAG in
    # external/donut/nvrhi/CMakeLists.txt); the check below warns when a Donut update changes it.
    set(RTXNS_VULKAN_HEADERS_VERSION "1.4.352")

    file(STRINGS "${DONUT_DIR}/nvrhi/CMakeLists.txt" _nvrhi_vk_tag_line REGEX "set[(]NVRHI_VULKAN_HEADERS_GIT_TAG \"v[0-9.]+\"")
    if(_nvrhi_vk_tag_line MATCHES "\"v([0-9.]+)\"")
        if(NOT CMAKE_MATCH_1 STREQUAL RTXNS_VULKAN_HEADERS_VERSION)
            message(WARNING "RTXNS pins Vulkan-Headers ${RTXNS_VULKAN_HEADERS_VERSION} but NVRHI now defaults to ${CMAKE_MATCH_1}; "
                            "update RTXNS_VULKAN_HEADERS_VERSION in support/cmake/FetchGraphicsHeaders.cmake.")
        endif()
    endif()

    # Keeps NVRHI's status message truthful about the version in use; the git fetch itself does not run.
    set(NVRHI_VULKAN_HEADERS_GIT_TAG "v${RTXNS_VULKAN_HEADERS_VERSION}" CACHE STRING "Vulkan-Headers version used by NVRHI (pinned by RTXNS)" FORCE)

    set(RTXNS_VULKAN_HEADERS_ROOT "" CACHE PATH "Optional: existing Vulkan-Headers checkout (folder with CMakeLists.txt and include/vulkan). Leave empty to download the pinned version into external/.")
    _rtxns_provide_headers("Vulkan-Headers" vulkan_headers "${RTXNS_VULKAN_HEADERS_ROOT}"
        "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v${RTXNS_VULKAN_HEADERS_VERSION}.zip"
        "vulkan-headers-${RTXNS_VULKAN_HEADERS_VERSION}" "include/vulkan/vulkan.h")
endif()

# ---- DirectX-Headers (github.com/microsoft/DirectX-Headers) ----
if(DONUT_WITH_DX12)
    if(NOT RTXNS_DIRECTX_HEADERS_VERSION)
        message(FATAL_ERROR "FetchGraphicsHeaders.cmake requires RTXNS_DIRECTX_HEADERS_VERSION (set by the DX12 preview toolchain selection)")
    endif()

    # Same purpose as for Vulkan-Headers; the version is chosen to match the Agility SDK preview.
    set(NVRHI_DIRECTX_HEADERS_GIT_TAG "v${RTXNS_DIRECTX_HEADERS_VERSION}" CACHE STRING "DirectX-Headers version used by NVRHI (pinned by RTXNS to match the Agility SDK)" FORCE)

    set(RTXNS_DIRECTX_HEADERS_ROOT "" CACHE PATH "Optional: existing DirectX-Headers checkout (folder with CMakeLists.txt and include/directx). Leave empty to download the pinned version into external/.")
    _rtxns_provide_headers("DirectX-Headers" directx_headers "${RTXNS_DIRECTX_HEADERS_ROOT}"
        "https://github.com/microsoft/DirectX-Headers/archive/refs/tags/v${RTXNS_DIRECTX_HEADERS_VERSION}.zip"
        "directx-headers-${RTXNS_DIRECTX_HEADERS_VERSION}" "include/directx/d3d12.h")
endif()
