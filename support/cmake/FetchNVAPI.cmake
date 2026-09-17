#
# Copyright (c) 2025 - 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include("${CMAKE_CURRENT_LIST_DIR}/FetchArchive.cmake")

if(NOT RTXNS_TARGET_ARCH)
	message(FATAL_ERROR "FetchNVAPI.cmake requires RTXNS_TARGET_ARCH (include support/cmake/TargetArch.cmake first)")
endif()

# NVAPI ships one import library per Windows target architecture.
if(RTXNS_TARGET_ARCH STREQUAL "x64")
	set(_nvapi_lib_relpath "amd64/nvapi64.lib")
elseif(RTXNS_TARGET_ARCH STREQUAL "arm64")
	set(_nvapi_lib_relpath "aarch64/nvapia64.lib")
elseif(RTXNS_TARGET_ARCH STREQUAL "arm64ec")
	set(_nvapi_lib_relpath "arm64ec/nvapia64ec.lib")
else()
	message(FATAL_ERROR "NVAPI does not provide a library for target architecture '${RTXNS_TARGET_ARCH}'")
endif()

# Pinned NVAPI SDK release. R615 is the first release that includes Windows on Arm (aarch64 / arm64ec)
# libraries. Keep the version name and the commit in sync when updating.
set(RTXNS_NVAPI_VERSION "R615")
set(RTXNS_NVAPI_COMMIT "87dca625e83fd89a983e19b904e5f3a580da90d2")

# Optional override for an SDK that is already on disk (folder containing nvapi.h and the amd64/aarch64 libs).
set(RTXNS_NVAPI_ROOT "" CACHE PATH "Path to an existing NVAPI SDK. Leave empty to download the pinned version into external/.")

if(RTXNS_NVAPI_ROOT)
	set(_nvapi_root "${RTXNS_NVAPI_ROOT}")
else()
	# Downloaded as a source archive of the pinned commit (a few MB) rather than cloned with its history
	# (>100 MB), into a versioned folder in the source tree that all build directories share.
	set(_nvapi_root "${CMAKE_SOURCE_DIR}/external/nvapi-${RTXNS_NVAPI_VERSION}")
	rtxns_fetch_archive(NVAPI "https://github.com/NVIDIA/nvapi/archive/${RTXNS_NVAPI_COMMIT}.zip" "${_nvapi_root}" "nvapi.h")
endif()

# Plain variables (not cache entries) so that build directories configured before the download
# location changed do not keep pointing at an old location.
set(NVAPI_INCLUDE_DIR "${_nvapi_root}")
set(NVAPI_LIBRARY "${_nvapi_root}/${_nvapi_lib_relpath}")

if(NOT EXISTS "${NVAPI_INCLUDE_DIR}/nvapi.h")
	message(FATAL_ERROR "NVAPI headers not found in '${NVAPI_INCLUDE_DIR}'. Delete the folder and re-run CMake to download the SDK again, or set RTXNS_NVAPI_ROOT.")
endif()
if(NOT EXISTS "${NVAPI_LIBRARY}")
	message(FATAL_ERROR "NVAPI library for ${RTXNS_TARGET_ARCH} not found at '${NVAPI_LIBRARY}'. Delete '${NVAPI_INCLUDE_DIR}' and re-run CMake to download the SDK again, or set RTXNS_NVAPI_ROOT.")
endif()

add_library(nvapi STATIC IMPORTED GLOBAL)
target_include_directories(nvapi INTERFACE "${NVAPI_INCLUDE_DIR}")
set_property(TARGET nvapi PROPERTY IMPORTED_LOCATION "${NVAPI_LIBRARY}")
message(STATUS "NVAPI: using ${NVAPI_LIBRARY}")
