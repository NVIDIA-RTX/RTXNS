#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# rtxns_fetch_archive(<name> <url> <dir> <probe>)
#
# Downloads the archive at <url> and extracts it into <dir>, unless <dir>/<probe> already exists. A
# single top-level directory inside the archive (GitHub source archives have one) is stripped, so the
# archive contents end up directly in <dir>.
#
# <dir> is meant to be a versioned folder under external/ that all build directories share: nothing is
# downloaded twice, a version bump changes the folder name, and configuring one build directory never
# removes a dependency another one is using. FetchContent is deliberately not used for these shared
# folders: it tracks its work per build directory, deletes the destination before re-extracting, and
# FetchContent_MakeAvailable() would add_subdirectory() any archive that ships a CMakeLists.txt.

include_guard(GLOBAL)

function(rtxns_fetch_archive name url dir probe)
    if(EXISTS "${dir}/${probe}")
        message(STATUS "RTXNS: using existing ${name} in ${dir}")
        return()
    endif()

    string(REGEX REPLACE ".*/" "" _archive_name "${url}")
    # The download and the extraction staging area live next to the destination so that the final step
    # is a rename on the same file system; both are removed again afterwards.
    set(_download_dir "${dir}.download")
    set(_staging_dir "${dir}.extract")
    set(_archive "${_download_dir}/${_archive_name}")
    file(REMOVE_RECURSE "${dir}" "${_download_dir}" "${_staging_dir}")
    file(MAKE_DIRECTORY "${_download_dir}")

    message(STATUS "RTXNS: downloading ${name} from ${url}...")
    set(_code 1)
    set(_error "")
    foreach(_attempt RANGE 1 3)
        file(DOWNLOAD "${url}" "${_archive}" STATUS _status INACTIVITY_TIMEOUT 60)
        list(GET _status 0 _code)
        list(GET _status 1 _error)
        if(_code EQUAL 0)
            break()
        endif()
        message(STATUS "RTXNS: download attempt ${_attempt} of ${name} failed: ${_error}")
        file(REMOVE "${_archive}")
    endforeach()
    if(NOT _code EQUAL 0)
        file(REMOVE_RECURSE "${_download_dir}")
        message(FATAL_ERROR "RTXNS: could not download ${name} from ${url}: ${_error}")
    endif()

    file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_staging_dir}")
    file(REMOVE_RECURSE "${_download_dir}")

    # Strip a single top-level directory (the layout of GitHub source archives).
    file(GLOB _entries LIST_DIRECTORIES TRUE "${_staging_dir}/*")
    list(LENGTH _entries _count)
    if(_count EQUAL 1 AND IS_DIRECTORY "${_entries}")
        file(RENAME "${_entries}" "${dir}")
        file(REMOVE_RECURSE "${_staging_dir}")
    else()
        file(RENAME "${_staging_dir}" "${dir}")
    endif()

    if(NOT EXISTS "${dir}/${probe}")
        file(REMOVE_RECURSE "${dir}")
        message(FATAL_ERROR "RTXNS: ${name} was extracted to '${dir}' but '${probe}' is missing from the archive")
    endif()
    message(STATUS "RTXNS: ${name} ready in ${dir}")
endfunction()
