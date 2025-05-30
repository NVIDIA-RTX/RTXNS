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