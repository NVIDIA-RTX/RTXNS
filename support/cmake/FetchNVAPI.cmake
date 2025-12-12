include(FetchContent)

set(NVAPI_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/nvapi/" CACHE STRING "Path to NVAPI include headers/shaders" )
set(NVAPI_LIBRARY "${CMAKE_SOURCE_DIR}/external/nvapi/amd64/nvapi64.lib" CACHE STRING "Path to NVAPI .lib file")

FetchContent_Declare(
	nvapi
	GIT_REPOSITORY https://github.com/NVIDIA/nvapi.git
	GIT_TAG d08488fcc82eef313b0464db37d2955709691e94
	SOURCE_DIR "${NVAPI_INCLUDE_DIR}"
)
FetchContent_MakeAvailable(nvapi)
add_library(nvapi STATIC IMPORTED GLOBAL)
target_include_directories(nvapi INTERFACE "${NVAPI_INCLUDE_DIR}")
set_property(TARGET nvapi PROPERTY IMPORTED_LOCATION "${NVAPI_LIBRARY}")