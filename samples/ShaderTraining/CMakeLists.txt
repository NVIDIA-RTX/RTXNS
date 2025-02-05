#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include(../../external/donut/compileshaders.cmake)

set(SHADER_COMPILE_OPTIONS "--matrixRowMajor -I ${SAMPLES_SHADER_INCLUDE_DIR} -I ${CMAKE_CURRENT_LIST_DIR} -X \"-capability spvCooperativeVectorNV -capability spvCooperativeVectorTrainingNV\" " )

set(project ShaderTraining)
set(folder "Samples/ShaderTraining")

file(GLOB ${project}_shaders "*.slang")
file(GLOB ${project}_sources "*.cpp" "*.h")

donut_compile_shaders_all_platforms(
    TARGET ${project}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    FOLDER ${folder}
    OUTPUT_BASE ${RTXNS_BINARY_DIR}/shaders/${project}
    SHADERMAKE_OPTIONS_SPIRV ${SHADER_COMPILE_OPTIONS}
    SOURCES ${${project}_shaders}
    SLANG
)

add_executable(${project} WIN32 ${${project}_sources})
target_link_libraries(${project} donut_app donut_engine NeuralShading Utils)
add_dependencies(${project} ${project}_shaders)

set_target_properties(${project} PROPERTIES 
	FOLDER ${folder}
	RUNTIME_OUTPUT_DIRECTORY                "${RTXNS_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${RTXNS_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${RTXNS_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${RTXNS_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${RTXNS_BINARY_DIR}"
)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()

