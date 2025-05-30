#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include(../../external/donut/compileshaders.cmake)

set(shader_includes
	${SAMPLES_SHADER_INCLUDE_DIR}
	${CMAKE_CURRENT_LIST_DIR}
)

set(SHADER_COMPILE_OPTIONS "--matrixRowMajor --hlsl2021" )

set(SHADER_COMPILE_OPTIONS_SPIRV " -X \"-Wno-41017 -capability spvCooperativeVectorNV -capability spvCooperativeVectorTrainingNV\" " )

set(SHADER_COMPILE_OPTIONS_DXIL " --shaderModel 6_9 --hlsl2021 -X \"-Wno-41012 -Wno-41016 -Wno-41017 -Xdxc -Vd\" " )

set(project SlangpyTraining)
set(folder "Samples/SlangpyTraining")

file(GLOB_RECURSE ${project}_shaders "*.hlsl" "*.hlsli" "*.slang")
file(GLOB_RECURSE ${project}_sources "*.cpp" "*.h" "*.md")

donut_compile_shaders_all_platforms(
    TARGET ${project}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
	INCLUDES ${shader_includes}
    FOLDER ${folder}
    OUTPUT_BASE ${RTXNS_BINARY_DIR}/shaders/${project}
	SHADERMAKE_OPTIONS ${SHADER_COMPILE_OPTIONS}
    SHADERMAKE_OPTIONS_SPIRV ${SHADER_COMPILE_OPTIONS_SPIRV}
	SHADERMAKE_OPTIONS_DXIL ${SHADER_COMPILE_OPTIONS_DXIL}
    SOURCES ${${project}_shaders}
    SLANG
)

add_executable(${project} WIN32 ${${project}_sources})
target_link_libraries(${project} donut_app donut_engine NeuralShading Utils)
add_dependencies(${project} ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()