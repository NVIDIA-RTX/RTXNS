#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include(compileshaders)

set(shader_includes
	${CMAKE_CURRENT_SOURCE_DIR}
)

set(SHADER_COMPILE_OPTIONS "--matrixRowMajor --hlsl2021" )

set(SHADER_COMPILE_OPTIONS_SPIRV "-X \"-Wno-41017 -capability spvCooperativeVectorNV -capability spvCooperativeVectorTrainingNV -capability spvAtomicFloat32AddEXT -capability spvAtomicFloat32MinMaxEXT\" " )

set(SHADER_COMPILE_OPTIONS_DXIL "-D DXIL --hlsl2021 -X \"-Wno-41012 -Wno-41016 -Wno-41017 -Xdxc -I${NVAPI_INCLUDE_DIR} -Xdxc -Vd\" " )

set(SHADER_MODEL "6_9")

set(project Utils)
set(folder "${LIBRARY_FILTER}/Utils")

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp" "*.h")
file(GLOB_RECURSE ${project}_shaders CONFIGURE_DEPENDS "*.hlsl" "*.hlsli" "*.slang")

donut_compile_shaders_all_platforms(
    TARGET ${project}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
	INCLUDES ${shader_includes}
    FOLDER ${folder}
    OUTPUT_BASE ${RTXNS_BINARY_DIR}/shaders/${project}
	SHADERMAKE_OPTIONS ${SHADER_COMPILE_OPTIONS}
    SHADERMAKE_OPTIONS_SPIRV ${SHADER_COMPILE_OPTIONS_SPIRV}
	SHADERMAKE_OPTIONS_DXIL ${SHADER_COMPILE_OPTIONS_DXIL}
	SHADER_MODEL ${SHADER_MODEL}
    SOURCES ${${project}_shaders}
    SLANG
)

add_library(${project} STATIC EXCLUDE_FROM_ALL ${sources})
target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (DONUT_WITH_DX12)
	target_link_libraries(${project} donut_app donut_engine NeuralShading nvapi implot)
    add_dependencies(${project} dx12-agility-sdk ${project}_shaders)
    target_compile_definitions(
		${project} PRIVATE DONUT_D3D_AGILITY_SDK_VERSION=${DONUT_D3D_AGILITY_SDK_VERSION}
		${project} PRIVATE DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION=${DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION}
	)
else()
	target_link_libraries(${project} donut_app donut_engine NeuralShading implot)
    add_dependencies(${project} ${project}_shaders)
endif()

set_target_properties(${project} PROPERTIES 
	FOLDER ${folder}
)

set(SAMPLE_UTILITES_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "" FORCE)
