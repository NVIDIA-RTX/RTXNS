#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


set(project NeuralShading_Shaders)
set(folder "${LIBRARY_FILTER}/NeuralShading_Shaders")

file(GLOB shaders "*.slang")

set_source_files_properties(${shaders} PROPERTIES VS_TOOL_OVERRIDE "None") 
add_custom_target(${project}
	DEPENDS ShaderMake
    SOURCES ${shaders})
	set_target_properties(${project} PROPERTIES 
	FOLDER ${folder}
)

set(SAMPLES_SHADER_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "" FORCE)