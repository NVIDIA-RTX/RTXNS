#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


file(GLOB sources "*.cpp" "*.h")

set(project Utils)
set(folder "${LIBRARY_FILTER}/Utils")

add_library(${project} STATIC EXCLUDE_FROM_ALL ${sources})
target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${project} donut_app donut_engine NeuralShading)


if (DONUT_WITH_DX12)
    add_dependencies(${project} dx12-agility-sdk)
    target_compile_definitions(
		${project} PRIVATE DONUT_D3D_AGILITY_SDK_VERSION=${DONUT_D3D_AGILITY_SDK_VERSION}
		${project} PRIVATE DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION=${DONUT_D3D_AGILITY_PREVIEW_SDK_VERSION}
	)
endif()

set_target_properties(${project} PROPERTIES 
	FOLDER ${folder}
)
