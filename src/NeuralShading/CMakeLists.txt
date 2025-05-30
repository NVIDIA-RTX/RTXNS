#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


file(GLOB sources "*.cpp" "*.h")

set(project NeuralShading)
set(folder "${LIBRARY_FILTER}/NeuralShading")

add_library(${project} STATIC EXCLUDE_FROM_ALL ${sources})
target_include_directories(${project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${project} donut_app donut_engine)

set_target_properties(${project} PROPERTIES 
	FOLDER ${folder}
)
