#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

set(NS_SAMPLE_SHADER_INCLUDES_COMMON ${NS_SAMPLE_SHADER_INCLUDE_DIR})

set(NS_SAMPLE_SHADER_COMPILE_OPTIONS "--matrixRowMajor --hlsl2021 --embedPDB" )
set(NS_SAMPLE_SHADER_COMPILE_OPTIONS_SPIRV " -X \"-Wno-41016 -Wno-41017 -capability spvCooperativeVectorNV -capability spvCooperativeVectorTrainingNV\" " )
if(DXC_HLSL_INCLUDE_DIR)
    set(NS_SAMPLE_SHADER_COMPILE_OPTIONS_DXIL " -X \"-Wno-41012 -Wno-41016 -Wno-41017 -Xdxc -I${DXC_HLSL_INCLUDE_DIR} -Xdxc -Vd\" " )
else()
    set(NS_SAMPLE_SHADER_COMPILE_OPTIONS_DXIL " -X \"-Wno-41012 -Wno-41016 -Wno-41017 -Xdxc -Vd\" " )
endif()

set(NS_SAMPLE_SHADER_MODEL "${RTXNS_DX12_SHADER_MODEL}")

add_subdirectory(SimpleInferencing)
add_subdirectory(ShaderTraining)
add_subdirectory(SimpleTraining)
add_subdirectory(SlangpyTraining)
add_subdirectory(SlangpyInferencing)
