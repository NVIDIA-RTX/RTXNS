/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import SlangpyTraining;

DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams        :REGISTER_SRV(0, 0);
Texture2D<float4> inputTexture      :REGISTER_SRV(1, 0);
RWTexture2D<float4> outputTexture   :REGISTER_UAV(0, 0);

float3 evalModel(ByteAddressBuffer weights, uint wo[MAX_LAYER_COUNT], uint bo[MAX_LAYER_COUNT], float2 uv)
{
    // Auto-generated defines from texture-training.py
    MODEL_TYPE model = MODEL_INITIALIZER;

    let inputParams = rtxns::CoopVecFromVector<VECTOR_FORMAT>(uv);

    let result = model.forward(inputParams);

    return rtxns::VectorFromCoopVec(result);
}

[shader("compute")]
[numthreads(8, 8, 1)] 
void main_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Get the UV coordinate from the thread ID
    float2 inputUV = float2(dispatchThreadID.x / float(gConst.imageWidth), dispatchThreadID.y / float(gConst.imageHeight));

    // Load offsets
    uint weightOffsets[MAX_LAYER_COUNT] = rtxns::UnpackArray<MAX_LAYER_COUNT_ALIGN4, MAX_LAYER_COUNT>(gConst.weightOffsets);
    uint biasOffsets[MAX_LAYER_COUNT] = rtxns::UnpackArray<MAX_LAYER_COUNT_ALIGN4, MAX_LAYER_COUNT>(gConst.biasOffsets);

    // Run the model
    float3 modelOutput = evalModel(gMLPParams, weightOffsets, biasOffsets, inputUV);

    // Write to output
    outputTexture[dispatchThreadID.xy] = float4(modelOutput, 1.0f);
}