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

import CooperativeVectorFunctions;
import Utils;
import LinearOps;

DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams            :REGISTER_SRV(0, 0);
Texture2D<float4> inputTexture          :REGISTER_SRV(1, 0);
RWTexture2D<float4> outputTexture       :REGISTER_UAV(0, 0);

[shader("compute")]
[numthreads(8, 8, 1)] 
void inference_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Set the input ID as the uv coordinate and frequency encode it for the network
    float2 inputUV = float2(dispatchThreadID.x / float(gConst.imageWidth), dispatchThreadID.y / float(gConst.imageHeight));
    CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams = rtxns::EncodeFrequency<half, 2>({inputUV.x, inputUV.y});

    // Load offsets
    uint weightOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets);
    uint biasOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets);

    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams;
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;
   
    // Forward propagation through the neural network
    // Input to hidden layer, then apply activation function
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
        inputParams, gMLPParams, weightOffsets[0], biasOffsets[0],
        MATRIX_LAYOUT, TYPE_INTERPRETATION);
    hiddenParams = rtxns::leakyReLU(hiddenParams, RELU_LEAK);

    // Hidden layers to hidden layers, then apply activation function 
    [ForceUnroll]
    for (uint layer = 1; layer < NUM_HIDDEN_LAYERS; layer++)
    {
        hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
            hiddenParams, gMLPParams, weightOffsets[layer], 
            biasOffsets[layer], 
            MATRIX_LAYOUT, TYPE_INTERPRETATION);
        hiddenParams = rtxns::leakyReLU(hiddenParams, RELU_LEAK);
    }

    // Hidden layer to output layer, then apply final activation function
    outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
        hiddenParams, gMLPParams, weightOffsets[NUM_HIDDEN_LAYERS], biasOffsets[NUM_HIDDEN_LAYERS],
        MATRIX_LAYOUT, TYPE_INTERPRETATION);
    outputParams = rtxns::sigmoid(outputParams);

    // Take the output from the neural network as the output color
    float4 color = {outputParams[0], outputParams[1], outputParams[2], 1.f};
    outputTexture[dispatchThreadID.xy] = color;
}