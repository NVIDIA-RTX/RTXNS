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

import CooperativeVectorDerivatives;
import CooperativeVectorFunctions;
import Utils;
import LinearOps;


DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams                    :REGISTER_SRV(0, 0);
Texture2D<float4> inputTexture                  :REGISTER_SRV(1, 0);
RWByteAddressBuffer gMLPParamsGradients         :REGISTER_UAV(0, 0);
RWStructuredBuffer<uint> gRandState             :REGISTER_UAV(1, 0);
RWTexture2D<float4> outputTexture               :REGISTER_UAV(2, 0);
RWTexture2D<float4> lossTexture                 :REGISTER_UAV(3, 0);

struct RNG
{
    uint state;

    __init(uint state) { this.state = state; }

    [mutating]
    float next()
    {
        float r = (state >> 8) * 0x1p-24;
        state = state * 2739110765U + 2739110765U;
        return r;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)] 
void training_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 batchSize = uint2(gConst.batchSizeX, gConst.batchSizeY);

    uint dispatchThreadIdxy = dispatchThreadID.y * batchSize.x + dispatchThreadID.x;

    RNG rng = RNG(gRandState[dispatchThreadIdxy]);

    // Get a random uv coordinate for the input and frequency encode it for improved convergance
    float2 inputUV = clamp(float2(rng.next(), rng.next()), 0.0, 1.0);
    CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams = rtxns::EncodeFrequency<half, 2>({inputUV.x, inputUV.y});

     // Load offsets
    uint weightOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets);
    uint biasOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets);

    // Create variables to cache the results from each stage
    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams[NUM_HIDDEN_LAYERS];
    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenActivated[NUM_HIDDEN_LAYERS];
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputActivated;

    // Forward propagation through the neural network
    // Input to hidden layer, then apply activation function
    hiddenParams[0] = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
        inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], MATRIX_LAYOUT, TYPE_INTERPRETATION);
    hiddenActivated[0] = rtxns::leakyReLU(hiddenParams[0], RELU_LEAK);

    // Hidden layers to hidden layers, then apply activation function 
    [ForceUnroll]
    for (uint layer = 1; layer < NUM_HIDDEN_LAYERS; layer++)
    {
        hiddenParams[layer] = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
            hiddenActivated[layer - 1], gMLPParams, weightOffsets[layer], biasOffsets[layer], 
            MATRIX_LAYOUT, TYPE_INTERPRETATION);
        hiddenActivated[layer] = rtxns::leakyReLU(hiddenParams[layer], RELU_LEAK);
    }

    // Hidden layer to output layer, then apply final activation function    
    outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
        hiddenActivated[NUM_HIDDEN_LAYERS - 1], gMLPParams, weightOffsets[NUM_HIDDEN_LAYERS],
        biasOffsets[NUM_HIDDEN_LAYERS], MATRIX_LAYOUT, TYPE_INTERPRETATION);
    outputActivated = rtxns::sigmoid(outputParams);

    // Take the output from the neural network as the output color
    float3 predictedRGB = {outputActivated[0], outputActivated[1], outputActivated[2]};

    // Now transform the input UVs according to the NetworkModel enum.
    // This can easily be extended to try many different transforms.
    uint2 actualUV;
    if (gConst.networkTransform == NetworkTransform.Flip)
    {
        float2 flipUV = inputUV.yx;
        actualUV = uint2(flipUV.xy * float2(gConst.imageHeight, gConst.imageWidth));
    }
    else if (gConst.networkTransform == NetworkTransform.Zoom)
    {
        float2 zoomUV = inputUV * 0.5 + 0.25;
        actualUV = uint2(zoomUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
    }
    else
    {
        actualUV = uint2(inputUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
    }

    // Load the texture according to the transformed input UVs. This will
    // provide the RGB that the model is trying to train towards.
    float3 actualRGB = inputTexture[actualUV].rgb;

    // Output the loss, scaled to greyscale for output
    uint2 lossUV = uint2(inputUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
    const float lossScaleFactor = 10.0f; // scale it up for better vis
    lossTexture[lossUV] = float4((predictedRGB - actualRGB) * lossScaleFactor + 0.5, 1);  

    // Compute the L2 loss gradient
    // L2Loss = (a-b)^2
    // L2Loss Derivative = 2(a-b)
    float3 lossGradient = 2.0 * (predictedRGB - actualRGB);
   
    // Scale by batch size 
    lossGradient /= (batchSize.x * batchSize.y);

    // Apply the LOSS_SCALE factor to retain precision. Remove it in the optimizer pass before use.
    lossGradient *= LOSS_SCALE;

    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> lossGradientCV = CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS>(VECTOR_FORMAT(lossGradient[0]), VECTOR_FORMAT(lossGradient[1]), VECTOR_FORMAT(lossGradient[2]));

    // Back-propogation pass, generate the gradients and accumulate the results into memory to be applied in the optimisation pass.
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputGradient;
    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenGradient;

    // Output layer (loss gradient) to final hidden layer
    outputGradient = rtxns::sigmoid_Derivative(outputParams, lossGradientCV);
    hiddenGradient = rtxns::LinearOp_Backward<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
       hiddenActivated[NUM_HIDDEN_LAYERS - 1], outputGradient, gMLPParams, gMLPParamsGradients, 
       weightOffsets[NUM_HIDDEN_LAYERS], biasOffsets[NUM_HIDDEN_LAYERS], MATRIX_LAYOUT, TYPE_INTERPRETATION);

    // Hidden layer to hidden layer 
    for(int layer = NUM_HIDDEN_LAYERS - 1; layer >= 1; layer--)
    {
        hiddenGradient = rtxns::leakyReLU_Derivative(hiddenParams[layer], RELU_LEAK, hiddenGradient);
        hiddenGradient = rtxns::LinearOp_Backward<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>
            (hiddenActivated[layer - 1], hiddenGradient, gMLPParams, gMLPParamsGradients, 
            weightOffsets[layer], biasOffsets[layer], MATRIX_LAYOUT, TYPE_INTERPRETATION);
    }

    // First hidden layer to input layer
    hiddenGradient = rtxns::leakyReLU_Derivative(hiddenParams[0], RELU_LEAK, hiddenGradient);
    rtxns::LinearOp_Backward<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
        inputParams, hiddenGradient, gMLPParams, gMLPParamsGradients, weightOffsets[0], 
        biasOffsets[0], MATRIX_LAYOUT, TYPE_INTERPRETATION);

    // Store the random state to continue iterating next time.
    gRandState[dispatchThreadIdxy] = rng.state;
}