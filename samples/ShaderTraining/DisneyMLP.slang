/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

import MLP;
import CooperativeVectorFunctions;
import Activation;
import Utils;

// 5 inputs are passed into this function; NdotL, NdotV, NdotH, LdotH, roughness
#define INPUT_FEATURES 5 

// The output is float4
#define OUTPUT_NEURONS 4

// EncodeFrequency expands the input by 6 per input feature
#define FREQUENCY_EXPANSION 6

float4 DisneyMLP<let HIDDEN_LAYERS : int, let HIDDEN_NEURONS : int>(
    float NdotL, float NdotV, float NdotH, float LdotH, float roughness, ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HIDDEN_LAYERS+1], uint biasOffsets[HIDDEN_LAYERS+1])
{
    // Calculate approximated core shader part using MLP
    float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };

    var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

    var model = rtxns::mlp::InferenceMLP<half, 
        HIDDEN_LAYERS, 
        INPUT_FEATURES * FREQUENCY_EXPANSION, 
        HIDDEN_NEURONS, 
        OUTPUT_NEURONS, 
        CoopVecMatrixLayout::TrainingOptimal, 
        CoopVecComponentType::Float16>
        (mlpBuffer, weightOffsets, biasOffsets);

    var outputParams = model.forward(inputParams, rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>(), rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>());
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}
