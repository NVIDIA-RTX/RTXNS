/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

float4 DisneyMLP(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{
    // Calculate approximated core shader part using MLP
    float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };

    var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);
    var model = rtxns::mlp::InferenceMLP<half, NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
        gMLPParams, rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

    var outputParams = model.forward(inputParams, rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>(), rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>());

    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}
