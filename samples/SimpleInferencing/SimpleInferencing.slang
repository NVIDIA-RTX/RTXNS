/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

import CooperativeVectorFunctions;
import Utils;
import LinearOps;

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
StructuredBuffer<half> gMLPParams   :REGISTER_SRV(0, 0);

[shader("vertex")]
void main_vs(float3 i_pos, float3 i_norm, out float4 o_pos: SV_Position, out float3 o_norm, out float3 o_view)
{
    o_pos = mul(float4(i_pos, 1), gConst.viewProject);
    o_norm = i_norm;
    o_view = gConst.cameraPos.xyz - i_pos;
}

float4 DisneyMLP(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{   
    uint4 weightOffsets = gConst.weightOffsets; 
    uint4 biasOffsets = gConst.biasOffsets;  

    CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams;
    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams;
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;

    // Encode input parameters, 5 inputs to 30 parameters 
    float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
    inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

    // Forward propagation through the neural network
    // Input to hidden layer, then apply activation function
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
        inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    hiddenParams = rtxns::relu(hiddenParams);

    // Hidden layer to hidden layer, then apply activation function 
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
        hiddenParams, gMLPParams, weightOffsets[1], biasOffsets[1], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    hiddenParams = rtxns::relu(hiddenParams);

    // Hidden layer to hidden layer, then apply activation function    
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
        hiddenParams, gMLPParams, weightOffsets[2], biasOffsets[2], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    hiddenParams = rtxns::relu(hiddenParams);

    // Hidden layer to output layer, then apply final activation function
    outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
        hiddenParams, gMLPParams, weightOffsets[3], biasOffsets[3], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    outputParams = exp(outputParams);

    // Take the output from the neural network as the output color
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}

[shader("fragment")]
void main_ps(float3 i_norm, float3 i_view, out float4 o_color: SV_Target0)
{
    float4 lightIntensity = gConst.lightIntensity;
    float4 lightDir =  gConst.lightDir;
    float4 baseColor = gConst.baseColor;
    float specular = gConst.specular;
    float roughness = gConst.roughness;
    float metallic = gConst.metallic;

    // Prepare input parameters
    float3 view = normalize(i_view);
    float3 norm = normalize(i_norm);
    float3 h = normalize(-lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -lightDir.xyz));

    // Calculate approximated core shader part using MLP
    float4 outParams = DisneyMLP(NdotL, NdotV, NdotH, LdotH, roughness);

    // Calculate final color
    float3 Cdlin = float3(pow(baseColor[0], 2.2), pow(baseColor[1], 2.2), pow(baseColor[2], 2.2));
    float3 Cspec0 = lerp(specular * .08 * float3(1), Cdlin, metallic);
    float3 brdfn = outParams.x * Cdlin * (1 - metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * float3(NdotL) * lightIntensity.rgb;

    o_color = float4(colorh, 1.f);
}
