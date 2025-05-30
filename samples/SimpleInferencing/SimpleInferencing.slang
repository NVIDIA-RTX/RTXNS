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
ByteAddressBuffer gMLPParams            :REGISTER_SRV(0, 0);

struct VertexIn
{
    float3 pos : POSITION; 
    float3 norm : NORMAL; 
};

struct VertexOut
{
    float4 pos : SV_Position;
    float3 norm : NORMAL;
    float3 view : VIEW;
}

[shader("vertex")]
void main_vs(
    VertexIn vIn,
    out VertexOut vOut)
{
    vOut.pos = mul(float4(vIn.pos, 1), gConst.viewProject);
    vOut.norm = vIn.norm;
    vOut.view = gConst.cameraPos.xyz - vIn.pos;
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
void main_ps( 
    VertexOut vOut,
    out float4 o_color : SV_Target0)
{
    float4 lightIntensity = gConst.lightIntensity;
    float4 lightDir =  gConst.lightDir;
    float4 baseColor = gConst.baseColor;
    float specular = gConst.specular;
    float roughness = gConst.roughness;
    float metallic = gConst.metallic;

    // Prepare input parameters
    float3 view = normalize(vOut.view);
    float3 norm = normalize(vOut.norm);
    float3 h = normalize(-lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -lightDir.xyz));

    // Calculate approximated core shader part using MLP
    float4 outParams = DisneyMLP(NdotL, NdotV, NdotH, LdotH, roughness);

    // Calculate final color
    float3 Cdlin = float3(pow(baseColor.r, 2.2), pow(baseColor.g, 2.2), pow(baseColor.b, 2.2));
    float3 Cspec0 = lerp(specular * .08f * float3(1,1,1), Cdlin, metallic);
    float3 brdfn = outParams.x * Cdlin * (1 - metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * float3(NdotL) * lightIntensity.rgb;

    o_color = float4(colorh, 1.f);
 }
