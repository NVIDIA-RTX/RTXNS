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

import CooperativeVectorAutoDiff;
import CooperativeVectorFunctions;
import Utils;
import Activation;
import MLP;
import Disney;
import DisneyMLP;

DECLARE_CBUFFER(InferenceConstantBufferEntry, gConst, 0, 0);
ByteAddressBuffer gMLPParams : REGISTER_SRV(0, 0);

struct PS_INPUT
{
    float4 pos : SV_Position;
    float3 norm : NORMAL;
    float3 view : VIEW;
}

[shader("vertex")]
void main_vs(
    float3 i_pos : POSITION, 
    float3 i_norm : NORMAL, 
    out PS_INPUT output)
{
    output.pos = mul(float4(i_pos, 1), gConst.directConstants.viewProject);
    output.norm = i_norm;
    output.view = gConst.directConstants.cameraPos.xyz - i_pos;
}

float3 calcColor(float4 params)
{
    float3 Cdlin = pow(gConst.directConstants.baseColor.rgb, 2.2);
    float3 Cspec0 = lerp(gConst.directConstants.specular * float3(.08f), Cdlin, gConst.directConstants.metallic);
    float3 brdfn = params.x * Cdlin * (1 - gConst.directConstants.metallic) + params.y * lerp(Cspec0, float3(1), params.z) + params.w;
    return clamp(brdfn * gConst.directConstants.lightIntensity.rgb, 0, 1);
}

[shader("fragment")]
void main_ps(
    PS_INPUT input,
    out float4 o_color : SV_Target0)
{
    DirectConstantBufferEntry directConstants = gConst.directConstants;

    // Prepare input parameters
    float3 view = normalize(input.view);
    float3 norm = normalize(input.norm);
    float3 h = normalize(-directConstants.lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -directConstants.lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -directConstants.lightDir.xyz));

    //----------- Calculate core shader part DIRECTLY
    float4 actualDisney = Disney(NdotL, NdotV, NdotH, LdotH, directConstants.roughness);

    // Calculate approximated core shader
    float4 outParams = DisneyMLP<NUM_HIDDEN_LAYERS, HIDDEN_NEURONS>(
        NdotL, NdotV, NdotH, LdotH, directConstants.roughness,
        gMLPParams,
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets)
    );

    o_color = float4((calcColor(actualDisney) - calcColor(outParams)) * NdotL * 4 + 0.5, 1.f);
}
