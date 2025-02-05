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

#include "DisneyMLP.h"

DECLARE_CBUFFER(InferenceConstantBufferEntry, gConst, 0, 0);
StructuredBuffer<half> gMLPParams : REGISTER_SRV(0, 0);

[shader("vertex")]
void main_vs(float3 i_pos, float3 i_norm, out float4 o_pos : SV_Position, out float3 o_norm, out float3 o_view)
{
    o_pos = mul(float4(i_pos, 1), gConst.viewProject);
    o_norm = i_norm;
    o_view = gConst.cameraPos.xyz - i_pos;
}

[shader("fragment")]
void main_ps(float3 i_norm, float3 i_view, out float4 o_color : SV_Target0)
{
    // Prepare input parameters
    float3 view = normalize(i_view);
    float3 norm = normalize(i_norm);
    float3 h = normalize(-gConst.lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -gConst.lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -gConst.lightDir.xyz));

    // Calculate approximated core shader
    float4 outParams = DisneyMLP(NdotL, NdotV, NdotH, LdotH, gConst.roughness);

    // Calculate final color
    float3 Cdlin = pow(gConst.baseColor.rgb, 2.2);
    float3 Cspec0 = lerp(gConst.specular * float3(.08f), Cdlin, gConst.metallic);
    float3 brdfn = outParams.x * Cdlin * (1 - gConst.metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * NdotL * gConst.lightIntensity.rgb;

    o_color = float4(colorh, 1.f);
}
