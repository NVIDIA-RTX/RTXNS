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

DECLARE_CBUFFER(DirectConstantBufferEntry, gConst, 0, 0);

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
    output.pos = mul(float4(i_pos, 1), gConst.viewProject);
    output.norm = i_norm;
    output.view = gConst.cameraPos.xyz - i_pos;
}

import Disney;

[shader("fragment")]
void main_ps(
    PS_INPUT input,
    out float4 o_color : SV_Target0)
{
    //----------- Prepare input parameters
    float3 view = normalize(input.view);
    float3 norm = normalize(input.norm);
    float3 h = normalize(-gConst.lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -gConst.lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -gConst.lightDir.xyz));

    //----------- Calculate core shader part DIRECTLY
    float4 outParams = Disney(NdotL, NdotV, NdotH, LdotH, gConst.roughness);

    //----------- Calculate final color
    float3 Cdlin = float3(pow(gConst.baseColor[0], 2.2), pow(gConst.baseColor[1], 2.2), pow(gConst.baseColor[2], 2.2));
    float3 Cspec0 = lerp(gConst.specular * .08 * float3(1), Cdlin, gConst.metallic);
    float3 brdfn = outParams.x * Cdlin * (1 - gConst.metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * float3(NdotL) * gConst.lightIntensity.rgb;

    o_color = float4(colorh, 1.f);
}
