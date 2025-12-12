/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "TrainingResults.h"
#include <donut/shaders/binding_helpers.hlsli>

import LossAccumulation;

#if defined(DXIL) && defined(NVAPI_INIT)
// Reserved a space and slot for NVAPI extensions 
#define NV_SHADER_EXTN_REGISTER_SPACE space0
#define NV_SHADER_EXTN_SLOT u99
#endif

DECLARE_CBUFFER(LossConstants, gConst, 0, 0);

Buffer<float> lossBuffer                            :REGISTER_SRV(0,0);
RWByteAddressBuffer accumulationBuffer              :REGISTER_UAV(0,0);
RWStructuredBuffer<TrainingResults> outputBuffer    :REGISTER_UAV(1,0);

static const uint THREADS_PER_GROUP = RESULTS_THREADS_PER_GROUP;
static const uint MAX_COMPONENTS = 1; 

groupshared float gLossShared[THREADS_PER_GROUP][MAX_COMPONENTS];

[shader("compute")]
[numthreads(THREADS_PER_GROUP, 1, 1)]
void lossReduction_cs(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    const uint index = dispatchThreadID.x;
    
    const LossConfig lossConfig = MakeLossConfig(MAX_COMPONENTS, gConst.bufferOffset);

    float loss = 0.f;

    if (index < gConst.batchSize)
    {
        loss = lossBuffer[index];
    }
    gLossShared[groupThreadID.x][0] = loss;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1)
    {
        if (groupThreadID.x < stride)
        {
            AddInPlace<MAX_COMPONENTS>(gLossShared[groupThreadID.x], gLossShared[groupThreadID.x + stride], lossConfig.GetComponentCount());
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupThreadID.x == 0)
    {
       AccumulateComponents<MAX_COMPONENTS>(accumulationBuffer, gLossShared[0], lossConfig);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void average_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float lossSum =  asfloat(accumulationBuffer.Load(0u));
    float epochLoss = lossSum / gConst.epochSampleCount;
    
    TrainingResults result = {}; 
    result.l2Loss = epochLoss; 
    result.epoch = gConst.epoch;
    result.debug0 = lossSum;
    result.debug1 = gConst.epochSampleCount;
    
    outputBuffer[0] = result; 
}

