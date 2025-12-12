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

import Optimizers;

DECLARE_CBUFFER(TrainingConstantBufferEntry, gConst, 0, 0);
RWBuffer<half> gMLPParams             : REGISTER_UAV(0, 0);
RWBuffer<float> gMLPParams32          : REGISTER_UAV(1, 0);
RWBuffer<half> gMLPParamsGradients    : REGISTER_UAV(2, 0);
RWBuffer<float> gMoments1             : REGISTER_UAV(3, 0);
RWBuffer<float> gMoments2             : REGISTER_UAV(4, 0);

[numthreads(THREADS_PER_GROUP_OPTIMIZE, 1, 1)]
void adam_cs(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= gConst.maxParamSize)
        return;

    float gradient = (float)gMLPParamsGradients[i];
    gMLPParamsGradients[i] = half(0.0);

    if (isfinite(gradient))
    {
        // On the first step load from the FP16 buffer
        float weightBias = gConst.currentStep == 1 ? (float)gMLPParams[i] : gMLPParams32[i];

        optimizers::Adam optimizer = optimizers::Adam(gMoments1, gMoments2, gConst.learningRate, LOSS_SCALE);

        float adjustedWeightBias = optimizer.step(weightBias, i, gradient, gConst.currentStep);

        gMLPParams32[i] = adjustedWeightBias;
        gMLPParams[i] = (half) adjustedWeightBias;
    }
}