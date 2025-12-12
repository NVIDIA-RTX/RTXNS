/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#define RESULTS_THREADS_PER_GROUP 64

struct TrainingResults
{
    float4 averageLoss = float4(0.f, 0.f, 0.f, 0.f);

    float l2Loss = 0.f;
    uint32_t epoch = 0;
    float debug0 = 0;
    float debug1 = 0;
};

struct LossConstants
{
    uint32_t batchSize = 0;
    uint32_t epochSampleCount = 0;
    uint32_t bufferOffset = 0;
    uint32_t epoch = 0;
};