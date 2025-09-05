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

class LearningRateScheduler
{
public:
    LearningRateScheduler();
    LearningRateScheduler(float baseRate, float minRate, int warmupSteps, int flatSteps, int decaySteps);
    float GetLearningRate(int step) const;

private:
    float m_baseRate = 1e-3f; // Initial peak learning rate
    float m_minRate = 1e-4f; // Floor learning rate at end of schedule
    int m_warmupSteps = 10000;
    int m_flatSteps = 100000;
    int m_decaySteps = 100000;
};