/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "LearningRateScheduler.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

LearningRateScheduler::LearningRateScheduler()
{
}

LearningRateScheduler::LearningRateScheduler(float baseRate, float minRate, int warmupSteps, int flatSteps, int decaySteps)
    : m_baseRate(baseRate), m_minRate(minRate), m_warmupSteps(warmupSteps), m_flatSteps(flatSteps), m_decaySteps(decaySteps)
{
}

float LearningRateScheduler::GetLearningRate(int step) const
{
    if (step <= 0)
    {
        // Guard against zero or negative steps, should not get here.
        return m_warmupSteps > 0 ? 0.f : m_baseRate;
    }
    else if (step < m_warmupSteps)
    {
        // Linear warm-up from 0 to the base rate
        float progress = static_cast<float>(step) / m_warmupSteps;
        return m_baseRate * progress;
    }
    else if (step < m_warmupSteps + m_flatSteps)
    {
        return m_baseRate;
    }
    else if (step < m_warmupSteps + m_flatSteps + m_decaySteps)
    {
        // Cosine decay from base rate to minimum learning rate
        int decayStep = step - (m_warmupSteps + m_flatSteps);
        float decayRatio = std::min(1.0f, static_cast<float>(decayStep) / m_decaySteps);

        float cosineDecay = 0.5f * (1.f + float(std::cos(decayRatio * M_PI)));
        return m_minRate + (m_baseRate - m_minRate) * cosineDecay;
    }
    else
    {
        return m_minRate;
    }
}