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

#include <donut/app/DeviceManager.h>
#include <donut/core/math/math.h>

using namespace donut::math;

#include "TrainingResults.h"

class ResultsReadbackHandler
{
public:
    ResultsReadbackHandler(nvrhi::DeviceHandle device);
    void SyncResults(nvrhi::CommandListHandle commandList);
    nvrhi::BufferHandle GetResultsBuffers() const;
    bool GetResults(TrainingResults& results) const;
    void Reset();

private:
    static constexpr int c_bufferCount = 2;
    nvrhi::DeviceHandle m_device;
    nvrhi::BufferHandle m_resultsBuffer;
    nvrhi::BufferHandle m_readbackBuffers[c_bufferCount];
    TrainingResults m_results;
    bool m_hasResults = false;
    bool m_hasResultsThisRun = false;
    int m_currentIndex = 0;
};