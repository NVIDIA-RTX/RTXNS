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

#include <nvrhi/nvrhi.h>

namespace rtxns
{

struct CoopVectorFeatures
{
    bool inferenceSupported = false;
    bool trainingSupported = false;
    bool fp16InferencingSupported = false;
    bool fp16TrainingSupported = false;
};

class GraphicsResources
{
public:
    GraphicsResources(nvrhi::DeviceHandle device);
    ~GraphicsResources();
    CoopVectorFeatures GetCoopVectorFeatures() const
    {
        return m_coopVectorFeatures;
    }

private:
    CoopVectorFeatures m_coopVectorFeatures;
};
} // namespace rtxns
