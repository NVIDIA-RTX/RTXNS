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

#include "GraphicsResources.h"
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>

namespace rtxns
{

GraphicsResources::GraphicsResources(nvrhi::DeviceHandle device)
{
    m_coopVectorFeatures.inferenceSupported = device->queryFeatureSupport(nvrhi::Feature::CooperativeVectorInferencing);
    m_coopVectorFeatures.trainingSupported = device->queryFeatureSupport(nvrhi::Feature::CooperativeVectorTraining);

    auto features = device->queryCoopVecFeatures();
    for (const auto& combo : features.matMulFormats)
    {
        if (combo.inputType == nvrhi::coopvec::DataType::Float16 && combo.inputInterpretation == nvrhi::coopvec::DataType::Float16 &&
            combo.matrixInterpretation == nvrhi::coopvec::DataType::Float16 && combo.outputType == nvrhi::coopvec::DataType::Float16)
        {
            m_coopVectorFeatures.fp16InferencingSupported = true;
            m_coopVectorFeatures.fp16TrainingSupported = features.trainingFloat16;
            break;
        }
    }
}

GraphicsResources::~GraphicsResources()
{
}

} // namespace rtxns
