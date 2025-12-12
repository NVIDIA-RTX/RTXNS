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

#include "NetworkTransforms.h"

struct UIData
{
    bool reset = false;
    bool training = true;
    bool load = false;
    std::string fileName;
    float trainingTime = 0.0f;
    uint32_t epochs = 0;
    uint32_t adamSteps = 0;
    float learningRate = 0.0f;
    NetworkTransform networkTransform = NetworkTransform::Identity;
};