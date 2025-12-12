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

struct UIData
{
    float lightIntensity = 1.f;
    float specular = 0.5f;
    float roughness = 0.4f;
    float metallic = 0.7f;

    float trainingTime = 0.0f;
    uint32_t epochs = 0;

    bool reset = false;
    bool training = true;
    bool load = false;
    std::string fileName;
};