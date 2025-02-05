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

#include <donut/core/math/math.h>

#include <vector>

struct Vertex
{
    dm::float3 position;
    dm::float3 normal;
};

std::pair<std::vector<Vertex>, std::vector<uint32_t>> GenerateSphere(float radius, uint32_t segmentsU, uint32_t segmentsV);
