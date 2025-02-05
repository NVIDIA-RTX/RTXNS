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

#define NTC_WITH_VULKAN 1

namespace donut::app
{
struct DeviceCreationParameters;
class DeviceManager;
} // namespace donut::app

void SetCoopVectorExtensionParameters(donut::app::DeviceCreationParameters& deviceParams, nvrhi::GraphicsAPI graphicsApi, bool enableSharedMemory);

bool IsDP4aSupported(nvrhi::IDevice* device);

bool IsFloat16Supported(nvrhi::IDevice* device);

// Call after device creation to verify the extension has been enabled
bool CoopVectorExtensionSupported(donut::app::DeviceManager* deviceManager);