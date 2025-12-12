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

#if DONUT_WITH_DX12
#include <wrl/client.h>
#include <nvapi.h>
#endif

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
#if DONUT_WITH_DX12
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        // Mute preview shader model (6.9) validation warning.
        ID3D12Device* d3d12Device = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
        if (d3d12Device->QueryInterface(IID_PPV_ARGS(&infoQueue)) == S_OK)
        {
            D3D12_MESSAGE_ID denyIds[] = { D3D12_MESSAGE_ID_NON_RETAIL_SHADER_MODEL_WONT_VALIDATE };

            D3D12_INFO_QUEUE_FILTER filter = {};
            filter.DenyList.NumIDs = _countof(denyIds);
            filter.DenyList.pIDList = denyIds;

            infoQueue->AddStorageFilterEntries(&filter);
        }

        // Set NVAPI shader extension slot and space
        if (m_nvapiInitialised = NvAPI_Initialize() == NVAPI_OK)
        {
            if (NvAPI_D3D12_SetNvShaderExtnSlotSpace(d3d12Device, NV_SHADER_EXTN_SLOT, NV_SHADER_EXTN_REGISTER_SPACE) != NVAPI_OK)
            {
                return;
            }
        }
    }
#endif
}

GraphicsResources::~GraphicsResources()
{
}

} // namespace rtxns
