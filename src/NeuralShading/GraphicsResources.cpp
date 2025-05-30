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

#if DONUT_WITH_DX12
#include "../../external/dx12-agility-sdk/build/native/include/d3d12.h"
#include <dxgi1_4.h>
#include <wrl/client.h>
#endif

#if DONUT_WITH_VULKAN
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#endif

#include "GraphicsResources.h"
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>

namespace rtxns
{

GraphicsResources::GraphicsResources(nvrhi::DeviceHandle device)
{
#if DONUT_WITH_VULKAN
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
    {
        VkInstance vkInstance = device->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
        VkPhysicalDevice vkPhysicalDevice = device->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);

        m_vkGetPhysicalDeviceCooperativeVectorPropertiesNV = (PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV)VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(
            vkInstance, "vkGetPhysicalDeviceCooperativeVectorPropertiesNV");
        assert(m_vkGetPhysicalDeviceCooperativeVectorPropertiesNV != nullptr && "Failed to get Vulkan function 'vkGetPhysicalDeviceCooperativeVectorPropertiesNV'.");

        // Get the property count
        uint32_t propertyCount = 0;
        if (m_vkGetPhysicalDeviceCooperativeVectorPropertiesNV(vkPhysicalDevice, &propertyCount, nullptr) != VK_SUCCESS)
        {
            return;
        }

        // If we vkGetPhysicalDeviceCooperativeVectorPropertiesNV returns we have inference and training support
        m_coopVectorFeatures.inferenceSupported = true;
        m_coopVectorFeatures.trainingSupported = true;

        std::vector<VkCooperativeVectorPropertiesNV> properties(propertyCount);
        // Init the sType fields
        for (auto& property : properties)
        {
            property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV;
        }

        // Get the actual properties
        if (m_vkGetPhysicalDeviceCooperativeVectorPropertiesNV(vkPhysicalDevice, &propertyCount, properties.data()) != VK_SUCCESS)
        {
            return;
        }

        for (const auto& property : properties)
        {
            if (property.sType == VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV && property.inputType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                property.inputInterpretation == VK_COMPONENT_TYPE_FLOAT16_KHR && property.matrixInterpretation == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                property.resultType == VK_COMPONENT_TYPE_FLOAT16_KHR)
            {
                m_coopVectorFeatures.fp16InferencingSupported = true;
                m_coopVectorFeatures.fp16TrainingSupported = true;
            }
        }
    }
#endif

#if DONUT_WITH_DX12
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        ID3D12Device* d3d12Device = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);

        // Check experimental features are enabled
        D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL experimentalOptions{};
        auto hr = d3d12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL, &experimentalOptions, sizeof(experimentalOptions));
        if (hr != S_OK)
        {
            donut::log::error("Coop vector is not supported.");
            return;
        }

        // Mute preview shader model (6.9) validation warning.
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
        if (d3d12Device->QueryInterface(IID_PPV_ARGS(&infoQueue)) == S_OK)
        {
            D3D12_MESSAGE_ID denyIds[] = { D3D12_MESSAGE_ID_NON_RETAIL_SHADER_MODEL_WONT_VALIDATE };

            D3D12_INFO_QUEUE_FILTER filter = {};
            filter.DenyList.NumIDs = _countof(denyIds);
            filter.DenyList.pIDList = denyIds;

            infoQueue->AddStorageFilterEntries(&filter);
        }

        // Check coop vector is supported
        if (experimentalOptions.CooperativeVectorTier >= D3D12_COOPERATIVE_VECTOR_TIER_1_0)
        {
            m_coopVectorFeatures.inferenceSupported = true;
        }
        else
        {
            return;
        }
        if (experimentalOptions.CooperativeVectorTier >= D3D12_COOPERATIVE_VECTOR_TIER_1_1)
        {
            m_coopVectorFeatures.trainingSupported = true;
        }

        // Get supported coop vector formats
        D3D12_FEATURE_DATA_COOPERATIVE_VECTOR coopVecData{};
        hr = d3d12Device->CheckFeatureSupport(D3D12_FEATURE_COOPERATIVE_VECTOR, &coopVecData, sizeof(coopVecData));
        if (hr != S_OK)
        {
            return;
        }

        std::vector<D3D12_COOPERATIVE_VECTOR_PROPERTIES_MUL> mulProperties(coopVecData.MatrixVectorMulAddPropCount);
        std::vector<D3D12_COOPERATIVE_VECTOR_PROPERTIES_ACCUMULATE> outerProductProperties;
        std::vector<D3D12_COOPERATIVE_VECTOR_PROPERTIES_ACCUMULATE> vectorAccumlateProperties;

        coopVecData.pMatrixVectorMulAddProperties = mulProperties.data();

        if (experimentalOptions.CooperativeVectorTier >= D3D12_COOPERATIVE_VECTOR_TIER_1_1)
        {
            outerProductProperties.resize(coopVecData.OuterProductAccumulatePropCount);
            coopVecData.pOuterProductAccumulateProperties = outerProductProperties.data();
            vectorAccumlateProperties.resize(coopVecData.VectorAccumulatePropCount);
            coopVecData.pVectorAccumulateProperties = vectorAccumlateProperties.data();
        }
        else
        {
            coopVecData.OuterProductAccumulatePropCount = 0;
            coopVecData.VectorAccumulatePropCount = 0;
        }

        if (d3d12Device->CheckFeatureSupport(D3D12_FEATURE_COOPERATIVE_VECTOR, &coopVecData, sizeof(coopVecData)) != S_OK)
        {
            return;
        }

        for (const auto& properties : mulProperties)
        {
            if (properties.InputType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16 && properties.InputInterpretation == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16 &&
                properties.MatrixInterpretation == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16 && properties.OutputType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16)
            {
                m_coopVectorFeatures.fp16InferencingSupported = true;
            }
        }

        bool opSupported = false;
        for (const auto& properties : outerProductProperties)
        {
            if (properties.InputType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16 && properties.AccumulationType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16)
            {
                opSupported = true;
            }
        }

        bool vaSupported = false;
        for (const auto& properties : vectorAccumlateProperties)
        {
            if (properties.InputType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16 && properties.AccumulationType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16)
            {
                vaSupported = true;
            }
        }
        m_coopVectorFeatures.fp16TrainingSupported = opSupported && vaSupported;
    }
#endif
}

GraphicsResources::~GraphicsResources()
{
}

} // namespace rtxns
