/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CoopVector.h"
#include <algorithm>

#if DONUT_WITH_VULKAN
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#endif

#if DONUT_WITH_DX12
#include <dxgi1_4.h>
#include <wrl.h>
#endif

using namespace rtxns;

namespace
{
/**
 * Bytes between a consecutive row or column (if row/column-major layout).
 * The stride is only used for row/column major layouts
 **/
size_t GetStride(const MatrixLayout layout, const uint32_t rows, const uint32_t cols, const size_t precision)
{
    size_t stride = 0;
    if (layout == MatrixLayout::RowMajor)
    {
        stride = cols * precision;
    }
    else if (layout == MatrixLayout::ColumnMajor)
    {
        stride = rows * precision;
    }
    return stride;
}
} // namespace

#if DONUT_WITH_VULKAN
namespace
{

VkComponentTypeKHR GetVkComponentType(rtxns::Precision precision)
{
    return precision == rtxns::Precision::F16 ? VK_COMPONENT_TYPE_FLOAT16_NV : VK_COMPONENT_TYPE_FLOAT32_NV;
}

VkCooperativeVectorMatrixLayoutNV GetVkLayout(const MatrixLayout layout)
{
    switch (layout)
    {
    case MatrixLayout::RowMajor:
        return VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
    case MatrixLayout::ColumnMajor:
        return VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_NV;
    case MatrixLayout::InferencingOptimal:
        return VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV;
    case MatrixLayout::TrainingOptimal:
        return VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV;
    default:
        return VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_MAX_ENUM_NV;
    }
}

VkConvertCooperativeVectorMatrixInfoNV GetVkConvertLayerDesc(
    int rows, int columns, Precision precision, MatrixLayout srcLayout, MatrixLayout dstLayout, size_t srcSize, size_t* dstSize, uint64_t srcData = 0, uint64_t dstData = 0)
{
    VkConvertCooperativeVectorMatrixInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
    info.pNext = nullptr;
    info.numRows = rows;
    info.numColumns = columns;
    info.srcComponentType = GetVkComponentType(precision);
    info.srcLayout = GetVkLayout(srcLayout);
    info.srcStride = GetStride(MatrixLayout::RowMajor, rows, columns, GetSize(precision));
    info.srcSize = srcSize;
    info.srcData.deviceAddress = srcData;
    info.dstComponentType = GetVkComponentType(precision);
    info.dstLayout = GetVkLayout(dstLayout);
    info.dstStride = GetStride(dstLayout, rows, columns, GetSize(precision));
    info.pDstSize = dstSize;
    info.dstData.deviceAddress = dstData;
    return info;
}

} // namespace

CoopVectorUtils_VK::CoopVectorUtils_VK(VkDevice vkDevice)
{
    m_vkDevice = vkDevice;
    assert(m_vkDevice != VK_NULL_HANDLE && "Failed to get Vulkan device handle from GFX.");

    m_vkConvertCooperativeVectorMatrixNV =
        (PFN_vkConvertCooperativeVectorMatrixNV)VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(m_vkDevice, "vkConvertCooperativeVectorMatrixNV");
    assert(m_vkConvertCooperativeVectorMatrixNV != nullptr && "Failed to get Vulkan function 'vkConvertCooperativeVectorMatrixNV'.");

    m_vkCmdConvertCooperativeVectorMatrixNV =
        (PFN_vkCmdConvertCooperativeVectorMatrixNV)VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(m_vkDevice, "vkCmdConvertCooperativeVectorMatrixNV");
    assert(m_vkCmdConvertCooperativeVectorMatrixNV != nullptr && "Failed to get Vulkan function 'vkCmdConvertCooperativeVectorMatrixNV'.");

    m_vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(m_vkDevice, "vkCmdCopyBuffer");
    assert(m_vkCmdCopyBuffer != nullptr && "Failed to get Vulkan function 'vkCmdCopyBuffer'.");

    m_vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(m_vkDevice, "vkGetBufferDeviceAddress");
    assert(m_vkGetBufferDeviceAddress != nullptr && "Failed to get Vulkan function 'vkGetBufferDeviceAddress'.");
}

size_t CoopVectorUtils_VK::QueryMatrixByteSize(const uint32_t rows, const uint32_t cols, const MatrixLayout layout, const Precision precision)
{
    assert(m_vkDevice);
    assert(m_vkConvertCooperativeVectorMatrixNV);
    assert(rows > 0 && rows <= 128 && "Number of rows must be 1..128.");
    assert(cols > 0 && cols <= 128 && "Number of columns must be 1..128.");

    size_t requiredSize = 0;

    VkConvertCooperativeVectorMatrixInfoNV info = GetVkConvertLayerDesc(rows, cols, precision, MatrixLayout::RowMajor, layout, 0, &requiredSize);

    VkResult res = m_vkConvertCooperativeVectorMatrixNV(m_vkDevice, &info);
    assert(res == VK_SUCCESS && "Call to vkConvertCooperativeVectorMatrixNV failed");
    assert(requiredSize > 0 && "Expected matrix size to be larger than zero.");

    return requiredSize;
}

void CoopVectorUtils_VK::ConvertDeviceMatrixLayout(
    NetworkLayout const& srcLayout, NetworkLayout const& dstLayout, void* srcBuffer, uint64_t srcBufferOffset, void* dstBuffer, uint64_t dstBufferOffset, void* commandList) const
{
    VkCommandBuffer vkCmdBuf = static_cast<VkCommandBuffer>(commandList);
    VkBuffer vkSrcBuffer = static_cast<VkBuffer>(srcBuffer);
    VkBuffer vkDstBuffer = static_cast<VkBuffer>(dstBuffer);

    // Obtain the device addresses of the buffers for the conversion functions
    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
    bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddressInfo.buffer = vkSrcBuffer;
    VkDeviceAddress const srcBufferVA = m_vkGetBufferDeviceAddress(m_vkDevice, &bufferDeviceAddressInfo);
    bufferDeviceAddressInfo.buffer = vkDstBuffer;
    VkDeviceAddress const dstBufferVA = m_vkGetBufferDeviceAddress(m_vkDevice, &bufferDeviceAddressInfo);

    // Convert weights
    std::vector<VkConvertCooperativeVectorMatrixInfoNV> convertInfos(srcLayout.networkLayers.size());
    for (int i = 0; i < srcLayout.networkLayers.size(); i++)
    {
        // Weights
        size_t dstLayerSize = dstLayout.networkLayers[i].weightSize;
        convertInfos[i] =
            GetVkConvertLayerDesc(srcLayout.networkLayers[i].outputs, srcLayout.networkLayers[i].inputs, srcLayout.matrixPrecision, srcLayout.matrixLayout, dstLayout.matrixLayout,
                                  srcLayout.networkLayers[i].weightSize, &dstLayerSize, srcBufferVA + srcBufferOffset + srcLayout.networkLayers[i].weightOffset,
                                  dstBufferVA + dstBufferOffset + dstLayout.networkLayers[i].weightOffset);
    }
    m_vkCmdConvertCooperativeVectorMatrixNV(vkCmdBuf, (uint32_t)convertInfos.size(), convertInfos.data());

    // Copy the bias
    std::vector<VkBufferCopy> copyRegions(srcLayout.networkLayers.size());
    for (int i = 0; i < srcLayout.networkLayers.size(); i++)
    {
        copyRegions[i].srcOffset = srcBufferOffset + srcLayout.networkLayers[i].biasOffset;
        copyRegions[i].dstOffset = dstBufferOffset + dstLayout.networkLayers[i].biasOffset;
        copyRegions[i].size = srcLayout.networkLayers[i].biasSize;
    }
    m_vkCmdCopyBuffer(vkCmdBuf, vkSrcBuffer, vkDstBuffer, (uint32_t)copyRegions.size(), copyRegions.data());
}
#endif

#if DONUT_WITH_DX12

namespace
{
D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT GetDX12MatrixLayout(const MatrixLayout layout)
{
    switch (layout)
    {
    case MatrixLayout::RowMajor:
        return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
    case MatrixLayout::ColumnMajor:
        return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_COLUMN_MAJOR;
    case MatrixLayout::InferencingOptimal:
        return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL;
    case MatrixLayout::TrainingOptimal:
        return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL;
    }
}

D3D12_LINEAR_ALGEBRA_DATATYPE GetDX12ComponentType(rtxns::Precision precision)
{
    return precision == rtxns::Precision::F16 ? D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16 : D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;
}

D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO GetDX12ConvertLayerDestInfo(int rows, int columns, MatrixLayout layout, Precision precision)
{
    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO info{};
    info.DestLayout = GetDX12MatrixLayout(layout);
    info.NumRows = rows;
    info.NumColumns = columns;
    info.DestStride = UINT(GetStride(layout, rows, columns, GetSize(precision)));
    info.DestDataType = GetDX12ComponentType(precision);
    return info;
}

D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO GetDX12ConvertLayerDesc(
    int rows, int columns, Precision precision, MatrixLayout srcLayout, MatrixLayout dstLayout, size_t srcSize, size_t dstSize, uint64_t srcData, uint64_t dstData)
{
    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO info{};
    info.DestInfo = GetDX12ConvertLayerDestInfo(rows, columns, dstLayout, precision);
    info.DestInfo.DestSize = UINT(dstSize);
    info.SrcInfo.SrcSize = UINT(srcSize);
    info.SrcInfo.SrcDataType = GetDX12ComponentType(precision);
    info.SrcInfo.SrcLayout = GetDX12MatrixLayout(srcLayout);
    info.SrcInfo.SrcStride = UINT(GetStride(MatrixLayout::RowMajor, rows, columns, GetSize(precision)));
    info.DataDesc.SrcVA = srcData;
    info.DataDesc.DestVA = dstData;
    return info;
}

D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO GetDX12CopyScaleBiasDesc(size_t biasSize, Precision precision, uint64_t srcData, uint64_t dstData)
{
    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO info{};
    info.DestInfo.DestSize = UINT(biasSize);
    info.DestInfo.DestLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
    info.DestInfo.DestStride = info.DestInfo.DestSize;
    info.DestInfo.NumRows = 1;
    info.DestInfo.NumColumns = UINT(biasSize / GetSize(precision));
    info.DestInfo.DestDataType = GetDX12ComponentType(precision);
    info.SrcInfo.SrcSize = info.DestInfo.DestSize;
    info.SrcInfo.SrcDataType = info.DestInfo.DestDataType;
    info.SrcInfo.SrcLayout = info.DestInfo.DestLayout;
    info.SrcInfo.SrcStride = info.DestInfo.DestStride;
    info.DataDesc.SrcVA = srcData;
    info.DataDesc.DestVA = dstData;
    return info;
}
} // namespace

CoopVectorUtils_DX12::CoopVectorUtils_DX12(ID3D12Device* d3d12Device)
{
    m_d3d12Device = d3d12Device;
    assert(m_d3d12Device != nullptr && "Failed to get D3D12 device from GFX.");
}

/**
 * Query the size of a matrix in bytes.
 * @return Size of matrix in bytes.
 */
size_t CoopVectorUtils_DX12::QueryMatrixByteSize(const uint32_t rows, const uint32_t cols, const MatrixLayout layout, const Precision precision /*= Precision::F16*/)
{
    assert(m_d3d12Device);
    assert(rows > 0 && rows <= 128 && "Number of rows must be 1..128.");
    assert(cols > 0 && cols <= 128 && "Number of columns must be 1..128.");

    Microsoft::WRL::ComPtr<ID3D12DevicePreview> devicePreview;
    assert(m_d3d12Device->QueryInterface(IID_PPV_ARGS(&devicePreview)) == S_OK && "Failed to get device preview");

    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO info = GetDX12ConvertLayerDestInfo(rows, cols, layout, precision);

    devicePreview->GetLinearAlgebraMatrixConversionDestinationInfo(&info);

    assert(info.DestSize > 0 && "Expected matrix size to be larger than zero.");
    return info.DestSize;
}

void rtxns::CoopVectorUtils_DX12::ConvertDeviceMatrixLayout(
    NetworkLayout const& srcLayout, NetworkLayout const& dstLayout, void* srcBuffer, uint64_t srcBufferOffset, void* dstBuffer, uint64_t dstBufferOffset, void* commandList) const
{
    ID3D12GraphicsCommandList* d3dCmdList = static_cast<ID3D12GraphicsCommandList*>(commandList);
    ID3D12Resource* d3dSrcBuffer = static_cast<ID3D12Resource*>(srcBuffer);
    ID3D12Resource* d3dDstBuffer = static_cast<ID3D12Resource*>(dstBuffer);

    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandListPreview> commandListPreview;
    assert(d3dCmdList->QueryInterface(IID_PPV_ARGS(&commandListPreview)) == S_OK && "Command list provided does not support matrix conversion");

    D3D12_GPU_VIRTUAL_ADDRESS const srcBufferVA = d3dSrcBuffer->GetGPUVirtualAddress();
    D3D12_GPU_VIRTUAL_ADDRESS const dstBufferVA = d3dDstBuffer->GetGPUVirtualAddress();

    // We need conversion data for each of the weights and bias separately so we need two entry for each layer
    std::vector<D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO> convertInfos(srcLayout.networkLayers.size() * 2);

    // Convert weights
    for (int i = 0; i < srcLayout.networkLayers.size(); i++)
    {
        // Weights
        convertInfos[i] = GetDX12ConvertLayerDesc(srcLayout.networkLayers[i].outputs, srcLayout.networkLayers[i].inputs, srcLayout.matrixPrecision, srcLayout.matrixLayout,
                                                  dstLayout.matrixLayout, srcLayout.networkLayers[i].weightSize, dstLayout.networkLayers[i].weightSize,
                                                  srcBufferVA + srcBufferOffset + srcLayout.networkLayers[i].weightOffset,
                                                  dstBufferVA + dstBufferOffset + dstLayout.networkLayers[i].weightOffset);
    }

    // Convert bias
    // D3D's CopyBufferRegion requires resource states incompatible with the conversion ops.
    // Use a degenerate form of a matrix conversion to copy the extra data to avoid placing a barrier.
    int infoOffset = int(srcLayout.networkLayers.size());
    for (int ii = 0; ii < srcLayout.networkLayers.size(); ii++)
    {
        convertInfos[ii + infoOffset] =
            GetDX12CopyScaleBiasDesc(srcLayout.networkLayers[ii].biasSize, srcLayout.matrixPrecision, srcBufferVA + srcBufferOffset + srcLayout.networkLayers[ii].biasOffset,
                                     dstBufferVA + dstBufferOffset + dstLayout.networkLayers[ii].biasOffset);
    }
    commandListPreview->ConvertLinearAlgebraMatrix(convertInfos.data(), UINT(convertInfos.size()));
}
#endif
