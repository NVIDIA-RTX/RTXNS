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

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

using namespace rtxns;

namespace
{


VkComponentTypeKHR GetComponentType(rtxns::Precision precision)
{
    return precision == rtxns::Precision::F16 ? VK_COMPONENT_TYPE_FLOAT16_NV : VK_COMPONENT_TYPE_FLOAT32_NV;
}

} // namespace

VkCooperativeVectorMatrixLayoutNV CoopVectorUtils_VK::GetLayout(const MatrixLayout layout)
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

CoopVectorUtils_VK::CoopVectorUtils_VK(VkDevice vkDevice)
{
    m_vkDevice = vkDevice;
    assert(m_vkDevice != VK_NULL_HANDLE && "Failed to get Vulkan device handle from GFX.");

    m_vkConvertCooperativeVectorMatrixNV =
        (PFN_vkConvertCooperativeVectorMatrixNV)VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(m_vkDevice, "vkConvertCooperativeVectorMatrixNV");
    assert(m_vkConvertCooperativeVectorMatrixNV != nullptr && "Failed to get Vulkan function 'vkConvertCooperativeVectorMatrixNV'.");
}

size_t CoopVectorUtils_VK::QueryMatrixByteSize(const uint32_t rows, const uint32_t cols, const MatrixLayout layout, const Precision precision)
{
    assert(m_vkDevice);
    assert(m_vkConvertCooperativeVectorMatrixNV);
    assert(rows > 0 && rows <= 128 && "Number of rows must be 1..128.");
    assert(cols > 0 && cols <= 128 && "Number of columns must be 1..128.");

    size_t requiredSize = 0;

    // Bytes between a consecutive row or column (if row/column-major layout).
    // The stride is only used for row/column major layouts
    size_t stride = (layout == MatrixLayout::RowMajor) ? cols * GetSize(precision) : rows * GetSize(precision);

    VkConvertCooperativeVectorMatrixInfoNV info = {};
    info.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
    info.numRows = rows;
    info.numColumns = cols;
    info.srcComponentType = GetComponentType(precision);
    info.srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
    info.srcStride = stride;
    info.srcSize = 0;
    info.srcData.hostAddress = nullptr;
    info.dstComponentType = GetComponentType(precision);
    info.dstLayout = GetLayout(layout);
    info.dstStride = stride;
    info.pDstSize = &requiredSize;
    info.dstData.hostAddress = nullptr;

    VkResult res = m_vkConvertCooperativeVectorMatrixNV(m_vkDevice, &info);
    assert(res == VK_SUCCESS && "Call to vkConvertCooperativeVectorMatrixNV failed");
    assert(requiredSize > 0 && "Expected matrix size to be larger than zero.");

    return requiredSize;
}

size_t CoopVectorUtils_VK::ConvertHostf32Matrix(
    const uint32_t rows, const uint32_t cols, const float* src, const size_t srcSize, uint8_t* dst, const size_t dstSize, const Precision dstPrecision, const MatrixLayout dstLayout) const
{
    assert(m_vkDevice);
    assert(m_vkConvertCooperativeVectorMatrixNV);
    assert(rows > 0 && rows <= 128 && "Number of rows must be 1..128.");
    assert(cols > 0 && cols <= 128 && "Number of columns must be 1..128.");

    size_t hostSize = rows * cols * GetSize(dstPrecision);
    void* hostAddress = (void*)src;

    assert(srcSize == rows * cols * sizeof(float) && "Unexpected source size.");
    std::vector<uint16_t> srcFP16(rows * cols);
    if (dstPrecision == Precision::F16)
    {
        // Convert matrix to float16.
        std::transform(src, src + rows * cols, srcFP16.data(), [](float w) { return rtxns::float32ToFloat16(w); });
        hostAddress = srcFP16.data();
    }
    size_t actualSize = dstSize;

    VkConvertCooperativeVectorMatrixInfoNV info = {};
    info.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
    info.numRows = rows;
    info.numColumns = cols;
    info.srcComponentType = GetComponentType(dstPrecision);
    info.srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
    info.srcStride = cols * GetSize(dstPrecision); // Bytes between a consecutive row or column (if row/column-major layout).
    info.srcSize = hostSize;
    info.srcData.hostAddress = hostAddress;
    info.dstComponentType = GetComponentType(dstPrecision);
    info.dstLayout = GetLayout(dstLayout);
    info.dstStride = 0; // Bytes between a consecutive row or column (if row/column-major layout).
    info.pDstSize = &actualSize;
    info.dstData.hostAddress = dst;

    VkResult res = m_vkConvertCooperativeVectorMatrixNV(m_vkDevice, &info);
    assert(res == VK_SUCCESS && "Call to vkConvertCooperativeVectorMatrixNV failed");
    assert(actualSize > 0 && "Expected matrix size to be larger than zero.");

    return actualSize;
}

size_t CoopVectorUtils_VK::ConvertHostMatrixLayout(const uint32_t rows,
                                                   const uint32_t cols,
                                                   const void* src,
                                                   const size_t srcSize,
                                                   const Precision srcPrecision,
                                                   const MatrixLayout srcLayout,
                                                   void* dst,
                                                   const size_t dstSize,
                                                   const Precision dstPrecision,
                                                   const MatrixLayout dstLayout) const
{
    assert(m_vkDevice);
    assert(m_vkConvertCooperativeVectorMatrixNV);
    assert(rows > 0 && rows <= 128 && "Number of rows must be 1..128.");
    assert(cols > 0 && cols <= 128 && "Number of columns must be 1..128.");
    assert(dstPrecision == srcPrecision);

    size_t actualSize = dstSize;

    // Bytes between a consecutive row or column (if row/column-major layout).
    // The stride is only used for row/column major layouts
    size_t srcStride = (srcLayout == MatrixLayout::RowMajor) ? cols * GetSize(srcPrecision) : rows * GetSize(srcPrecision);
    size_t dstStride = (dstLayout == MatrixLayout::RowMajor) ? cols * GetSize(dstPrecision) : rows * GetSize(dstPrecision);

    VkConvertCooperativeVectorMatrixInfoNV info = {};
    info.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
    info.numRows = rows;
    info.numColumns = cols;
    info.srcComponentType = GetComponentType(srcPrecision);
    info.srcLayout = GetLayout(srcLayout);
    info.srcStride = srcStride;
    info.srcSize = srcSize;
    info.srcData.hostAddress = src;
    info.dstComponentType = GetComponentType(dstPrecision);
    info.dstLayout = GetLayout(dstLayout);
    info.dstStride = dstStride;
    info.pDstSize = &actualSize;
    info.dstData.hostAddress = dst;

    VkResult res = m_vkConvertCooperativeVectorMatrixNV(m_vkDevice, &info);
    assert(res == VK_SUCCESS && "Call to vkConvertCooperativeVectorMatrixNV failed");
    assert(actualSize > 0 && "Expected matrix size to be larger than zero.");

    return actualSize;
}
