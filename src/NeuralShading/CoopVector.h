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
#endif

#include <vector>
#include <donut/app/DeviceManager.h>


#include "Float16.h"
#include "NeuralNetworkTypes.h"

namespace rtxns
{

class ICoopVectorUtils
{
public:
    size_t GetMatrixAlignment()
    {
        return s_matrixAlignment;
    }
    size_t GetVectorAlignment()
    {
        return s_vectorAlignment;
    }

    /**
     * Query the size of a matrix in bytes.
     * @return Size of matrix in bytes.
     */
    virtual size_t QueryMatrixByteSize(const uint32_t rows, const uint32_t cols, const MatrixLayout layout, const Precision precision = Precision::F16) = 0;

    /**
     * Convert matrix on the device between any layouts.
     * The Precision must currently be the same.
     * @return Size of matrix in bytes.
     */
    virtual void ConvertDeviceMatrixLayout(NetworkLayout const& srcLayout,
                                           NetworkLayout const& dstLayout,
                                           void* srcBuffer,
                                           uint64_t srcBufferOffset,
                                           void* dstBuffer,
                                           uint64_t dstBufferOffset,
                                           void* commandList) const = 0;

protected:
    static const size_t s_matrixAlignment = 64; ///< Minimum byte alignment according to spec.
    static const size_t s_vectorAlignment = 16; ///< Minimum byte alignment according to spec.
};

#if DONUT_WITH_VULKAN
class CoopVectorUtils_VK : public ICoopVectorUtils
{
public:
    CoopVectorUtils_VK(VkDevice vkDevice);

    /**
     * Query the size of a matrix in bytes.
     * @return Size of matrix in bytes.
     */
    size_t QueryMatrixByteSize(const uint32_t rows, const uint32_t cols, const MatrixLayout layout, const Precision precision = Precision::F16);

    /**
     * Convert matrix on the device between any layouts.
     * The Precision must currently be the same.
     * @return Size of matrix in bytes.
     */
    void ConvertDeviceMatrixLayout(
        NetworkLayout const& srcLayout, NetworkLayout const& dstLayout, void* srcBuffer, uint64_t srcBufferOffset, void* dstBuffer, uint64_t dstBufferOffset, void* commandList) const;

private:
    VkDevice m_vkDevice = nullptr;
    PFN_vkConvertCooperativeVectorMatrixNV m_vkConvertCooperativeVectorMatrixNV = nullptr;
    PFN_vkCmdConvertCooperativeVectorMatrixNV m_vkCmdConvertCooperativeVectorMatrixNV = nullptr;
    PFN_vkCmdCopyBuffer m_vkCmdCopyBuffer = nullptr;
    PFN_vkGetBufferDeviceAddress m_vkGetBufferDeviceAddress = nullptr;
};
#endif

#if DONUT_WITH_DX12
class CoopVectorUtils_DX12 : public ICoopVectorUtils
{
public:
    CoopVectorUtils_DX12(ID3D12Device* d3d12Device);

    /**
     * Query the size of a matrix in bytes.
     * @return Size of matrix in bytes.
     */
    size_t QueryMatrixByteSize(const uint32_t rows, const uint32_t cols, const MatrixLayout layout, const Precision precision = Precision::F16);

    /**
     * Convert matrix on the device between any layouts.
     * The Precision must currently be the same.
     * @return Size of matrix in bytes.
     */
    void ConvertDeviceMatrixLayout(
        NetworkLayout const& srcLayout, NetworkLayout const& dstLayout, void* srcBuffer, uint64_t srcBufferOffset, void* dstBuffer, uint64_t dstBufferOffset, void* commandList) const;

private:
    ID3D12Device* m_d3d12Device = nullptr;
};
#endif
} // namespace rtxns