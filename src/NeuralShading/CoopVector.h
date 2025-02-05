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

#include <vector>

#include <donut/app/DeviceManager.h>

#include "Float16.h"

namespace rtxns
{
enum class MatrixLayout
{
    RowMajor,
    ColumnMajor,
    InferencingOptimal,
    TrainingOptimal,
};

enum class Precision
{
    F16,
    F32
};

constexpr size_t GetSize(Precision precision)
{
    switch (precision)
    {
    case Precision::F16:
        return sizeof(uint16_t); // 2 bytes
    case Precision::F32:
        return sizeof(float);
    default:
        return 0; // Should not get here
    }
}

class ICoopVectorUtils
{
public:
    /**
     * Query the size of a matrix in bytes.
     * @return Size of matrix in bytes.
     */
    virtual size_t QueryMatrixByteSize(const uint32_t rows, const uint32_t cols, const MatrixLayout layout, const Precision precision = Precision::F16) = 0;

    /**
     * Convert matrix on the host from row-major layout in float32 to GPU-specific layout in dstPrecision.
     * @return Size of matrix in bytes.
     */
    virtual size_t ConvertHostf32Matrix(const uint32_t rows,
                                        const uint32_t cols,
                                        const float* src,
                                        const size_t srcSize,
                                        uint8_t* dst,
                                        const size_t dstSize,
                                        const Precision dstPrecision,
                                        const MatrixLayout dstLayout) const = 0;

    /**
     * Convert matrix on the host between any layouts.
     * The Precision must currently be the same.
     * @return Size of matrix in bytes.
     */
    virtual size_t ConvertHostMatrixLayout(const uint32_t rows,
                                           const uint32_t cols,
                                           const void* src,
                                           const size_t srcSize,
                                           const Precision srcPrecision,
                                           const MatrixLayout srcLayout,
                                           void* dst,
                                           const size_t dstSize,
                                           const Precision dstPrecision,
                                           const MatrixLayout dstLayout) const = 0;

    virtual size_t GetMatrixAlignment() = 0;
    virtual size_t GetVectorAlignment() = 0;
};


class CoopVectorUtils_VK : public ICoopVectorUtils
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

    static VkCooperativeVectorMatrixLayoutNV GetLayout(const MatrixLayout layout);

    CoopVectorUtils_VK(VkDevice vkDevice);

    /**
     * Query the size of a matrix in bytes.
     * @return Size of matrix in bytes.
     */
    size_t QueryMatrixByteSize(const uint32_t rows, const uint32_t cols, const MatrixLayout layout, const Precision precision = Precision::F16);

    /**
     * Convert matrix on the host from row-major layout in float32 to GPU-specific layout in dstPrecision.
     * @return Size of matrix in bytes.
     */
    size_t ConvertHostf32Matrix(const uint32_t rows,
                                const uint32_t cols,
                                const float* src,
                                const size_t srcSize,
                                uint8_t* dst,
                                const size_t dstSize,
                                const Precision dstPrecision,
                                const MatrixLayout dstLayout) const;

    /**
     * Convert matrix on the host between any layouts.
     * The Precision must currently be the same.
     * @return Size of matrix in bytes.
     */
    size_t ConvertHostMatrixLayout(const uint32_t rows,
                                   const uint32_t cols,
                                   const void* src,
                                   const size_t srcSize,
                                   const Precision srcPrecision,
                                   const MatrixLayout srcLayout,
                                   void* dst,
                                   const size_t dstSize,
                                   const Precision dstPrecision,
                                   const MatrixLayout dstLayout) const;

protected:
    static const size_t s_matrixAlignment = 64; ///< Minimum byte alignment according to spec.
    static const size_t s_vectorAlignment = 16; ///< Minimum byte alignment according to spec.

private:
    VkDevice m_vkDevice = nullptr;
    PFN_vkConvertCooperativeVectorMatrixNV m_vkConvertCooperativeVectorMatrixNV = nullptr;
};
} // namespace rtxns