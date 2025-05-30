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
#include <cstdint>

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

struct NetworkArchitecture
{
    uint32_t numHiddenLayers = 0;
    uint32_t inputNeurons = 0;
    uint32_t hiddenNeurons = 0;
    uint32_t outputNeurons = 0;
    Precision weightPrecision = Precision::F16;
    Precision biasPrecision = Precision::F16;
};

struct NetworkLayer
{
    uint32_t inputs = 0; ///< Columns in the weight matrix.
    uint32_t outputs = 0; ///< Rows in the weight matrix.
    size_t weightSize = 0; ///< Size of the weight matrix in bytes.
    size_t biasSize = 0; ///< Size of the bias vector in bytes.
    uint32_t weightOffset = 0; ///< Offset to the weights in bytes.
    uint32_t biasOffset = 0; ///< Offset to the biases in bytes.
};

struct NetworkLayout
{
    MatrixLayout matrixLayout = MatrixLayout::RowMajor;
    Precision matrixPrecision = Precision::F16;
    size_t networkSize = 0;
    std::vector<NetworkLayer> networkLayers;
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

} // namespace rtxns