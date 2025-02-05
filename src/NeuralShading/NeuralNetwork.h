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
#include <filesystem>
#include <nvrhi/utils.h>
#include "CoopVector.h"

namespace rtxns
{
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

class Network
{
public:
    Network(nvrhi::DeviceHandle device);
    ~Network(){};

    bool Initialise(const NetworkArchitecture& netArch, MatrixLayout layout);
    bool InitialiseFromJson(donut::vfs::IFileSystem& fs, const std::string& fileName);
    bool InitialiseFromFile(const std::string& fileName);
    bool InitialiseFromNetwork(const Network& network, MatrixLayout layout);

    bool ChangeLayout(MatrixLayout layout);

    bool WriteToFile(const std::string& fileName);

    void UpdateFromBufferToFile(nvrhi::BufferHandle gpuBuffer, const std::string& fileName);

    const NetworkArchitecture& GetNetworkArchitecture() const
    {
        return m_networkArchitecture;
    }

    const std::vector<NetworkLayer>& GetNetworkLayers() const
    {
        return m_networkLayers;
    }

    const std::vector<uint8_t>& GetNetworkParams() const
    {
        return m_params;
    }

    const MatrixLayout& GetMatrixLayout() const
    {
        return m_layout;
    }

private:
    bool ValidateNetworkArchitecture(const NetworkArchitecture& netArch);

private:
    nvrhi::DeviceHandle m_device;
    NetworkArchitecture m_networkArchitecture;
    std::vector<NetworkLayer> m_networkLayers;
    std::vector<uint8_t> m_params;
    MatrixLayout m_layout;
};
}; // namespace rtxns