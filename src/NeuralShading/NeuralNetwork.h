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

#include "NeuralNetworkTypes.h"

namespace rtxns
{

class NetworkUtilities
{
public:
    NetworkUtilities(nvrhi::DeviceHandle device);
    ~NetworkUtilities()
    {
    }

    bool ValidateNetworkArchitecture(NetworkArchitecture const& netArch);

    bool CompareNetworkArchitecture(const NetworkArchitecture& a, const NetworkArchitecture& b);

    // Create host side network layout.
    NetworkLayout CreateHostNetworkLayout(NetworkArchitecture const& netArch);

    // Set the weights and bias size / offsets for each layer in the network.
    void SetNetworkLayerSizes(NetworkLayout& layout);

    // Returns a updated network layout where the weights and bias size / offsets have been update
    // for the new matrix layout
    // Can be device optimal matrix layout
    NetworkLayout GetNewMatrixLayout(NetworkLayout const& srcLayout, MatrixLayout newMatrixLayout, Precision newPrecision = Precision::F16);

    // Converts weights and bias buffers from src layout to the dst layout.
    // Both buffers must be device side.
    // Both networks must be of the same network layout, only differing in MatrixLayout
    void ConvertWeights(NetworkLayout const& srcLayout,
                        NetworkLayout const& dstLayout,
                        nvrhi::BufferHandle srcBuffer,
                        uint64_t srcBufferOffset,
                        nvrhi::BufferHandle dstBuffer,
                        uint64_t dstBufferOffset,
                        nvrhi::DeviceHandle device,
                        nvrhi::CommandListHandle commandList);

private:
    nvrhi::DeviceHandle m_device;
};

// Represent a host side neural network.
// Stores the network layout and parameters.
// Functionality to initialize a network to starting values or load from file.
// Also write parameters back to file
class HostNetwork
{
public:
    HostNetwork(std::shared_ptr<NetworkUtilities> networkUtils);
    ~HostNetwork(){};

    // Create host side network from provided architecture with initial values.
    bool Initialise(const NetworkArchitecture& netArch);
    // Create host side network of provided architecture and initial values from a json file.
    bool InitialiseFromJson(donut::vfs::IFileSystem& fs, const std::string& fileName);
    // Create host side network of provided architecture and initial values from a file.
    bool InitialiseFromFile(const std::string& fileName);
    // Create host side network from an existing network.
    bool InitialiseFromNetwork(HostNetwork const& network);
    // Reset the weight and bias parameters to starting values
    void ResetParameters();
    // Write the current network and parameters to file.
    bool WriteToFile(const std::string& fileName);
    // Convert device layout to host layout and update the host side parameters.
    void UpdateFromBufferToFile(nvrhi::BufferHandle hostLayoutBuffer,
                                nvrhi::BufferHandle deviceLayoutBuffer,
                                NetworkLayout const& hostLayout,
                                NetworkLayout const& deviceLayout,
                                const std::string& fileName,
                                nvrhi::DeviceHandle device,
                                nvrhi::CommandListHandle commandList);

    const NetworkArchitecture& GetNetworkArchitecture() const
    {
        return m_networkArchitecture;
    }

    const std::vector<uint8_t>& GetNetworkParams() const
    {
        return m_networkParams;
    }

    const NetworkLayout& GetNetworkLayout() const
    {
        return m_networkLayout;
    }

private:
    std::shared_ptr<NetworkUtilities> m_networkUtils;
    NetworkArchitecture m_networkArchitecture;
    std::vector<uint8_t> m_networkParams;
    NetworkLayout m_networkLayout;
};
}; // namespace rtxns