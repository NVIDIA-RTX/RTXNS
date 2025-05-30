/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <fstream>
#include <sstream>
#include <random>
#include <donut/core/json.h>
#include "NeuralNetwork.h"

using namespace donut;
using namespace donut::math;
using namespace rtxns;

namespace
{
/// Helper to align an integer value to a given alignment.
template <typename T>
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type align_to(T alignment, T value)
{
    return ((value + alignment - T(1)) / alignment) * alignment;
}

const uint32_t HEADER_VERSION = 0xA1C0DE01;
const uint32_t MAX_SUPPORTED_LAYERS = 8;

struct NetworkFileHeader
{
    uint32_t version = HEADER_VERSION;
    NetworkArchitecture netArch;
    NetworkLayer layers[MAX_SUPPORTED_LAYERS];
    MatrixLayout layout;
    size_t dataSize;
};

} // namespace

NetworkUtilities::NetworkUtilities(nvrhi::DeviceHandle device)
{
#if DONUT_WITH_VULKAN
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
    {
        m_coopVecUtils = std::make_unique<CoopVectorUtils_VK>(device->getNativeObject(nvrhi::ObjectTypes::VK_Device));
    }
#endif
#if DONUT_WITH_DX12
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        m_coopVecUtils = std::make_unique<CoopVectorUtils_DX12>(device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device));
    }
#endif
    assert(m_coopVecUtils && "Failed to create CoopVec Utility");
}

bool rtxns::NetworkUtilities::ValidateNetworkArchitecture(NetworkArchitecture const& netArch)
{
    if (netArch.numHiddenLayers + 1 > MAX_SUPPORTED_LAYERS)
    {
        log::error("Too many layers - %d > %d", netArch.numHiddenLayers + 1, MAX_SUPPORTED_LAYERS);
        return false;
    }

    if (netArch.inputNeurons * netArch.outputNeurons * netArch.hiddenNeurons == 0)
    {
        log::error("Neuron counts must all be positive - (%d, %d, %d)", netArch.inputNeurons, netArch.outputNeurons, netArch.hiddenNeurons);
        return false;
    }

    // Only Float16 weights are supported in the SDK
    if (netArch.weightPrecision != Precision::F16)
    {
        log::error("Weight precision not supported - must be f16.");
        return false;
    }

    if (netArch.biasPrecision != Precision::F16)
    {
        log::error("Bias precision not supported - must be f16.");
        return false;
    }

    return true;
}

// Create host side network layout
rtxns::NetworkLayout rtxns::NetworkUtilities::CreateHostNetworkLayout(NetworkArchitecture const& netArch)
{
    NetworkLayout layout;
    layout.matrixPrecision = netArch.weightPrecision;
    layout.matrixLayout = MatrixLayout::RowMajor; // Host side matrix layout

    const uint32_t numLayers = netArch.numHiddenLayers + 1; // hidden layers + input
    size_t offset = 0;

    layout.networkLayers.clear();

    // Create a network layout from the provides architecture
    for (uint32_t i = 0; i < numLayers; i++)
    {
        uint32_t inputs = (i == 0) ? netArch.inputNeurons : netArch.hiddenNeurons;
        uint32_t outputs = (i == numLayers - 1) ? netArch.outputNeurons : netArch.hiddenNeurons;

        NetworkLayer layer = {};
        layer.inputs = inputs;
        layer.outputs = outputs;

        layout.networkLayers.push_back(layer);
    }

    SetNetworkLayerSizes(layout);

    return layout;
}

// Set the weight and bias size / offsets for each layer in the network.
void NetworkUtilities::SetNetworkLayerSizes(NetworkLayout& layout)
{
    size_t offset = 0;
    // Calculate size and offsets for the new layout
    for (int i = 0; i < layout.networkLayers.size(); i++)
    {
        NetworkLayer& layer = layout.networkLayers[i];
        layer.weightSize = m_coopVecUtils->QueryMatrixByteSize(layer.outputs, layer.inputs, layout.matrixLayout, layout.matrixPrecision);
        layer.biasSize = layer.outputs * GetSize(layout.matrixPrecision);

        offset = align_to(m_coopVecUtils->GetMatrixAlignment(), offset);
        layer.weightOffset = (uint32_t)offset;
        offset += layer.weightSize;

        offset = align_to(m_coopVecUtils->GetVectorAlignment(), offset);
        layer.biasOffset = (uint32_t)offset;
        offset += layer.biasSize;
    }
    offset = align_to(m_coopVecUtils->GetMatrixAlignment(), offset);
    layout.networkSize = offset;
}

// Returns a updated network layout where the weights and bias size / offsets have been update
// for the new matrix layout..
// Can be device optimal matrix layout
rtxns::NetworkLayout NetworkUtilities::GetNewMatrixLayout(NetworkLayout const& srcLayout, MatrixLayout newMatrixLayout)
{
    NetworkLayout newLayout = srcLayout;

    // Check the new matrix layout does not match the current one
    if (newMatrixLayout == srcLayout.matrixLayout)
    {
        return newLayout;
    }
    newLayout.matrixLayout = newMatrixLayout;
    SetNetworkLayerSizes(newLayout);
    return newLayout;
}

// Converts weights and bias buffers from src layout to the dst layout.
// Both buffers must be device side.
// Both networks must be of the same network layout, only differing in MatrixLayout
void NetworkUtilities::ConvertWeights(NetworkLayout const& srcLayout,
                                      NetworkLayout const& dstLayout,
                                      nvrhi::BufferHandle srcBuffer,
                                      uint64_t srcBufferOffset,
                                      nvrhi::BufferHandle dstBuffer,
                                      uint64_t dstBufferOffset,
                                      nvrhi::DeviceHandle device,
                                      nvrhi::CommandListHandle commandList)
{
    // Unwrap the command list and buffer objects from NVRHI
    bool const isVulkan = device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN;
    nvrhi::ObjectType const commandListType = isVulkan ? nvrhi::ObjectTypes::VK_CommandBuffer : nvrhi::ObjectTypes::D3D12_GraphicsCommandList;
    nvrhi::ObjectType const bufferType = isVulkan ? nvrhi::ObjectTypes::VK_Buffer : nvrhi::ObjectTypes::D3D12_Resource;

    void* nativeCommandList = commandList->getNativeObject(commandListType);
    void* nativeSrcBuffer = srcBuffer->getNativeObject(bufferType);
    void* nativeDstBuffer = dstBuffer->getNativeObject(bufferType);

    // Set required barriers
    commandList->setBufferState(srcBuffer, nvrhi::ResourceStates::ShaderResource);
    commandList->setBufferState(dstBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->commitBarriers();

    // Convert the matrix parameters on the device
    m_coopVecUtils->ConvertDeviceMatrixLayout(srcLayout, dstLayout, nativeSrcBuffer, 0, nativeDstBuffer, 0, nativeCommandList);
}

HostNetwork::HostNetwork(std::shared_ptr<NetworkUtilities> networkUtils) : m_networkUtils(networkUtils)
{
    assert(m_networkUtils && "Network Utilities not present");
}

// Create host side network with initial values.
bool rtxns::HostNetwork::Initialise(const NetworkArchitecture& netArch)
{
    m_networkArchitecture = netArch;
    if (!m_networkUtils->ValidateNetworkArchitecture(m_networkArchitecture))
    {
        log::error("CreateTrainingNetwork: Failed to validate network.");
        return false;
    }

    // Compute size and offset of each weight matrix and bias vector.
    // These are placed after each other in memory with padding to fulfill the alignment requirements.
    m_networkLayout = m_networkUtils->CreateHostNetworkLayout(m_networkArchitecture);

    // Initialize the weight and bias
    m_networkParams.clear();
    m_networkParams.resize(m_networkLayout.networkSize, 0);

    static std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (uint32_t i = 0; i < m_networkLayout.networkLayers.size(); i++)
    {
        const auto& layer = m_networkLayout.networkLayers[i];
        std::vector<uint16_t> weights;
        weights.resize(size_t(layer.inputs * layer.outputs), 0);
        std::generate(weights.begin(), weights.end(), [&, k = sqrt(6.f / (layer.inputs + layer.outputs))]() { return rtxns::float32ToFloat16(dist(gen) * k); });
        std::memcpy(m_networkParams.data() + layer.weightOffset, weights.data(), layer.weightSize);

        std::vector<uint16_t> bias(layer.outputs);
        std::generate(bias.begin(), bias.end(), [&, k = sqrt(6.f / bias.size())]() { return rtxns::float32ToFloat16(dist(gen) * k); });
        std::memcpy(m_networkParams.data() + layer.biasOffset, bias.data(), layer.biasSize);
    }
    return true;
}

// Create host side network of provided architecture and initial values from a json file.
bool HostNetwork::InitialiseFromJson(donut::vfs::IFileSystem& fs, const std::string& fileName)
{
    // loads an inference data set
    Json::Value js;
    if (!json::LoadFromFile(fs, fileName, js))
    {
        log::error("LoadFromJson: Failed to load input file.");
        return false;
    }

    Json::Value jChannels = js["channels"];
    std::vector<int> channels(jChannels.size());
    transform(jChannels.begin(), jChannels.end(), channels.begin(), [](const auto& e) { return e.asInt(); });

    uint32_t numLayers = (uint32_t)channels.size() - 1;

    if (numLayers > MAX_SUPPORTED_LAYERS)
    {
        log::error("LoadFromJson: Number of layers not supported %d > %d.", numLayers, MAX_SUPPORTED_LAYERS);
        return false;
    }

    if (numLayers < 2)
    {
        log::error("LoadFromJson: Number of layers not supported %d <= 2.", numLayers);
        return false;
    }

    m_networkArchitecture.biasPrecision = Precision::F16;
    m_networkArchitecture.weightPrecision = Precision::F16;
    m_networkArchitecture.inputNeurons = channels[0];
    m_networkArchitecture.hiddenNeurons = channels[1]; // TOD0 Validate - we currently only support same size hidden layers
    m_networkArchitecture.outputNeurons = channels[channels.size() - 1];
    m_networkArchitecture.numHiddenLayers = numLayers - 1;

    m_networkLayout = m_networkUtils->CreateHostNetworkLayout(m_networkArchitecture);

    // Copy weights and biases into host side format, stored contiguously in one buffer.
    Json::Value jsonLayers = js["layers"];
    m_networkParams.clear();
    m_networkParams.resize(m_networkLayout.networkSize, 0);

    for (uint32_t ii = 0; ii < numLayers; ii++)
    {
        const NetworkLayer& layer = m_networkLayout.networkLayers[ii];

        // Copy weights into host layout.
        Json::Value jWeights = jsonLayers["layer_" + std::to_string(ii) + "_w"];
        assert(jWeights.size() == layer.inputs * layer.outputs && "Unexpected number of weights");

        std::vector<uint16_t> weights16(jWeights.size());
        transform(jWeights.begin(), jWeights.end(), weights16.begin(), [](const auto& e) { return rtxns::float32ToFloat16(e.asFloat()); });
        std::memcpy(m_networkParams.data() + layer.weightOffset, weights16.data(), layer.weightSize);

        // Copy biases into host layout.
        Json::Value jBiases = jsonLayers["layer_" + std::to_string(ii) + "_b"];
        assert(jBiases.size() == layer.outputs && "Unexpected number of biases");

        std::vector<uint16_t> bias16(jBiases.size());
        transform(jBiases.begin(), jBiases.end(), bias16.begin(), [](const auto& e) { return rtxns::float32ToFloat16(e.asFloat()); });
        std::memcpy(m_networkParams.data() + layer.biasOffset, bias16.data(), layer.biasSize);
    }

    return true;
}

// Create host side network of provided architecture and initial values from a json file.
bool HostNetwork::InitialiseFromFile(const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary);
    if (file.is_open())
    {
        NetworkFileHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (header.version != HEADER_VERSION)
        {
            log::error("Invalid file header");
            return false;
        }
        if (!m_networkUtils->ValidateNetworkArchitecture(header.netArch))
        {
            log::error("LoadFromFile: Failed to validate network.");
            return false;
        }

        m_networkArchitecture = header.netArch;
        m_networkLayout.matrixLayout = header.layout;
        m_networkLayout.matrixPrecision = header.netArch.weightPrecision;

        m_networkLayout.networkLayers.clear();
        for (uint32_t ii = 0; ii < m_networkArchitecture.numHiddenLayers + 1; ii++)
        {
            m_networkLayout.networkLayers.push_back(header.layers[ii]);
        }

        m_networkParams.resize(header.dataSize);

        file.read(reinterpret_cast<char*>(m_networkParams.data()), header.dataSize);
        file.close();
        return true;
    }
    log::error("File not found");
    return false;
}

// Create host side network from an existing network.
bool HostNetwork::InitialiseFromNetwork(HostNetwork const& network)
{
    if (!m_networkUtils->ValidateNetworkArchitecture(network.GetNetworkArchitecture()))
    {
        log::error("Failed to validate network.");
        return false;
    }

    *this = network;
    return true;
}

// Write the current network and parameters to file.
bool HostNetwork::WriteToFile(const std::string& fileName)
{
    // Write the buffer data to a file
    std::ofstream file(fileName, std::ios::binary);
    if (file.is_open())
    {
        NetworkFileHeader header;
        header.version = HEADER_VERSION;
        header.netArch = m_networkArchitecture;
        header.layout = m_networkLayout.matrixLayout;

        for (uint32_t ii = 0; ii < m_networkLayout.networkLayers.size(); ii++)
        {
            header.layers[ii] = m_networkLayout.networkLayers[ii];
        }
        header.dataSize = m_networkParams.size();
        file.write(reinterpret_cast<char*>(&header), sizeof(header));
        file.write(reinterpret_cast<char*>(m_networkParams.data()), m_networkParams.size());
        file.close();
        return true;
    }
    log::error("Failed to open the file for writing!");
    return false;
}

// Convert device layout to host layout and update the host side parameters.
void HostNetwork::UpdateFromBufferToFile(nvrhi::BufferHandle hostLayoutBuffer,
                                         nvrhi::BufferHandle deviceLayoutBuffer,
                                         NetworkLayout const& hostLayout,
                                         NetworkLayout const& deviceLayout,
                                         const std::string& fileName,
                                         nvrhi::DeviceHandle device,
                                         nvrhi::CommandListHandle commandList)
{
    commandList->open();

    // Convert device layout to a host layout
    m_networkUtils->ConvertWeights(deviceLayout, hostLayout, deviceLayoutBuffer, 0, hostLayoutBuffer, 0, device, commandList);

    commandList->setBufferState(hostLayoutBuffer, nvrhi::ResourceStates::CopySource);
    commandList->commitBarriers();

    // Create a staging buffer for reading back from the GPU
    nvrhi::BufferDesc stagingDesc;
    size_t bufferSize = hostLayoutBuffer->getDesc().byteSize;
    stagingDesc.byteSize = bufferSize;
    stagingDesc.cpuAccess = nvrhi::CpuAccessMode::Read; // This allows the CPU to read the data
    // stagingDesc.isVolatile = true; // Staging buffers are typically volatile
    stagingDesc.debugName = "Staging Buffer";

    // Allocate the staging buffer
    nvrhi::BufferHandle stagingBuffer = device->createBuffer(stagingDesc);
    if (!stagingBuffer)
    {
        log::error("Failed to create a staging buffer!");
        return;
    }

    // Copy data from the GPU buffer to the staging buffer
    commandList->copyBuffer(stagingBuffer, 0, hostLayoutBuffer, 0, bufferSize);
    commandList->close();
    device->executeCommandList(commandList);

    // Map the staging buffer to CPU memory
    void* mappedData = device->mapBuffer(stagingBuffer, nvrhi::CpuAccessMode::Read);
    if (!mappedData)
    {
        log::error("Failed to map the staging buffer!");
        return;
    }

    // The buffer size should match the current parameters size
    // if not the layout may have changed
    if (m_networkParams.size() != bufferSize)
    {
        m_networkParams.resize(bufferSize);
    }
    std::memcpy(m_networkParams.data(), mappedData, bufferSize);

    // Unmap and clean up
    device->unmapBuffer(stagingBuffer);
    WriteToFile(fileName);
}
