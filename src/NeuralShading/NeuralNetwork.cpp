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
#include "CoopVector.h"
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
} // namespace

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

Network::Network(nvrhi::DeviceHandle device) : m_device(device), m_layout(MatrixLayout::RowMajor)
{
}

bool Network::ValidateNetworkArchitecture(const NetworkArchitecture& netArch)
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

    // NV doesn't support f32 weights
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

bool Network::Initialise(const NetworkArchitecture& netArch, MatrixLayout layout)
{
    if (!ValidateNetworkArchitecture(netArch))
    {
        log::error("CreateTrainingNetwork: Failed to validate network.");
        return false;
    }

    CoopVectorUtils_VK coopVecUtils(m_device->getNativeObject(nvrhi::ObjectTypes::VK_Device));
    m_networkArchitecture = netArch;
    m_layout = layout;

    const uint32_t numLayers = m_networkArchitecture.numHiddenLayers + 1; // hidden layers + input
    size_t offset = 0;

    m_networkLayers.clear();

    // Compute size and offset of each weight matrix and bias vector.
    // These are placed after each other in memory with padding to fulfill the alignment requirements.
    for (uint32_t i = 0; i < numLayers; i++)
    {
        uint32_t inputs = (i == 0) ? m_networkArchitecture.inputNeurons : m_networkArchitecture.hiddenNeurons;
        uint32_t outputs = (i == numLayers - 1) ? m_networkArchitecture.outputNeurons : m_networkArchitecture.hiddenNeurons;

        NetworkLayer layer = {};
        layer.inputs = inputs;
        layer.outputs = outputs;
        layer.weightSize = (uint32_t)coopVecUtils.QueryMatrixByteSize(layer.outputs, layer.inputs, m_layout, m_networkArchitecture.weightPrecision);
        layer.biasSize = layer.outputs * GetSize(m_networkArchitecture.biasPrecision);

        offset = align_to(coopVecUtils.GetMatrixAlignment(), offset);
        layer.weightOffset = (uint32_t)offset;
        offset += layer.weightSize;

        offset = align_to(coopVecUtils.GetVectorAlignment(), offset);
        layer.biasOffset = (uint32_t)offset;
        offset += layer.biasSize;

        m_networkLayers.push_back(layer);
    }

    // Initialize the weight and bias
    m_params.clear();
    m_params.resize(offset, 0);

    static std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (uint32_t i = 0; i < numLayers; i++)
    {
        const auto& layer = m_networkLayers[i];
        std::vector<float> weights;
        weights.resize(layer.inputs * layer.outputs, 0.f);
        std::generate(weights.begin(), weights.end(), [&, k = sqrt(6.f / (layer.inputs + layer.outputs))]() { return dist(gen) * k; });
        coopVecUtils.ConvertHostf32Matrix(layer.outputs, layer.inputs, weights.data(), weights.size() * sizeof(float), m_params.data() + layer.weightOffset, layer.weightSize,
                                          m_networkArchitecture.weightPrecision, m_layout);

        // Initialize biases then copy into GPU layout.
        // The biases are quantized to FP16 and stored contiguously.
        std::vector<float> bias(layer.outputs);
        std::generate(bias.begin(), bias.end(), [&, k = sqrt(6.f / bias.size())]() { return dist(gen) * k; });

        if (m_networkArchitecture.biasPrecision == Precision::F16)
        {
            std::vector<uint16_t> bias16(bias.size());
            std::transform(bias.begin(), bias.end(), bias16.begin(), [](float v) { return rtxns::float32ToFloat16(v); });
            std::memcpy(m_params.data() + layer.biasOffset, bias16.data(), layer.biasSize);
        }
        else
        {
            assert(0);
        }
    }

    return true;
}

bool Network::InitialiseFromNetwork(const Network& network, MatrixLayout layout)
{
    m_networkArchitecture = network.m_networkArchitecture;
    m_networkLayers = network.m_networkLayers;
    m_layout = layout;

    CoopVectorUtils_VK coopVecUtils(m_device->getNativeObject(nvrhi::ObjectTypes::VK_Device));
    const uint32_t numLayers = network.m_networkArchitecture.numHiddenLayers + 1; // hidden layers + input
    size_t offset = 0;

    // prepare the layers for the new sizes
    for (uint32_t ii = 0; ii < numLayers; ii++)
    {
        NetworkLayer& layer = m_networkLayers[ii];
        layer.weightSize = (uint32_t)coopVecUtils.QueryMatrixByteSize(layer.outputs, layer.inputs, layout, m_networkArchitecture.weightPrecision);
        layer.biasSize = layer.outputs * GetSize(m_networkArchitecture.biasPrecision);

        offset = align_to(coopVecUtils.GetMatrixAlignment(), offset);
        layer.weightOffset = (uint32_t)offset;
        offset += layer.weightSize;

        offset = align_to(coopVecUtils.GetVectorAlignment(), offset);
        layer.biasOffset = (uint32_t)offset;
        offset += layer.biasSize;
    }

    // Initialize the weight and bias
    m_params.clear();
    m_params.resize(offset, 0);

    // Copy the data over
    for (uint32_t ii = 0; ii < numLayers; ii++)
    {
        const NetworkLayer& dstLayer = m_networkLayers[ii];
        const NetworkLayer& srcLayer = network.m_networkLayers[ii];

        // Src is the original, dst is the new
        size_t actualSize = coopVecUtils.ConvertHostMatrixLayout(srcLayer.outputs, srcLayer.inputs, network.m_params.data() + srcLayer.weightOffset, srcLayer.weightSize,
                                                                 network.m_networkArchitecture.weightPrecision, network.m_layout, m_params.data() + dstLayer.weightOffset,
                                                                 dstLayer.weightSize, m_networkArchitecture.weightPrecision, m_layout);

        assert(actualSize == dstLayer.weightSize);

        std::memcpy(m_params.data() + dstLayer.biasOffset, network.m_params.data() + srcLayer.biasOffset, srcLayer.biasSize);
    }

    return true;
}

bool Network::InitialiseFromJson(donut::vfs::IFileSystem& fs, const std::string& fileName)
{
    // loads an inference data set
    CoopVectorUtils_VK coopVecUtils(m_device->getNativeObject(nvrhi::ObjectTypes::VK_Device));

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

    size_t offset = 0;
    m_networkLayers.clear();

    // Compute size and offset of each weight matrix and bias vector.
    // These are placed after each other in memory with padding to fulfill the alignment requirements.
    for (uint32_t ii = 0; ii < numLayers; ii++)
    {
        NetworkLayer layer = {};

        layer.inputs = channels[ii]; // cols
        layer.outputs = channels[ii + 1]; // rows
        layer.weightSize = (uint32_t)coopVecUtils.QueryMatrixByteSize(layer.outputs, layer.inputs, MatrixLayout::InferencingOptimal, m_networkArchitecture.weightPrecision);
        layer.biasSize = layer.outputs * sizeof(uint16_t);

        offset = align_to(coopVecUtils.GetMatrixAlignment(), offset);
        layer.weightOffset = (uint32_t)offset;
        offset += layer.weightSize;

        offset = align_to(coopVecUtils.GetVectorAlignment(), offset);
        layer.biasOffset = (uint32_t)offset;
        offset += layer.biasSize;

        m_networkLayers.push_back(layer);
    }

    // Copy weights and biases into GPU format, stored contiguously in one buffer.
    Json::Value jsonLayers = js["layers"];
    m_params.clear();
    m_params.resize(offset, 0);

    for (uint32_t ii = 0; ii < numLayers; ii++)
    {
        const NetworkLayer& layer = m_networkLayers[ii];

        // Copy weights into GPU layout.
        // We rely on the driver host side matrix conversion function for this.
        Json::Value jWeights = jsonLayers["layer_" + std::to_string(ii) + "_w"];
        assert(jWeights.size() == layer.inputs * layer.outputs && "Unexpected number of weights");

        std::vector<float> weights(jWeights.size());
        transform(jWeights.begin(), jWeights.end(), weights.begin(), [](const auto& e) { return e.asFloat(); });

        coopVecUtils.ConvertHostf32Matrix(layer.outputs, layer.inputs, weights.data(), weights.size() * sizeof(float), m_params.data() + layer.weightOffset, layer.weightSize,
                                          m_networkArchitecture.weightPrecision, MatrixLayout::InferencingOptimal);

        // Copy biases into GPU layout.
        // The biases are quantized to FP16 and stored contiguously.
        Json::Value jBiases = jsonLayers["layer_" + std::to_string(ii) + "_b"];
        assert(jBiases.size() == layer.outputs && "Unexpected number of biases");

        std::vector<uint16_t> bias16(jBiases.size());
        transform(jBiases.begin(), jBiases.end(), bias16.begin(), [](const auto& e) { return rtxns::float32ToFloat16(e.asFloat()); });
        std::memcpy(m_params.data() + layer.biasOffset, bias16.data(), layer.biasSize);
    }
    m_layout = MatrixLayout::InferencingOptimal;

    return true;
}

bool Network::InitialiseFromFile(const std::string& fileName)
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
        if (!ValidateNetworkArchitecture(header.netArch))
        {
            log::error("LoadFromFile: Failed to validate network.");
            return false;
        }
        m_layout = header.layout;
        m_networkArchitecture = header.netArch;
        m_networkLayers.clear();
        for (uint32_t ii = 0; ii < m_networkArchitecture.numHiddenLayers + 1; ii++)
        {
            m_networkLayers.push_back(header.layers[ii]);
        }

        m_params.resize(header.dataSize);

        file.read(reinterpret_cast<char*>(m_params.data()), header.dataSize);
        file.close();
        return true;
    }
    log::error("File not found");
    return false;
}

bool Network::WriteToFile(const std::string& fileName)
{
    // Write the buffer data to a file
    std::ofstream file(fileName, std::ios::binary);
    if (file.is_open())
    {
        NetworkFileHeader header;
        header.version = HEADER_VERSION;
        header.netArch = m_networkArchitecture;
        header.layout = m_layout;

        for (uint32_t ii = 0; ii < m_networkLayers.size(); ii++)
        {
            header.layers[ii] = m_networkLayers[ii];
        }
        header.dataSize = m_params.size();
        file.write(reinterpret_cast<char*>(&header), sizeof(header));
        file.write(reinterpret_cast<char*>(m_params.data()), m_params.size());
        file.close();
        return true;
    }
    log::error("Failed to open the file for writing!");
    return false;
}

void Network::UpdateFromBufferToFile(nvrhi::BufferHandle gpuBuffer, const std::string& fileName)
{
    // Create a staging buffer for reading back from the GPU
    nvrhi::BufferDesc stagingDesc;
    size_t bufferSize = gpuBuffer->getDesc().byteSize;
    stagingDesc.byteSize = bufferSize;
    stagingDesc.cpuAccess = nvrhi::CpuAccessMode::Read; // This allows the CPU to read the data
    // stagingDesc.isVolatile = true; // Staging buffers are typically volatile
    stagingDesc.debugName = "Staging Buffer";

    // Allocate the staging buffer
    nvrhi::BufferHandle stagingBuffer = m_device->createBuffer(stagingDesc);
    if (!stagingBuffer)
    {
        log::error("Failed to create a staging buffer!");
        return;
    }

    // Copy data from the GPU buffer to the staging buffer
    nvrhi::CommandListHandle commandList = m_device->createCommandList();
    commandList->open();
    commandList->copyBuffer(stagingBuffer, 0, gpuBuffer, 0, bufferSize);
    commandList->close();
    m_device->executeCommandList(commandList);

    // Map the staging buffer to CPU memory
    void* mappedData = m_device->mapBuffer(stagingBuffer, nvrhi::CpuAccessMode::Read);
    if (!mappedData)
    {
        log::error("Failed to map the staging buffer!");
        return;
    }

    // Resize and update the params
    m_params.resize(bufferSize);
    std::memcpy(m_params.data(), mappedData, bufferSize);

    {
        rtxns::Network outputNetwork(m_device);
        outputNetwork.InitialiseFromNetwork(*this, rtxns::MatrixLayout::RowMajor);
        outputNetwork.WriteToFile(fileName);
    }

    // Unmap and clean up
    m_device->unmapBuffer(stagingBuffer);
}

bool Network::ChangeLayout(MatrixLayout layout)
{
    rtxns::Network network(m_device);
    if (network.InitialiseFromNetwork(*this, layout))
    {
        *this = network;
        return true;
    }
    return false;
}
