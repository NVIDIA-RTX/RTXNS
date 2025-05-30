/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/engine/View.h>
#include <donut/app/ApplicationBase.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/UserInterfaceUtils.h>
#include <donut/app/Camera.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/Scene.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/BindingCache.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <donut/core/json.h>
#include <nvrhi/utils.h>
#include <random>
#include <fstream>
#include <sstream>

#include "DeviceUtils.h"
#include "GraphicsResources.h"
#include "CoopVector.h"
#include "GeometryUtils.h"
#include "NeuralNetwork.h"
#include "DirectoryHelper.h"

using namespace donut;
using namespace donut::math;

#include "NetworkConfig.h"
#include <donut/shaders/view_cb.h>

static const char* g_windowTitle = "RTX Neural Shading Example: Simple Training (Ground Truth | Training | Loss )";

struct UIData
{
    bool reset = false;
    bool training = true;
    bool load = false;
    std::string fileName;
    float trainingTime = 0.0f;
    uint32_t epochs = 0;
    NetworkTransform networkTransform = NetworkTransform::Identity;
};

class SimpleTraining : public app::ApplicationBase
{
public:
    SimpleTraining(app::DeviceManager* deviceManager, UIData* ui) : ApplicationBase(deviceManager), m_uiParams(ui)
    {
    }

    bool Init()
    {
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/SimpleTraining" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        m_RootFS = std::make_shared<vfs::RootFileSystem>();
        m_RootFS->mount("/shaders/donut", frameworkShaderPath);
        m_RootFS->mount("/shaders/app", appShaderPath);

        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), m_RootFS, "/shaders");
        m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_ShaderFactory);
        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        m_commandList = GetDevice()->createCommandList();
        m_commandList->open();

        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        m_TextureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS, m_DescriptorTableManager);

        const std::filesystem::path dataPath = GetLocalPath("assets/data");
        std::filesystem::path textureFileName = dataPath / "nvidia-logo.png";
        std::shared_ptr<engine::LoadedTexture> texture = m_TextureCache->LoadTextureFromFile(textureFileName, true, nullptr, m_commandList);
        if (texture->texture == nullptr)
        {
            log::error("Failed to load texture.");
            return false;
        }
        m_InputTexture = texture->texture;

        ////////////////////
        //
        // Create the Neural network class and initialise it the hyper parameters from NetworkConfig.h.
        //
        ////////////////////
        m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
        m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);

        // Store the network architecture that is expected in the shaders
        m_shaderNetworkArch = {};
        m_shaderNetworkArch.inputNeurons = INPUT_NEURONS;
        m_shaderNetworkArch.hiddenNeurons = HIDDEN_NEURONS;
        m_shaderNetworkArch.outputNeurons = OUTPUT_NEURONS;
        m_shaderNetworkArch.numHiddenLayers = NUM_HIDDEN_LAYERS;
        m_shaderNetworkArch.biasPrecision = NETWORK_PRECISION;
        m_shaderNetworkArch.weightPrecision = NETWORK_PRECISION;

        if (!m_neuralNetwork->Initialise(m_shaderNetworkArch))
        {
            log::error("Failed to create a network.");
            return false;
        }

        // Get a device optimized layout
        m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

        ////////////////////
        //
        // Create the shaders/buffers/textures for the Neural Training
        //
        ////////////////////
        m_InferencePass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SimpleTraining_Inference", "inference_cs", nullptr, nvrhi::ShaderType::Compute);
        m_TrainingPass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SimpleTraining_Training", "training_cs", nullptr, nvrhi::ShaderType::Compute);
        m_OptimizerPass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SimpleTraining_Optimizer", "adam_cs", nullptr, nvrhi::ShaderType::Compute);
        m_ConvertWeightsPass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SimpleTraining_Optimizer", "convert_weights_cs", nullptr, nvrhi::ShaderType::Compute);

        const auto& params = m_neuralNetwork->GetNetworkParams();

        nvrhi::BufferDesc paramsBufferDesc;
        paramsBufferDesc.byteSize = params.size();
        paramsBufferDesc.debugName = "MLPParamsHostBuffer";
        paramsBufferDesc.canHaveUAVs = true;
        paramsBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        paramsBufferDesc.keepInitialState = true;
        m_mlpHostBuffer = GetDevice()->createBuffer(paramsBufferDesc);

        // Create a buffer for a device optimized parameters layout
        paramsBufferDesc.byteSize = m_deviceNetworkLayout.networkSize;
        paramsBufferDesc.canHaveRawViews = true;
        paramsBufferDesc.canHaveUAVs = true;
        paramsBufferDesc.canHaveTypedViews = true;
        paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
        paramsBufferDesc.debugName = "MLPParamsByteAddressBuffer";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        m_mlpDeviceBuffer = GetDevice()->createBuffer(paramsBufferDesc);

        // Upload the parameters
        UpdateDeviceNetworkParameters(m_commandList);

        m_TotalParamCount = (uint32_t)(paramsBufferDesc.byteSize / sizeof(uint16_t));

        paramsBufferDesc.debugName = "MLPParametersFloat";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        paramsBufferDesc.byteSize = m_TotalParamCount * sizeof(float); // convert to float
        paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
        m_mlpDeviceFloatBuffer = GetDevice()->createBuffer(paramsBufferDesc);
        m_commandList->beginTrackingBufferState(m_mlpDeviceFloatBuffer, nvrhi::ResourceStates::UnorderedAccess);
        m_commandList->clearBufferUInt(m_mlpDeviceFloatBuffer, 0);

        paramsBufferDesc.debugName = "MLPGradientsBuffer";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        paramsBufferDesc.byteSize = (m_TotalParamCount * sizeof(uint16_t) + 3) & ~3; // Round up to nearest multiple of 4
        paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
        m_mlpGradientsBuffer = GetDevice()->createBuffer(paramsBufferDesc);
        m_commandList->beginTrackingBufferState(m_mlpGradientsBuffer, nvrhi::ResourceStates::UnorderedAccess);
        m_commandList->clearBufferUInt(m_mlpGradientsBuffer, 0);

        paramsBufferDesc.debugName = "MLPMoments1Buffer";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        paramsBufferDesc.byteSize = m_TotalParamCount * sizeof(float);
        paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
        paramsBufferDesc.canHaveRawViews = false;
        m_mlpMoments1Buffer = GetDevice()->createBuffer(paramsBufferDesc);
        m_commandList->beginTrackingBufferState(m_mlpMoments1Buffer, nvrhi::ResourceStates::UnorderedAccess);
        m_commandList->clearBufferUInt(m_mlpMoments1Buffer, 0);

        paramsBufferDesc.debugName = "MLPMoments2Buffer";
        m_mlpMoments2Buffer = GetDevice()->createBuffer(paramsBufferDesc);
        m_commandList->beginTrackingBufferState(m_mlpMoments2Buffer, nvrhi::ResourceStates::UnorderedAccess);
        m_commandList->clearBufferUInt(m_mlpMoments2Buffer, 0);

        uint32_t imageSize = m_InputTexture->getDesc().width * m_InputTexture->getDesc().height;
        paramsBufferDesc.debugName = "RandStateBuffer";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        paramsBufferDesc.byteSize = BATCH_SIZE_X * BATCH_SIZE_Y * 4;
        paramsBufferDesc.structStride = sizeof(uint32_t);
        m_RandStateBuffer = GetDevice()->createBuffer(paramsBufferDesc);

        std::mt19937 gen(1337);
        std::uniform_int_distribution<uint32_t> dist;
        std::vector<uint32_t> buff(BATCH_SIZE_X * BATCH_SIZE_Y);
        for (uint32_t i = 0; i < buff.size(); i++)
        {
            buff[i] = dist(gen);
        }

        m_commandList->writeBuffer(m_RandStateBuffer, buff.data(), buff.size() * sizeof(uint32_t));
        m_commandList->beginTrackingBufferState(m_RandStateBuffer, nvrhi::ResourceStates::UnorderedAccess);

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
        GetDevice()->waitForIdle();

        auto inputTexDesc = m_InputTexture->getDesc();
        inputTexDesc.debugName = "InferenceTexture";
        inputTexDesc.format = nvrhi::Format::RGBA16_FLOAT;
        inputTexDesc.isUAV = true;
        inputTexDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        inputTexDesc.keepInitialState = true;
        m_InferenceTexture = GetDevice()->createTexture(inputTexDesc);

        inputTexDesc.debugName = "LossTexture";
        m_LossTexture = GetDevice()->createTexture(inputTexDesc);

        // Set up the constant buffers
        m_NeuralConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(NeuralConstants), "NeuralConstantBuffer")
                                                               .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                               .setKeepInitialState(true));

        ////////////////////
        //
        // Create the pipelines for each neural pass
        //
        ////////////////////
        // Inference Pass
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_NeuralConstantBuffer),
            nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer),
            nvrhi::BindingSetItem::Texture_SRV(1, m_InputTexture),
            nvrhi::BindingSetItem::Texture_UAV(0, m_InferenceTexture),
        };
        nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_InferencePass.m_BindingLayout, m_InferencePass.m_BindingSet);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { m_InferencePass.m_BindingLayout };
        pipelineDesc.CS = m_InferencePass.m_ShaderCS;
        m_InferencePass.m_Pipeline = GetDevice()->createComputePipeline(pipelineDesc);

        // Training Pass
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_NeuralConstantBuffer),
            nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer),
            nvrhi::BindingSetItem::Texture_SRV(1, m_InputTexture),
            nvrhi::BindingSetItem::RawBuffer_UAV(0, m_mlpGradientsBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_RandStateBuffer),
            nvrhi::BindingSetItem::Texture_UAV(2, m_InferenceTexture),
            nvrhi::BindingSetItem::Texture_UAV(3, m_LossTexture),
        };
        nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_TrainingPass.m_BindingLayout, m_TrainingPass.m_BindingSet);

        pipelineDesc.bindingLayouts = { m_TrainingPass.m_BindingLayout };
        pipelineDesc.CS = m_TrainingPass.m_ShaderCS;
        m_TrainingPass.m_Pipeline = GetDevice()->createComputePipeline(pipelineDesc);

        // Optimization Pass
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_NeuralConstantBuffer),  nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_mlpDeviceBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_mlpDeviceFloatBuffer), nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_mlpGradientsBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(3, m_mlpMoments1Buffer),    nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_mlpMoments2Buffer),
        };
        nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_OptimizerPass.m_BindingLayout, m_OptimizerPass.m_BindingSet);

        pipelineDesc.bindingLayouts = { m_OptimizerPass.m_BindingLayout };
        pipelineDesc.CS = m_OptimizerPass.m_ShaderCS;
        m_OptimizerPass.m_Pipeline = GetDevice()->createComputePipeline(pipelineDesc);

        // Convert weights to float
        nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_ConvertWeightsPass.m_BindingLayout, m_ConvertWeightsPass.m_BindingSet);
        pipelineDesc.bindingLayouts = { m_ConvertWeightsPass.m_BindingLayout };
        pipelineDesc.CS = m_ConvertWeightsPass.m_ShaderCS;
        m_ConvertWeightsPass.m_Pipeline = GetDevice()->createComputePipeline(pipelineDesc);

        return true;
    }

    // expects an open command list
    void UpdateDeviceNetworkParameters(nvrhi::CommandListHandle commandList)
    {
        // Upload the host side parameters
        m_commandList->setBufferState(m_mlpHostBuffer, nvrhi::ResourceStates::CopyDest);
        m_commandList->commitBarriers();
        m_commandList->writeBuffer(m_mlpHostBuffer, m_neuralNetwork->GetNetworkParams().data(), m_neuralNetwork->GetNetworkParams().size());

        // Convert to GPU optimized layout
        m_networkUtils->ConvertWeights(m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

        // Update barriers for use
        m_commandList->setBufferState(m_mlpDeviceBuffer, nvrhi::ResourceStates::ShaderResource);
        m_commandList->commitBarriers();
        m_convertWeights = true;
    }

    // expects an open command list
    void ResetTrainingData(nvrhi::CommandListHandle commandList)
    {
        // Validate the loaded file matches the shaders
        if (!m_neuralNetwork->Initialise(m_shaderNetworkArch))
        {
            log::error("Failed to create a network.");
            return;
        }

        // Get a device optimized layout
        m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

        // Upload the parameters
        UpdateDeviceNetworkParameters(commandList);

        // Clear buffers
        commandList->clearBufferUInt(m_mlpDeviceFloatBuffer, 0);
        commandList->clearBufferUInt(m_mlpGradientsBuffer, 0);
        commandList->clearBufferUInt(m_mlpMoments1Buffer, 0);
        commandList->clearBufferUInt(m_mlpMoments2Buffer, 0);

        m_uiParams->epochs = 0;
        m_uiParams->trainingTime = 0.0f;

        m_AdamCurrentStep = 1;
    }

    bool LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) override
    {
        engine::Scene* scene = new engine::Scene(GetDevice(), *m_ShaderFactory, fs, m_TextureCache, m_DescriptorTableManager, nullptr);

        if (scene->Load(sceneFileName))
        {
            m_Scene = std::unique_ptr<engine::Scene>(scene);
            return true;
        }

        return false;
    }

    std::shared_ptr<engine::ShaderFactory> GetShaderFactory() const
    {
        return m_ShaderFactory;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        if (m_uiParams->training)
        {
            m_uiParams->trainingTime += fElapsedTimeSeconds;
        }

        GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true);

        ////////////////////
        //
        // Load/Save the Neural network if required
        //
        ////////////////////
        if (!m_uiParams->fileName.empty())
        {
            if (m_uiParams->load)
            {
                // rtxns::Network network(GetDevice());
                if (m_neuralNetwork->InitialiseFromFile(m_uiParams->fileName))
                {
                    // Validate the loaded file against what the shaders expect
                    if (m_networkUtils->ValidateNetworkArchitecture(m_shaderNetworkArch))
                    {
                        m_commandList = GetDevice()->createCommandList();
                        m_commandList->open();

                        // Get a device optimized layout
                        m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

                        // Upload the parameters
                        UpdateDeviceNetworkParameters(m_commandList);

                        // Clear buffers
                        m_commandList->clearBufferUInt(m_mlpDeviceFloatBuffer, 0);
                        m_commandList->clearBufferUInt(m_mlpGradientsBuffer, 0);
                        m_commandList->clearBufferUInt(m_mlpMoments1Buffer, 0);
                        m_commandList->clearBufferUInt(m_mlpMoments2Buffer, 0);

                        m_uiParams->epochs = 0;
                        m_uiParams->trainingTime = 0.0f;

                        m_AdamCurrentStep = 1;

                        m_commandList->close();
                        GetDevice()->executeCommandList(m_commandList);
                    }
                }
            }
            else
            {
                m_neuralNetwork->UpdateFromBufferToFile(
                    m_mlpHostBuffer, m_mlpDeviceBuffer, m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout, m_uiParams->fileName, GetDevice(), m_commandList);
            }
            m_uiParams->fileName = "";
        }
    }

    void BackBufferResizing() override
    {
        m_Framebuffer = nullptr;
        m_BindingCache->Clear();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        m_commandList->open();

        if (m_uiParams->reset)
        {
            ResetTrainingData(m_commandList);
            m_uiParams->reset = false;
        }

        nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.f));

        ////////////////////
        //
        // Update the Constant buffer
        //
        ////////////////////
        NeuralConstants neuralConstants = {};

        for (int i = 0; i < NUM_TRANSITIONS; ++i)
        {
            neuralConstants.weightOffsets[i / 4][i % 4] = m_deviceNetworkLayout.networkLayers[i].weightOffset;
            neuralConstants.biasOffsets[i / 4][i % 4] = m_deviceNetworkLayout.networkLayers[i].biasOffset;
        }

        neuralConstants.imageWidth = m_InferenceTexture->getDesc().width;
        neuralConstants.imageHeight = m_InferenceTexture->getDesc().height;
        neuralConstants.maxParamSize = m_TotalParamCount;
        neuralConstants.learningRate = m_LearningRate;
        neuralConstants.currentStep = m_AdamCurrentStep;
        neuralConstants.batchSizeX = BATCH_SIZE_X;
        neuralConstants.batchSizeY = BATCH_SIZE_Y;
        neuralConstants.networkTransform = m_uiParams->networkTransform;
        m_commandList->writeBuffer(m_NeuralConstantBuffer, &neuralConstants, sizeof(neuralConstants));

        nvrhi::ComputeState state;

        ////////////////////
        //
        // Start the training loop
        //
        ////////////////////
        if (m_uiParams->training)
        {
            if (m_convertWeights)
            {
                state.bindings = { m_ConvertWeightsPass.m_BindingSet };
                state.pipeline = m_ConvertWeightsPass.m_Pipeline;
                m_commandList->beginMarker("ConvertWeights");
                m_commandList->setComputeState(state);
                m_commandList->dispatch(dm::div_ceil(m_TotalParamCount, 32), 1, 1);
                m_commandList->endMarker();
                m_convertWeights = false;
            }

            for (uint32_t batch = 0; batch < BATCH_COUNT; batch++)
            {
                // run the training pass
                state.bindings = { m_TrainingPass.m_BindingSet };
                state.pipeline = m_TrainingPass.m_Pipeline;
                m_commandList->beginMarker("Training");
                m_commandList->setComputeState(state);
                m_commandList->dispatch(dm::div_ceil(BATCH_SIZE_X, 8), dm::div_ceil(BATCH_SIZE_Y, 8), 1);
                m_commandList->endMarker();

                // optimizer pass
                state.bindings = { m_OptimizerPass.m_BindingSet };
                state.pipeline = m_OptimizerPass.m_Pipeline;
                m_commandList->beginMarker("Update Weights");
                m_commandList->setComputeState(state);
                m_commandList->dispatch(dm::div_ceil(m_TotalParamCount, 32), 1, 1);
                m_commandList->endMarker();

                neuralConstants.currentStep = ++m_AdamCurrentStep;
                m_commandList->writeBuffer(m_NeuralConstantBuffer, &neuralConstants, sizeof(neuralConstants));
            }
            m_uiParams->epochs++;
        }

        {
            // inference pass
            state.bindings = { m_InferencePass.m_BindingSet };
            state.pipeline = m_InferencePass.m_Pipeline;
            m_commandList->beginMarker("Inference");
            m_commandList->setComputeState(state);
            m_commandList->dispatch(dm::div_ceil(m_InferenceTexture->getDesc().width, 8), dm::div_ceil(m_InferenceTexture->getDesc().height, 8), 1);
            m_commandList->endMarker();
        }

        ////////////////////
        //
        // Render the outputs
        //
        ////////////////////
        for (uint32_t viewIndex = 0; viewIndex < 3; ++viewIndex)
        {
            // Construct the viewport so that all viewports form a grid.
            const float width = float(fbinfo.width) / 3;
            const float height = float(fbinfo.height);
            const float left = width * viewIndex;
            const float top = 0;

            const nvrhi::Viewport viewport = nvrhi::Viewport(left, left + width, top, top + height, 0.f, 1.f);

            if (viewIndex == 0)
            {
                // Draw original image
                engine::BlitParameters blitParams;
                blitParams.targetFramebuffer = framebuffer;
                blitParams.targetViewport = viewport;
                blitParams.sourceTexture = m_InputTexture;
                m_CommonPasses->BlitTexture(m_commandList, blitParams, m_BindingCache.get());
            }
            else if (viewIndex == 1)
            {
                // Draw inferenced image
                engine::BlitParameters blitParams;
                blitParams.targetFramebuffer = framebuffer;
                blitParams.targetViewport = viewport;
                blitParams.sourceTexture = m_InferenceTexture;
                m_CommonPasses->BlitTexture(m_commandList, blitParams, m_BindingCache.get());
            }
            else if (viewIndex == 2)
            {
                // Draw loss image
                engine::BlitParameters blitParams;
                blitParams.targetFramebuffer = framebuffer;
                blitParams.targetViewport = viewport;
                blitParams.sourceTexture = m_LossTexture;
                m_CommonPasses->BlitTexture(m_commandList, blitParams, m_BindingCache.get());
            }
        }

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
    }

private:
    std::shared_ptr<vfs::RootFileSystem> m_RootFS;

    nvrhi::CommandListHandle m_commandList;

    struct NeuralPass
    {
        nvrhi::ShaderHandle m_ShaderCS;
        nvrhi::BindingLayoutHandle m_BindingLayout;
        nvrhi::BindingSetHandle m_BindingSet;
        nvrhi::ComputePipelineHandle m_Pipeline;
    };

    NeuralPass m_InferencePass;
    NeuralPass m_TrainingPass;
    NeuralPass m_OptimizerPass;
    NeuralPass m_ConvertWeightsPass;

    nvrhi::BufferHandle m_NeuralConstantBuffer;
    nvrhi::BufferHandle m_OptimisationConstantBuffer;

    nvrhi::TextureHandle m_InputTexture;
    nvrhi::TextureHandle m_InferenceTexture;
    nvrhi::TextureHandle m_LossTexture;

    nvrhi::BufferHandle m_mlpHostBuffer;
    nvrhi::BufferHandle m_mlpDeviceBuffer;
    nvrhi::BufferHandle m_mlpDeviceFloatBuffer;
    nvrhi::BufferHandle m_mlpGradientsBuffer;
    nvrhi::BufferHandle m_mlpMoments1Buffer;
    nvrhi::BufferHandle m_mlpMoments2Buffer;
    nvrhi::BufferHandle m_RandStateBuffer;

    nvrhi::FramebufferHandle m_Framebuffer;

    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::unique_ptr<engine::Scene> m_Scene;
    std::shared_ptr<engine::DescriptorTableManager> m_DescriptorTableManager;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    rtxns::NetworkArchitecture m_shaderNetworkArch;
    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;
    std::unique_ptr<rtxns::HostNetwork> m_neuralNetwork;
    rtxns::NetworkLayout m_deviceNetworkLayout;

    uint m_TotalParamCount = 0;
    uint m_AdamCurrentStep = 1;
    float m_LearningRate = LEARNING_RATE;
    bool m_convertWeights = true;

    UIData* m_uiParams;
};

class UserInterface : public app::ImGui_Renderer
{
public:
    UserInterface(app::DeviceManager* deviceManager, UIData* uiParams) : ImGui_Renderer(deviceManager), m_uiParams(uiParams)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void buildUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);

        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        bool reset = ImGui::Combo("##networkTransform", (int*)&m_uiParams->networkTransform,
                                  "1:1 Mapping\0"
                                  "Zoom\0"
                                  "X/Y Flip\0");
        ImGui::Text("Epochs : %d", m_uiParams->epochs);
        ImGui::Text("Training Time : %.2f s", m_uiParams->trainingTime);
        if (ImGui::Button(m_uiParams->training ? "Disable Training" : "Enable Training"))
        {
            m_uiParams->training = !m_uiParams->training;
        }
        reset |= (ImGui::Button("Reset Training"));
        if (ImGui::Button("Load Model"))
        {
            std::string fileName;
            if (app::FileDialog(true, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
            {
                m_uiParams->fileName = fileName;
                m_uiParams->load = true;
            }
        }
        if (ImGui::Button("Save Model"))
        {
            std::string fileName;
            if (app::FileDialog(false, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
            {
                m_uiParams->fileName = fileName;
                m_uiParams->load = false;
            }
        }

        ImGui::End();

        if (reset)
        {
            m_uiParams->reset = true;
        }
    }

private:
    UIData* m_uiParams;
};

#ifdef WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI graphicsApi = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    if (graphicsApi == nvrhi::GraphicsAPI::D3D11)
    {
        log::error("This sample does not support D3D11.");
        return 1;
    }

    app::DeviceManager* deviceManager = app::DeviceManager::Create(graphicsApi);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif
    // w/h based on input texture
    deviceParams.backBufferWidth = 768 * 3;
    deviceParams.backBufferHeight = 768;

    ////////////////////
    //
    // Setup the CoopVector extensions.
    //
    ////////////////////
    SetCoopVectorExtensionParameters(deviceParams, graphicsApi, false, g_windowTitle);

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters. Please try a NVIDIA driver version greater than 570");
        return 1;
    }

    auto graphicsResources = std::make_unique<rtxns::GraphicsResources>(deviceManager->GetDevice());
    if (!graphicsResources->GetCoopVectorFeatures().inferenceSupported && !graphicsResources->GetCoopVectorFeatures().trainingSupported &&
        !graphicsResources->GetCoopVectorFeatures().fp16InferencingSupported && !graphicsResources->GetCoopVectorFeatures().fp16TrainingSupported)
    {
        log::fatal("Not all required Coop Vector features are available");
        return 1;
    }

    {
        UIData uiData;
        SimpleTraining example(deviceManager, &uiData);
        UserInterface gui(deviceManager, &uiData);
        if (example.Init() && gui.Init(example.GetShaderFactory()))
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->AddRenderPassToBack(&gui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&gui);
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
