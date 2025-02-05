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

        m_CommandList = GetDevice()->createCommandList();
        m_CommandList->open();

        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        m_TextureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS, m_DescriptorTableManager);

        const std::filesystem::path dataPath = GetLocalPath("assets/data");
        std::filesystem::path textureFileName = dataPath / "nvidia-logo.png";
        std::shared_ptr<engine::LoadedTexture> texture = m_TextureCache->LoadTextureFromFile(textureFileName, true, nullptr, m_CommandList);
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
        m_NeuralNetwork = std::make_unique<rtxns::Network>(GetDevice());

        // Validate the loaded file matches the shaders
        rtxns::NetworkArchitecture netArch = {};
        netArch.inputNeurons = INPUT_NEURONS;
        netArch.hiddenNeurons = HIDDEN_NEURONS;
        netArch.outputNeurons = OUTPUT_NEURONS;
        netArch.numHiddenLayers = NUM_HIDDEN_LAYERS;
        netArch.biasPrecision = NETWORK_PRECISION;
        netArch.weightPrecision = NETWORK_PRECISION;

        if (!m_NeuralNetwork->Initialise(netArch, rtxns::MatrixLayout::TrainingOptimal))
        {
            log::error("Failed to create a network.");
            return false;
        }

        ////////////////////
        //
        // Create the shaders/buffers/textures for the Neural Training
        //
        ////////////////////
        m_InferencePass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SimpleTraining_Inference", "main_cs", nullptr, nvrhi::ShaderType::Compute);
        m_TrainingPass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SimpleTraining_Training", "main_cs", nullptr, nvrhi::ShaderType::Compute);
        m_OptimizerPass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SimpleTraining_Optimizer", "adam_cs", nullptr, nvrhi::ShaderType::Compute);
        m_ConvertWeightsPass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SimpleTraining_Optimizer", "convert_weights_cs", nullptr, nvrhi::ShaderType::Compute);

        nvrhi::BufferDesc paramsBufferDesc;
        paramsBufferDesc.byteSize = m_NeuralNetwork->GetNetworkParams().size();
        paramsBufferDesc.structStride = sizeof(uint16_t);
        paramsBufferDesc.canHaveUAVs = true;
        paramsBufferDesc.debugName = "MLPParameters";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        paramsBufferDesc.keepInitialState = true;
        paramsBufferDesc.canHaveUAVs = true;
        m_mlpParamsBuffer = GetDevice()->createBuffer(paramsBufferDesc);

        m_TotalParamCount = (uint32_t)(paramsBufferDesc.byteSize / sizeof(uint16_t));

        m_CommandList->beginTrackingBufferState(m_mlpParamsBuffer, nvrhi::ResourceStates::CopyDest);
        m_CommandList->writeBuffer(m_mlpParamsBuffer, m_NeuralNetwork->GetNetworkParams().data(), m_NeuralNetwork->GetNetworkParams().size());

        paramsBufferDesc.debugName = "MLPParametersf";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        paramsBufferDesc.byteSize = m_TotalParamCount * sizeof(float); // convert to float
        paramsBufferDesc.structStride = sizeof(float);
        m_mlpParamsFloatBuffer = GetDevice()->createBuffer(paramsBufferDesc);
        m_CommandList->beginTrackingBufferState(m_mlpParamsFloatBuffer, nvrhi::ResourceStates::UnorderedAccess);
        m_CommandList->clearBufferUInt(m_mlpParamsFloatBuffer, 0);

        paramsBufferDesc.debugName = "MLPGradientsBuffer";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        paramsBufferDesc.byteSize = (m_TotalParamCount * sizeof(uint16_t) + 3) & ~3; // Round up to nearest multiple of 4
        paramsBufferDesc.structStride = sizeof(uint16_t);
        m_mlpGradientsBuffer = GetDevice()->createBuffer(paramsBufferDesc);
        m_CommandList->beginTrackingBufferState(m_mlpGradientsBuffer, nvrhi::ResourceStates::UnorderedAccess);
        m_CommandList->clearBufferUInt(m_mlpGradientsBuffer, 0);

        paramsBufferDesc.debugName = "MLPMoments1Buffer";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        paramsBufferDesc.byteSize = m_TotalParamCount * sizeof(float);
        paramsBufferDesc.structStride = sizeof(float);
        m_mlpMoments1Buffer = GetDevice()->createBuffer(paramsBufferDesc);
        m_CommandList->beginTrackingBufferState(m_mlpMoments1Buffer, nvrhi::ResourceStates::UnorderedAccess);
        m_CommandList->clearBufferUInt(m_mlpMoments1Buffer, 0);

        paramsBufferDesc.debugName = "MLPMoments2Buffer";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        paramsBufferDesc.byteSize = m_TotalParamCount * sizeof(float);
        paramsBufferDesc.structStride = sizeof(float);
        m_mlpMoments2Buffer = GetDevice()->createBuffer(paramsBufferDesc);
        m_CommandList->beginTrackingBufferState(m_mlpMoments2Buffer, nvrhi::ResourceStates::UnorderedAccess);
        m_CommandList->clearBufferUInt(m_mlpMoments2Buffer, 0);

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

        m_CommandList->writeBuffer(m_RandStateBuffer, buff.data(), buff.size() * sizeof(uint32_t));
        m_CommandList->beginTrackingBufferState(m_RandStateBuffer, nvrhi::ResourceStates::UnorderedAccess);

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        GetDevice()->waitForIdle();

        auto texDesc = m_InputTexture->getDesc();
        texDesc.debugName = "InferenceTexture";
        texDesc.format = nvrhi::Format::RGBA16_FLOAT;
        texDesc.isRenderTarget = true;
        texDesc.isUAV = true;
        m_InferenceTexture = GetDevice()->createTexture(texDesc);

        texDesc = m_InputTexture->getDesc();
        texDesc.debugName = "LossTexture";
        texDesc.format = nvrhi::Format::RGBA16_FLOAT;
        texDesc.isRenderTarget = false;
        texDesc.isUAV = true;
        m_LossTexture = GetDevice()->createTexture(texDesc);

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
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_mlpParamsBuffer),
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
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_mlpParamsBuffer),
            nvrhi::BindingSetItem::Texture_SRV(1, m_InputTexture),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_mlpGradientsBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_RandStateBuffer),
            nvrhi::BindingSetItem::Texture_UAV(2, m_InferenceTexture),
            nvrhi::BindingSetItem::Texture_UAV(3, m_LossTexture),
        };
        nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_TrainingPass.m_BindingLayout, m_TrainingPass.m_BindingSet);

        pipelineDesc.bindingLayouts = { m_TrainingPass.m_BindingLayout };
        pipelineDesc.CS = m_TrainingPass.m_ShaderCS;
        m_TrainingPass.m_Pipeline = GetDevice()->createComputePipeline(pipelineDesc);

        // Optimisation Pass
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_NeuralConstantBuffer),       nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_mlpParamsBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_mlpParamsFloatBuffer), nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_mlpGradientsBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(3, m_mlpMoments1Buffer),    nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_mlpMoments2Buffer),
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
    void UpdateNetworkParameters(nvrhi::CommandListHandle commandList)
    {
        commandList->writeBuffer(m_mlpParamsBuffer, m_NeuralNetwork->GetNetworkParams().data(), m_NeuralNetwork->GetNetworkParams().size());
        m_ConvertWeights = true;
    }

    // expects an open command list
    void ResetTrainingData(nvrhi::CommandListHandle commandList)
    {
        // Validate the loaded file matches the shaders
        rtxns::NetworkArchitecture netArch = {};
        netArch.inputNeurons = INPUT_NEURONS;
        netArch.hiddenNeurons = HIDDEN_NEURONS;
        netArch.outputNeurons = OUTPUT_NEURONS;
        netArch.numHiddenLayers = NUM_HIDDEN_LAYERS;
        netArch.biasPrecision = NETWORK_PRECISION;
        netArch.weightPrecision = NETWORK_PRECISION;

        if (!m_NeuralNetwork->Initialise(netArch, rtxns::MatrixLayout::TrainingOptimal))
        {
            log::error("Failed to create a network.");
            return;
        }

        commandList->writeBuffer(m_mlpParamsBuffer, m_NeuralNetwork->GetNetworkParams().data(), m_NeuralNetwork->GetNetworkParams().size());
        commandList->clearBufferUInt(m_mlpParamsFloatBuffer, 0);
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
                rtxns::Network network(GetDevice());
                if (network.InitialiseFromFile(m_uiParams->fileName))
                {
                    // Validate the loaded file against what the shaders expect
                    if ((network.GetNetworkArchitecture().inputNeurons == INPUT_NEURONS) && (network.GetNetworkArchitecture().outputNeurons == OUTPUT_NEURONS) &&
                        (network.GetNetworkArchitecture().hiddenNeurons == HIDDEN_NEURONS) && (network.GetNetworkArchitecture().numHiddenLayers == NUM_HIDDEN_LAYERS) &&
                        (network.GetNetworkArchitecture().biasPrecision == NETWORK_PRECISION) && (network.GetNetworkArchitecture().weightPrecision == NETWORK_PRECISION))
                    {
                        m_CommandList = GetDevice()->createCommandList();
                        m_CommandList->open();

                        ResetTrainingData(m_CommandList);

                        m_NeuralNetwork->InitialiseFromNetwork(network, rtxns::MatrixLayout::TrainingOptimal);

                        UpdateNetworkParameters(m_CommandList);

                        m_CommandList->close();
                        GetDevice()->executeCommandList(m_CommandList);
                    }
                }
            }
            else
            {
                m_NeuralNetwork->UpdateFromBufferToFile(m_mlpParamsBuffer, m_uiParams->fileName);
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

        m_CommandList->open();

        if (m_uiParams->reset)
        {
            ResetTrainingData(m_CommandList);
            m_uiParams->reset = false;
        }

        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));

        ////////////////////
        //
        // Update the Constant buffer
        //
        ////////////////////
        NeuralConstants neuralConstants = {};

        for (int i = 0; i < NUM_TRANSITIONS; ++i)
        {
            neuralConstants.weightOffsets[i / 4][i % 4] = m_NeuralNetwork->GetNetworkLayers()[i].weightOffset;
            neuralConstants.biasOffsets[i / 4][i % 4] = m_NeuralNetwork->GetNetworkLayers()[i].biasOffset;
        }

        neuralConstants.imageWidth = m_InferenceTexture->getDesc().width;
        neuralConstants.imageHeight = m_InferenceTexture->getDesc().height;
        neuralConstants.maxParamSize = m_TotalParamCount;
        neuralConstants.learningRate = m_LearningRate;
        neuralConstants.currentStep = m_AdamCurrentStep;
        neuralConstants.batchSizeX = BATCH_SIZE_X;
        neuralConstants.batchSizeY = BATCH_SIZE_Y;
        neuralConstants.networkTransform = m_uiParams->networkTransform;
        m_CommandList->writeBuffer(m_NeuralConstantBuffer, &neuralConstants, sizeof(neuralConstants));

        nvrhi::ComputeState state;

        ////////////////////
        //
        // Start the training loop
        //
        ////////////////////
        if (m_uiParams->training)
        {
            if (m_ConvertWeights)
            {
                state.bindings = { m_ConvertWeightsPass.m_BindingSet };
                state.pipeline = m_ConvertWeightsPass.m_Pipeline;
                m_CommandList->beginMarker("ConvertWeights");
                m_CommandList->setComputeState(state);
                m_CommandList->dispatch(dm::div_ceil(m_TotalParamCount, 32), 1, 1);
                m_CommandList->endMarker();
                m_ConvertWeights = false;
            }

            for (uint32_t batch = 0; batch < BATCH_COUNT; batch++)
            {
                // run the training pass
                state.bindings = { m_TrainingPass.m_BindingSet };
                state.pipeline = m_TrainingPass.m_Pipeline;
                m_CommandList->beginMarker("Training");
                m_CommandList->setComputeState(state);
                m_CommandList->dispatch(dm::div_ceil(BATCH_SIZE_X, 8), dm::div_ceil(BATCH_SIZE_Y, 8), 1);
                m_CommandList->endMarker();

                // optimizer pass
                state.bindings = { m_OptimizerPass.m_BindingSet };
                state.pipeline = m_OptimizerPass.m_Pipeline;
                m_CommandList->beginMarker("Update Weights");
                m_CommandList->setComputeState(state);
                m_CommandList->dispatch(dm::div_ceil(m_TotalParamCount, 32), 1, 1);
                m_CommandList->endMarker();

                neuralConstants.currentStep = ++m_AdamCurrentStep;
                m_CommandList->writeBuffer(m_NeuralConstantBuffer, &neuralConstants, sizeof(neuralConstants));
            }
            m_uiParams->epochs++;
        }

        {
            // inference pass
            state.bindings = { m_InferencePass.m_BindingSet };
            state.pipeline = m_InferencePass.m_Pipeline;
            m_CommandList->beginMarker("Inference");
            m_CommandList->setComputeState(state);
            m_CommandList->dispatch(dm::div_ceil(m_InferenceTexture->getDesc().width, 8), dm::div_ceil(m_InferenceTexture->getDesc().height, 8), 1);
            m_CommandList->endMarker();
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
                m_CommonPasses->BlitTexture(m_CommandList, blitParams, m_BindingCache.get());
            }
            else if (viewIndex == 1)
            {
                // Draw inferenced image
                engine::BlitParameters blitParams;
                blitParams.targetFramebuffer = framebuffer;
                blitParams.targetViewport = viewport;
                blitParams.sourceTexture = m_InferenceTexture;
                m_CommonPasses->BlitTexture(m_CommandList, blitParams, m_BindingCache.get());
            }
            else if (viewIndex == 2)
            {
                // Draw loss image
                engine::BlitParameters blitParams;
                blitParams.targetFramebuffer = framebuffer;
                blitParams.targetViewport = viewport;
                blitParams.sourceTexture = m_LossTexture;
                m_CommonPasses->BlitTexture(m_CommandList, blitParams, m_BindingCache.get());
            }
        }

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
    }

private:
    std::shared_ptr<vfs::RootFileSystem> m_RootFS;

    nvrhi::CommandListHandle m_CommandList;

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

    nvrhi::BufferHandle m_mlpParamsBuffer;
    nvrhi::BufferHandle m_mlpParamsFloatBuffer;
    nvrhi::BufferHandle m_mlpGradientsBuffer;
    nvrhi::BufferHandle m_mlpMoments1Buffer;
    nvrhi::BufferHandle m_mlpMoments2Buffer;
    nvrhi::BufferHandle m_RandStateBuffer;

    nvrhi::FramebufferHandle m_Framebuffer;

    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::unique_ptr<engine::Scene> m_Scene;
    std::shared_ptr<engine::DescriptorTableManager> m_DescriptorTableManager;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    std::unique_ptr<rtxns::Network> m_NeuralNetwork;

    uint m_TotalParamCount = 0;
    uint m_AdamCurrentStep = 1;
    float m_LearningRate = LEARNING_RATE;
    bool m_ConvertWeights = true;

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
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI graphicsApi = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    if (graphicsApi == nvrhi::GraphicsAPI::D3D11 || graphicsApi == nvrhi::GraphicsAPI::D3D12)
    {
        log::error("This sample does not support D3D11 or D3D12.");
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
    SetCoopVectorExtensionParameters(deviceParams, graphicsApi, false);

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters. Please try a NVIDIA driver version greater than 570");
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
