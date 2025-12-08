/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/app/ApplicationBase.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/UserInterfaceUtils.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/json.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/BindingCache.h>
#include <nvrhi/utils.h>

#include "DeviceUtils.h"
#include "GraphicsResources.h"
#include "GeometryUtils.h"
#include "NeuralNetwork.h"
#include "Float16.h"

#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <format>

using namespace donut;
using namespace donut::math;

#include "NetworkConfig.h"

static const char* g_windowTitle = "RTX Neural Shading Example: Shader Training (Ground Truth | Training | Loss )";
constexpr int g_viewsNum = 3;
constexpr int g_statisticsPerFrames = 100;

static std::random_device rd;

struct UIData
{
    float lightIntensity = 1.f;
    float specular = 0.5f;
    float roughness = 0.4f;
    float metallic = 0.7f;

    float trainingTime = 0.0f;
    uint32_t epochs = 0;

    bool reset = false;
    bool training = true;
    bool load = false;
    std::string fileName;
};

class SimpleShading : public app::IRenderPass
{

public:
    SimpleShading(app::DeviceManager* deviceManager, UIData* uiParams) : IRenderPass(deviceManager), m_userInterfaceParameters(uiParams)
    {
    }

    bool Init()
    {
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();

        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/ShaderTraining" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app", appShaderPath);

        m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_commonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);
        m_bindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        ////////////////////
        //
        // Create the Neural network class and initialize it the hyper parameters from NetworkConfig.h.
        //
        ////////////////////
        m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
        m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
        if (!m_neuralNetwork->Initialise(m_netArch))
        {
            log::error("Failed to create a network.");
            return false;
        }

        ////////////////////
        //
        // Create the shaders/buffers for the Neural Training
        //
        ////////////////////
        m_trainingPass.computeShader = m_shaderFactory->CreateShader("app/computeTraining", "main_cs", nullptr, nvrhi::ShaderType::Compute);
        m_optimizerPass.computeShader = m_shaderFactory->CreateShader("app/computeOptimizer", "adam_cs", nullptr, nvrhi::ShaderType::Compute);

        m_trainingConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(TrainingConstantBufferEntry), "TrainingConstantBuffer")
                                                                 .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                 .setKeepInitialState(true));

        ////////////////////
        //
        // Continue to load the render data and create the required structures
        //
        ////////////////////
        auto [vertices, indices] = GenerateSphere(1, 64, 64);
        m_indicesNum = (int)indices.size();

        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(1).setElementStride(sizeof(Vertex)),
        };

        // Initialize direct pass
        {
            m_directPass.constantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(DirectConstantBufferEntry), "DirectConstantBuffer")
                                                                        .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                        .setKeepInitialState(true));
            m_directPass.vertexShader = m_shaderFactory->CreateShader("app/renderDisney", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_directPass.pixelShader = m_shaderFactory->CreateShader("app/renderDisney", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            assert(m_directPass.vertexShader && m_directPass.pixelShader);

            m_directPass.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_directPass.vertexShader);
        }

        // Initialize neural pass
        {
            m_inferencePass.constantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(InferenceConstantBufferEntry), "NeuralConstantBuffer")
                                                                           .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                           .setKeepInitialState(true));
            m_inferencePass.vertexShader = m_shaderFactory->CreateShader("app/renderInference", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_inferencePass.pixelShader = m_shaderFactory->CreateShader("app/renderInference", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            assert(m_inferencePass.vertexShader && m_inferencePass.pixelShader);

            m_inferencePass.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_inferencePass.vertexShader);
        }

        // Initialize difference pass
        {
            m_differencePass.constantBuffer = m_inferencePass.constantBuffer;
            m_differencePass.vertexShader = m_shaderFactory->CreateShader("app/renderDifference", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_differencePass.pixelShader = m_shaderFactory->CreateShader("app/renderDifference", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            assert(m_differencePass.vertexShader && m_differencePass.pixelShader);

            m_differencePass.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_differencePass.vertexShader);
        }

        // Create and fill render buffers
        {
            m_commandList = GetDevice()->createCommandList();
            m_commandList->open();

            nvrhi::BufferDesc vertexBufferDesc;
            vertexBufferDesc.byteSize = vertices.size() * sizeof(vertices[0]);
            vertexBufferDesc.isVertexBuffer = true;
            vertexBufferDesc.debugName = "VertexBuffer";
            vertexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            m_vertexBuffer = GetDevice()->createBuffer(vertexBufferDesc);

            m_commandList->beginTrackingBufferState(m_vertexBuffer, nvrhi::ResourceStates::CopyDest);
            m_commandList->writeBuffer(m_vertexBuffer, vertices.data(), vertices.size() * sizeof(vertices[0]));
            m_commandList->setPermanentBufferState(m_vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

            nvrhi::BufferDesc indexBufferDesc;
            indexBufferDesc.byteSize = indices.size() * sizeof(indices[0]);
            indexBufferDesc.isIndexBuffer = true;
            indexBufferDesc.debugName = "IndexBuffer";
            indexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            m_indexBuffer = GetDevice()->createBuffer(indexBufferDesc);

            m_commandList->beginTrackingBufferState(m_indexBuffer, nvrhi::ResourceStates::CopyDest);
            m_commandList->writeBuffer(m_indexBuffer, indices.data(), indices.size() * sizeof(indices[0]));
            m_commandList->setPermanentBufferState(m_indexBuffer, nvrhi::ResourceStates::IndexBuffer);

            m_commandList->close();
            GetDevice()->executeCommandList(m_commandList);
        }

        // Direct binding
        {
            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_directPass.constantBuffer) };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_directPass.bindingLayout, m_directPass.bindingSet);
        }

        CreateMLPBuffers();

        m_disneyTimer = GetDevice()->createTimerQuery();
        m_neuralTimer = GetDevice()->createTimerQuery();
        m_trainingTimer = GetDevice()->createTimerQuery();
        m_optimizerTimer = GetDevice()->createTimerQuery();

        return true;
    }

    void CreateMLPBuffers()
    {
        const auto& params = m_neuralNetwork->GetNetworkParams();

        for (int i = 0; i < NUM_TRANSITIONS; ++i)
        {
            m_weightOffsets[i / 4][i % 4] = m_neuralNetwork->GetNetworkLayout().networkLayers[i].weightOffset;
            m_biasOffsets[i / 4][i % 4] = m_neuralNetwork->GetNetworkLayout().networkLayers[i].biasOffset;
        }

        // Get a device optimized layout
        m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

        m_totalParameterCount = uint(params.size() / sizeof(uint16_t));
        m_batchSize = BATCH_SIZE;

        // Create and fill buffers
        {
            m_commandList = GetDevice()->createCommandList();
            m_commandList->open();

            nvrhi::BufferDesc paramsBufferDesc;

            paramsBufferDesc.debugName = "MLPParamsDeviceBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            paramsBufferDesc.byteSize = params.size();
            paramsBufferDesc.keepInitialState = true;
            paramsBufferDesc.canHaveUAVs = true;
            m_mlpHostBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            paramsBufferDesc.debugName = "MLPParamsDeviceBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = m_deviceNetworkLayout.networkSize;
            paramsBufferDesc.canHaveRawViews = true;
            paramsBufferDesc.canHaveTypedViews = true;
            paramsBufferDesc.canHaveUAVs = true;
            paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
            m_mlpDeviceBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            // Upload the parameters
            UpdateDeviceNetworkParameters(m_commandList);

            paramsBufferDesc.debugName = "MLPParamsBuffer32";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            paramsBufferDesc.byteSize = m_totalParameterCount * sizeof(float);
            paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
            m_mlpParamsBuffer32 = GetDevice()->createBuffer(paramsBufferDesc);

            m_commandList->beginTrackingBufferState(m_mlpParamsBuffer32, nvrhi::ResourceStates::CopyDest);
            {
                std::vector<float> fbuf(m_totalParameterCount);
                std::transform((uint16_t*)params.data(), ((uint16_t*)params.data()) + m_totalParameterCount, fbuf.begin(), [](auto v) { return rtxns::float16ToFloat32(v); });
                m_commandList->writeBuffer(m_mlpParamsBuffer32, fbuf.data(), paramsBufferDesc.byteSize);
            }

            paramsBufferDesc.debugName = "MLPGradientsBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = (m_totalParameterCount * sizeof(uint16_t) + 3) & ~3; // Round up to nearest multiple of 4
            paramsBufferDesc.structStride = sizeof(uint16_t);
            paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
            m_mlpGradientsBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            m_commandList->beginTrackingBufferState(m_mlpGradientsBuffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->clearBufferUInt(m_mlpGradientsBuffer, 0);

            paramsBufferDesc.debugName = "MLPMoments1Buffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = m_totalParameterCount * sizeof(float);
            paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
            paramsBufferDesc.canHaveRawViews = false;
            m_mlpMoments1Buffer = GetDevice()->createBuffer(paramsBufferDesc);

            m_commandList->beginTrackingBufferState(m_mlpMoments1Buffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->clearBufferUInt(m_mlpMoments1Buffer, 0);

            paramsBufferDesc.debugName = "MLPMoments2Buffer";
            m_mlpMoments2Buffer = GetDevice()->createBuffer(paramsBufferDesc);

            m_commandList->beginTrackingBufferState(m_mlpMoments2Buffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->clearBufferUInt(m_mlpMoments2Buffer, 0);

            m_commandList->close();
            GetDevice()->executeCommandList(m_commandList);
        }

        nvrhi::BindingSetDesc bindingSetDesc = {};
        // Training binding
        {
            m_trainingPass.bindingSet = nullptr;
            m_trainingPass.bindingLayout = nullptr;

            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingConstantBuffer),
                nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::RawBuffer_UAV(0, m_mlpGradientsBuffer),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_trainingPass.bindingLayout, m_trainingPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_trainingPass.bindingLayout };
            pipelineDesc.CS = m_trainingPass.computeShader;
            m_trainingPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Optimization binding
        {
            m_optimizerPass.bindingSet = nullptr;
            m_optimizerPass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingConstantBuffer), nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_mlpParamsBuffer32),     nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_mlpGradientsBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(3, m_mlpMoments1Buffer),     nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_mlpMoments2Buffer),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_optimizerPass.bindingLayout, m_optimizerPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_optimizerPass.bindingLayout };
            pipelineDesc.CS = m_optimizerPass.computeShader;
            m_optimizerPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Inference binding
        {
            m_inferencePass.pipeline = nullptr;
            m_inferencePass.bindingSet = nullptr;
            m_inferencePass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_inferencePass.constantBuffer), nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer) };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_inferencePass.bindingLayout, m_inferencePass.bindingSet);
        }

        // Difference binding
        {
            m_differencePass.pipeline = nullptr;
            m_differencePass.bindingSet = nullptr;
            m_differencePass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_differencePass.constantBuffer), nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer) };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_differencePass.bindingLayout, m_differencePass.bindingSet);
        }

        // Reset training parameters
        m_currentOptimizationStep = 0;
        m_userInterfaceParameters->epochs = 0;
        m_userInterfaceParameters->trainingTime = 0.0f;
    }

    // expects an open command list
    void UpdateDeviceNetworkParameters(nvrhi::CommandListHandle commandList)
    {
        // Upload the host side parameters
        commandList->setBufferState(m_mlpHostBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->commitBarriers();
        commandList->writeBuffer(m_mlpHostBuffer, m_neuralNetwork->GetNetworkParams().data(), m_neuralNetwork->GetNetworkParams().size());

        // Convert to GPU optimized layout
        m_networkUtils->ConvertWeights(m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

        // Update barriers for use
        commandList->setBufferState(m_mlpDeviceBuffer, nvrhi::ResourceStates::ShaderResource);
        commandList->commitBarriers();
        // m_convertWeights = true;
    }


    std::shared_ptr<engine::ShaderFactory> GetShaderFactory() const
    {
        return m_shaderFactory;
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        if (m_pressedFlag)
        {
            float2 delta = float2(float(xpos), float(ypos)) - m_currentXY;
            float a, e, d;
            cartesianToSpherical(m_lightDir, a, e, d);
            a += delta.x * 0.01f;
            e += delta.y * 0.01f;
            m_lightDir = sphericalToCartesian(a, e, d);
        }

        m_currentXY = float2(float(xpos), float(ypos));
        return true;
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        m_pressedFlag = action == 1;
        return true;
    }

    void Animate(float seconds) override
    {
        if (m_userInterfaceParameters->training)
        {
            m_userInterfaceParameters->trainingTime += seconds;
        }

        auto toMicroSeconds = [&](const auto& timer) { return int(GetDevice()->getTimerQueryTime(timer) * 1000000); };

        auto t = toMicroSeconds(m_disneyTimer);
        if (t != 0)
        {
            m_extraStatus = std::format(" - Disney - {:3d}us, Neural - {:3d}us, Training - {:3d}us, Optimization - {:3d}us", t, toMicroSeconds(m_neuralTimer),
                                        toMicroSeconds(m_trainingTimer), toMicroSeconds(m_optimizerTimer));
        }
        GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true, m_extraStatus.c_str());

        ////////////////////
        //
        // Reset/Load/Save the Neural network if required
        //
        ////////////////////
        if (m_userInterfaceParameters->reset)
        {
            m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
            if (m_neuralNetwork->Initialise(m_netArch))
            {
                CreateMLPBuffers();
            }
            else
            {
                log::error("Failed to create a network.");
            }

            m_userInterfaceParameters->reset = false;
        }

        if (!m_userInterfaceParameters->fileName.empty())
        {
            if (m_userInterfaceParameters->load)
            {
                m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
                m_neuralNetwork->InitialiseFromFile(m_userInterfaceParameters->fileName);
                CreateMLPBuffers();
            }
            else
            {
                m_neuralNetwork->UpdateFromBufferToFile(m_mlpHostBuffer, m_mlpDeviceBuffer, m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout,
                                                        m_userInterfaceParameters->fileName, GetDevice(), m_commandList);
            }
            m_userInterfaceParameters->fileName = "";
        }
    }

    void BackBufferResizing() override
    {
        m_directPass.pipeline = nullptr;
        m_inferencePass.pipeline = nullptr;
        m_differencePass.pipeline = nullptr;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        std::uniform_int_distribution<uint64_t> ldist;
        uint64_t seed = ldist(rd);

        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
        const float height = float(fbinfo.height);
        const float width = height;

        // Update statistics every g_statisticsPerFrames frames
        bool updateStat = GetDeviceManager()->GetCurrentBackBufferIndex() % g_statisticsPerFrames == 0;

        // Camera at (0,0,2) looking at (0,0,-1) direction, up direction (0,1,0)
        float3 cameraUp(0, 1, 0);
        float4 viewDir(0, 0, -1, 0);

        // Fill out the constant buffer slices for multiple views of the model.
        DirectConstantBufferEntry directModelConstant{ {},
                                                       {},
                                                       { 0, 0, 2, 0 },
                                                       float4(m_lightDir, 1.f),
                                                       float4(m_userInterfaceParameters->lightIntensity),
                                                       float4(.82f, .67f, .16f, 1.f),
                                                       m_userInterfaceParameters->specular,
                                                       m_userInterfaceParameters->roughness,
                                                       m_userInterfaceParameters->metallic };
        directModelConstant.view = affineToHomogeneous(translation(-directModelConstant.cameraPos.xyz()) * lookatZ(-viewDir.xyz(), cameraUp));
        directModelConstant.viewProject = directModelConstant.view * perspProjD3DStyle(radians(67.4f), float(width) / float(height), 0.1f, 10.f);

        ////////////////////
        //
        // Fill out the inference constant buffer including the neural weight/bias offsets.
        //
        ////////////////////
        InferenceConstantBufferEntry inferenceModelConstant;
        static_cast<DirectConstantBufferEntry&>(inferenceModelConstant) = directModelConstant;
        std::ranges::copy(m_weightOffsets, inferenceModelConstant.weightOffsets);
        std::ranges::copy(m_biasOffsets, inferenceModelConstant.biasOffsets);

        m_commandList->open();

        ////////////////////
        //
        // Start the training loop
        //
        ////////////////////
        if (m_userInterfaceParameters->training)
        {
            for (int i = 0; i < BATCH_COUNT; ++i)
            {
                TrainingConstantBufferEntry trainingModelConstant = {
                    .maxParamSize = m_totalParameterCount, .learningRate = m_learningRate, .currentStep = float(++m_currentOptimizationStep), .batchSize = m_batchSize, .seed = seed
                };
                std::ranges::copy(m_weightOffsets, trainingModelConstant.weightOffsets);
                std::ranges::copy(m_biasOffsets, trainingModelConstant.biasOffsets);

                m_commandList->writeBuffer(m_trainingConstantBuffer, &trainingModelConstant, sizeof(trainingModelConstant));

                nvrhi::ComputeState state;

                // Training pass
                state.bindings = { m_trainingPass.bindingSet };
                state.pipeline = m_trainingPass.pipeline;
                m_commandList->beginMarker("Training");

                if (updateStat && i == 0)
                {
                    GetDevice()->resetTimerQuery(m_trainingTimer);
                    m_commandList->beginTimerQuery(m_trainingTimer);
                }

                m_commandList->setComputeState(state);
                m_commandList->dispatch(m_batchSize / 64, 1, 1);

                if (updateStat && i == 0)
                {
                    m_commandList->endTimerQuery(m_trainingTimer);
                }
                m_commandList->endMarker();

                // Optimizer pass
                state.bindings = { m_optimizerPass.bindingSet };
                state.pipeline = m_optimizerPass.pipeline;
                m_commandList->beginMarker("Update Weights");

                if (updateStat && i == 0)
                {
                    GetDevice()->resetTimerQuery(m_optimizerTimer);
                    m_commandList->beginTimerQuery(m_optimizerTimer);
                }

                m_commandList->setComputeState(state);
                m_commandList->dispatch(div_ceil(m_totalParameterCount, 32), 1, 1);

                if (updateStat && i == 0)
                {
                    m_commandList->endTimerQuery(m_optimizerTimer);
                }
                m_commandList->endMarker();
            }

            ++m_userInterfaceParameters->epochs;
        }

        nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.f));

        RenderPass* passes[] = { &m_directPass, &m_inferencePass, &m_differencePass };
        for (int viewIndex = 0; viewIndex < g_viewsNum; ++viewIndex)
        {
            nvrhi::TimerQueryHandle timer;
            if (viewIndex < 2 && updateStat)
            {
                timer = viewIndex == 0 ? m_disneyTimer.Get() : m_neuralTimer.Get();
                GetDevice()->resetTimerQuery(timer);
                m_commandList->beginTimerQuery(timer);
            }

            auto& pass = *passes[viewIndex];

            if (!pass.pipeline)
            {
                nvrhi::GraphicsPipelineDesc psoDesc;
                psoDesc.VS = pass.vertexShader;
                psoDesc.PS = pass.pixelShader;
                psoDesc.inputLayout = pass.inputLayout;
                psoDesc.bindingLayouts = { pass.bindingLayout };
                psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
                psoDesc.renderState.depthStencilState.depthTestEnable = false;

                pass.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
            }

            if (viewIndex == 0)
            {
                m_commandList->writeBuffer(pass.constantBuffer, &directModelConstant, sizeof(directModelConstant));
            }
            else
            {
                m_commandList->writeBuffer(pass.constantBuffer, &inferenceModelConstant, sizeof(inferenceModelConstant));
            }

            nvrhi::GraphicsState state;
            state.bindings = { pass.bindingSet };
            state.indexBuffer = { m_indexBuffer, nvrhi::Format::R32_UINT, 0 };

            state.vertexBuffers = {
                { m_vertexBuffer, 0, offsetof(Vertex, position) },
                { m_vertexBuffer, 1, offsetof(Vertex, normal) },
            };
            state.pipeline = pass.pipeline;
            state.framebuffer = framebuffer;

            // Construct the viewport so that all viewports form a grid.
            const float left = width * viewIndex;
            const float top = 0;

            const nvrhi::Viewport viewport = nvrhi::Viewport(left, left + width, 0, height, 0.f, 1.f);
            state.viewport.addViewportAndScissorRect(viewport);

            // Update the pipeline, bindings, and other state.
            m_commandList->setGraphicsState(state);

            // Draw the model.
            nvrhi::DrawArguments args;
            args.vertexCount = m_indicesNum;
            m_commandList->drawIndexed(args);

            if (viewIndex < 2 && updateStat)
            {
                m_commandList->endTimerQuery(timer);
            }
        }

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
    }

private:
    std::string m_extraStatus;
    nvrhi::TimerQueryHandle m_disneyTimer;
    nvrhi::TimerQueryHandle m_neuralTimer;
    nvrhi::TimerQueryHandle m_trainingTimer;
    nvrhi::TimerQueryHandle m_optimizerTimer;

    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<engine::CommonRenderPasses> m_commonPasses;
    std::unique_ptr<engine::BindingCache> m_bindingCache;

    struct RenderPass
    {
        nvrhi::ShaderHandle vertexShader;
        nvrhi::ShaderHandle pixelShader;
        nvrhi::BufferHandle constantBuffer;
        nvrhi::InputLayoutHandle inputLayout;
        nvrhi::BindingLayoutHandle bindingLayout;
        nvrhi::BindingSetHandle bindingSet;
        nvrhi::GraphicsPipelineHandle pipeline;
    };

    RenderPass m_directPass;
    RenderPass m_inferencePass;
    RenderPass m_differencePass;

    float3 m_lightDir{ -0.761f, -0.467f, -0.450f };
    float2 m_currentXY;
    bool m_pressedFlag = false;

    nvrhi::BufferHandle m_vertexBuffer;
    nvrhi::BufferHandle m_indexBuffer;

    nvrhi::BufferHandle m_trainingConstantBuffer;
    nvrhi::BufferHandle m_mlpHostBuffer;
    nvrhi::BufferHandle m_mlpDeviceBuffer;
    nvrhi::BufferHandle m_mlpParamsBuffer32;
    nvrhi::BufferHandle m_mlpGradientsBuffer;
    nvrhi::BufferHandle m_mlpMoments1Buffer;
    nvrhi::BufferHandle m_mlpMoments2Buffer;

    uint m_totalParameterCount = 0;
    uint m_batchSize = BATCH_SIZE;
    uint m_currentOptimizationStep = 0;
    float m_learningRate = LEARNING_RATE;

    nvrhi::CommandListHandle m_commandList;

    int m_indicesNum = 0;

    struct NeuralPass
    {
        nvrhi::ShaderHandle computeShader;
        nvrhi::BindingLayoutHandle bindingLayout;
        nvrhi::BindingSetHandle bindingSet;
        nvrhi::ComputePipelineHandle pipeline;
    };

    NeuralPass m_trainingPass;
    NeuralPass m_optimizerPass;

    uint4 m_weightOffsets[NUM_TRANSITIONS_ALIGN4];
    uint4 m_biasOffsets[NUM_TRANSITIONS_ALIGN4];

    UIData* m_userInterfaceParameters;

    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;
    std::unique_ptr<rtxns::HostNetwork> m_neuralNetwork;
    rtxns::NetworkLayout m_deviceNetworkLayout;

    rtxns::NetworkArchitecture m_netArch = {
        .numHiddenLayers = NUM_HIDDEN_LAYERS,
        .inputNeurons = INPUT_NEURONS,
        .hiddenNeurons = HIDDEN_NEURONS,
        .outputNeurons = OUTPUT_NEURONS,
        .weightPrecision = rtxns::Precision::F16,
        .biasPrecision = rtxns::Precision::F16,
    };
};

class UserInterface : public app::ImGui_Renderer
{
public:
    UserInterface(app::DeviceManager* deviceManager, UIData* ui) : ImGui_Renderer(deviceManager), m_userInterfaceParameters(ui)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void buildUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);
        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::SliderFloat("Light Intensity", &m_userInterfaceParameters->lightIntensity, 0.f, 20.f);
        ImGui::SliderFloat("Specular", &m_userInterfaceParameters->specular, 0.f, 1.f);
        ImGui::SliderFloat("Roughness", &m_userInterfaceParameters->roughness, 0.3f, 1.f);
        ImGui::SliderFloat("Metallic", &m_userInterfaceParameters->metallic, 0.f, 1.f);

        ImGui::Text("Epochs : %d", m_userInterfaceParameters->epochs);
        ImGui::Text("Training Time : %.2f s", m_userInterfaceParameters->trainingTime);

        if (ImGui::Button(m_userInterfaceParameters->training ? "Disable Training" : "Enable Training"))
        {
            m_userInterfaceParameters->training = !m_userInterfaceParameters->training;
        }

        if (ImGui::Button("Reset Training"))
        {
            m_userInterfaceParameters->reset = true;
        }

        if (ImGui::Button("Load Model"))
        {
            std::string fileName;
            if (app::FileDialog(true, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
            {
                m_userInterfaceParameters->fileName = fileName;
                m_userInterfaceParameters->load = true;
            }
        }

        if (ImGui::Button("Save Model"))
        {
            std::string fileName;
            if (app::FileDialog(false, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
            {
                m_userInterfaceParameters->fileName = fileName;
                m_userInterfaceParameters->load = false;
            }
        }

        ImGui::End();
    }

private:
    UIData* m_userInterfaceParameters;
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
    std::unique_ptr<app::DeviceManager> deviceManager(app::DeviceManager::Create(graphicsApi));

    app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = deviceParams.backBufferHeight * g_viewsNum;

#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    ////////////////////
    //
    // Setup the CoopVector extensions.
    //
    ////////////////////
    SetCoopVectorExtensionParameters(deviceParams, graphicsApi, true, g_windowTitle);

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle))
    {
        if (graphicsApi == nvrhi::GraphicsAPI::VULKAN)
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters. Please try a NVIDIA driver version greater than 570");
        }
        if (graphicsApi == nvrhi::GraphicsAPI::D3D12)
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters. Please use the Shader Model 6-9-Preview Driver, link in the README");
        }
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
        SimpleShading example(deviceManager.get(), &uiData);
        UserInterface gui(deviceManager.get(), &uiData);

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

    return 0;
}
