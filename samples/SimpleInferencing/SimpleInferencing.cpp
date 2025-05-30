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
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>

#include "DeviceUtils.h"
#include "GraphicsResources.h"
#include "GeometryUtils.h"
#include "NeuralNetwork.h"
#include "DirectoryHelper.h"

#include <iostream>
#include <fstream>
#include <format>

using namespace donut;
using namespace donut::math;

#include "NetworkConfig.h"

static const char* g_windowTitle = "RTX Neural Shading Example: Simple Inferencing";

struct UIData
{
    float lightIntensity = 1.f;
    float specular = 0.5f;
    float roughness = 0.4f;
    float metallic = 0.7f;
};

class SimpleInferencing : public app::IRenderPass
{
public:
    SimpleInferencing(app::DeviceManager* deviceManager, UIData* uiParams) : IRenderPass(deviceManager), m_userInterfaceParameters(uiParams)
    {
    }

    bool Init()
    {
        ////////////////////
        //
        // Create the Neural network class and initialise it from a file.
        //
        ////////////////////
        m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
        rtxns::HostNetwork net(m_networkUtils);
        if (!net.InitialiseFromFile(GetLocalPath("assets/data").string() + std::string("/disney.ns.bin")))
        {
            log::debug("Loaded Neural Shading Network from file failed.");
            return false;
        }

        // We are expecting 4 layers, validate
        assert(net.GetNetworkLayout().networkLayers.size() == 4);

        // Get a device optimized layout
        rtxns::NetworkLayout deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(net.GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

        // Store the weight and bias offsets into a uint4.
        m_weightOffsets = dm::uint4(deviceNetworkLayout.networkLayers[0].weightOffset, deviceNetworkLayout.networkLayers[1].weightOffset,
                                    deviceNetworkLayout.networkLayers[2].weightOffset, deviceNetworkLayout.networkLayers[3].weightOffset);

        m_biasOffsets = dm::uint4(deviceNetworkLayout.networkLayers[0].biasOffset, deviceNetworkLayout.networkLayers[1].biasOffset, deviceNetworkLayout.networkLayers[2].biasOffset,
                                  deviceNetworkLayout.networkLayers[3].biasOffset);

        ////////////////////
        //
        // Continue to load the render data and create the required structures
        //
        ////////////////////
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/SimpleInferencing" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app", appShaderPath);

        m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_vertexShader = m_shaderFactory->CreateShader("app/SimpleInferencing", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_pixelShader = m_shaderFactory->CreateShader("app/SimpleInferencing", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

        if (!m_vertexShader || !m_pixelShader)
        {
            return false;
        }

        auto [vertices, indices] = GenerateSphere(1, 64, 64);
        m_indicesNum = (int)indices.size();

        m_constantBuffer = GetDevice()->createBuffer(
            nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(NeuralConstants), "ConstantBuffer").setInitialState(nvrhi::ResourceStates::ConstantBuffer).setKeepInitialState(true));

        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(1).setElementStride(sizeof(Vertex)),
        };
        m_inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_vertexShader);

        engine::CommonRenderPasses commonPasses(GetDevice(), m_shaderFactory);

        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        engine::TextureCache textureCache(GetDevice(), nativeFS, nullptr);

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

        ////////////////////
        //
        // Create buffers for storing the neural parameters/weights and biases
        //
        ////////////////////
        const auto& params = net.GetNetworkParams();

        // Create a buffer for the host side weight and bias parameters
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = params.size();
        bufferDesc.debugName = "MLPParamsUploadBuffer";
        bufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        bufferDesc.keepInitialState = true;
        m_mlpHostBuffer = GetDevice()->createBuffer(bufferDesc);

        // Create a buffer for a device optimized parameters layout
        bufferDesc.byteSize = deviceNetworkLayout.networkSize;
        bufferDesc.canHaveRawViews = true;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.debugName = "MLPParamsByteAddressBuffer";
        bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        m_mlpDeviceBuffer = GetDevice()->createBuffer(bufferDesc);

        // Upload the parameters
        m_commandList->writeBuffer(m_mlpHostBuffer, params.data(), params.size());

        // Convert to GPU optimized layout
        m_networkUtils->ConvertWeights(net.GetNetworkLayout(), deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

        m_commandList->setBufferState(m_mlpDeviceBuffer, nvrhi::ResourceStates::ShaderResource);
        m_commandList->commitBarriers();

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);

        ////////////////////
        //
        // Create the binding set
        //
        ////////////////////
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = { // Note: using viewIndex to construct a buffer range.
                                    nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
                                    // Parameters buffer
                                    nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer)
        };

        // Create the binding layout (if it's empty -- so, on the first iteration) and the binding set.
        if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_bindingLayout, m_bindingSet))
        {
            log::error("Couldn't create the binding set or layout");
            return false;
        }

        m_neuralTimer = GetDevice()->createTimerQuery();

        return true;
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
        auto t = int(GetDevice()->getTimerQueryTime(m_neuralTimer) * 1000000);
        if (t != 0)
        {
            m_extraStatus = std::format(" - Neural - {:3d}us", t);
        }

        GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true, m_extraStatus.c_str());
    }

    void BackBufferResizing() override
    {
        m_pipeline = nullptr;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
        const float width = float(fbinfo.width);
        const float height = float(fbinfo.height);
        const float left = 0;
        const float top = 0;

        bool updateStat = GetDeviceManager()->GetCurrentBackBufferIndex() % 100 == 0;

        if (!m_pipeline)
        {
            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_vertexShader;
            psoDesc.PS = m_pixelShader;
            psoDesc.inputLayout = m_inputLayout;
            psoDesc.bindingLayouts = { m_bindingLayout };
            psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
            psoDesc.renderState.depthStencilState.depthTestEnable = false;

            m_pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        }

        m_commandList->open();

        nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.f));

        // Camera at (0,0,2) looking at (0,0,-1) direction, up direction (0,1,0)
        float3 cameraUp(0, 1, 0);
        float4 viewDir(0, 0, -1, 0);

        ////////////////////
        //
        // Fill out the constant buffer including the neural weight/bias offsets.
        //
        ////////////////////
        NeuralConstants modelConstant{ {},
                                       {},
                                       { 0, 0, 2, 0 },
                                       float4(m_lightDir, 1.f),
                                       float4(m_userInterfaceParameters->lightIntensity),
                                       float4(.82f, .67f, .16f, 1.f),
                                       m_userInterfaceParameters->specular,
                                       m_userInterfaceParameters->roughness,
                                       m_userInterfaceParameters->metallic,
                                       0.f,
                                       m_weightOffsets,
                                       m_biasOffsets };
        modelConstant.view = affineToHomogeneous(translation(-modelConstant.cameraPos.xyz()) * lookatZ(-viewDir.xyz(), cameraUp));
        modelConstant.viewProject = modelConstant.view * perspProjD3DStyle(radians(67.4f), float(width) / float(height), 0.1f, 10.f);

        // Upload the constant buffer.
        m_commandList->writeBuffer(m_constantBuffer, &modelConstant, sizeof(modelConstant));

        nvrhi::GraphicsState state;
        state.bindings = { m_bindingSet };
        state.indexBuffer = { m_indexBuffer, nvrhi::Format::R32_UINT, 0 };

        state.vertexBuffers = {
            { m_vertexBuffer, 0, offsetof(Vertex, position) },
            { m_vertexBuffer, 1, offsetof(Vertex, normal) },
        };
        state.pipeline = m_pipeline;
        state.framebuffer = framebuffer;

        const nvrhi::Viewport viewport = nvrhi::Viewport(left, left + width, top, top + height, 0.f, 1.f);
        state.viewport.addViewportAndScissorRect(viewport);

        if (updateStat)
        {
            GetDevice()->resetTimerQuery(m_neuralTimer);
            m_commandList->beginTimerQuery(m_neuralTimer);
        }

        // Update the pipeline, bindings, and other state.
        m_commandList->setGraphicsState(state);

        // Draw the model.
        nvrhi::DrawArguments args;
        args.vertexCount = m_indicesNum;
        m_commandList->drawIndexed(args);

        if (updateStat)
        {
            m_commandList->endTimerQuery(m_neuralTimer);
        }

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
    }

private:
    std::string m_extraStatus;
    nvrhi::TimerQueryHandle m_neuralTimer;
    nvrhi::ShaderHandle m_vertexShader;
    nvrhi::ShaderHandle m_pixelShader;
    nvrhi::BufferHandle m_constantBuffer;
    nvrhi::BufferHandle m_mlpHostBuffer;
    nvrhi::BufferHandle m_mlpDeviceBuffer;
    nvrhi::BufferHandle m_vertexBuffer;
    nvrhi::BufferHandle m_indexBuffer;
    nvrhi::InputLayoutHandle m_inputLayout;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::GraphicsPipelineHandle m_pipeline;
    nvrhi::CommandListHandle m_commandList;

    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;

    float3 m_lightDir{ -0.761f, -0.467f, -0.450f };
    float2 m_currentXY;
    bool m_pressedFlag = false;

    int m_indicesNum = 0;

    dm::uint4 m_weightOffsets; // Offsets to weight matrices in bytes.
    dm::uint4 m_biasOffsets; // Offsets to bias vectors in bytes.

    UIData* m_userInterfaceParameters;
};

class UserInterface : public app::ImGui_Renderer
{
private:
    UIData* m_userInterfaceParameters;

public:
    UserInterface(app::DeviceManager* deviceManager, UIData* uiParams) : ImGui_Renderer(deviceManager), m_userInterfaceParameters(uiParams)
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

        ImGui::End();
    }
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
        log::error("This sample does not support D3D11");
        return 1;
    }
    std::unique_ptr<app::DeviceManager> deviceManager(app::DeviceManager::Create(graphicsApi));

    app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = deviceParams.backBufferHeight;

#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableGPUValidation = false;
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
        log::fatal("Cannot initialize a graphics device with the requested parameters. Please try a NVIDIA driver version greater than 570");
        return 1;
    }
    auto graphicsResources = std::make_unique<rtxns::GraphicsResources>(deviceManager->GetDevice());
    if (!graphicsResources->GetCoopVectorFeatures().inferenceSupported && !graphicsResources->GetCoopVectorFeatures().fp16InferencingSupported)
    {
        log::fatal("Not all required Coop Vector features are available");
        return 1;
    }

    {
        UIData uiData;
        SimpleInferencing example(deviceManager.get(), &uiData);
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
