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
#include "GeometryUtils.h"
#include "NeuralNetwork.h"
#include "DirectoryHelper.h"

using namespace donut;
using namespace donut::math;

#include "NetworkConfig.h"
#include <donut/shaders/view_cb.h>

static const char* g_windowTitle = "RTX Neural Shading Example: SlangPy Inferencing (Ground Truth | Training | Loss )";
const int g_ViewSize = 512, g_ViewOffset = 10;

class SlangpyInferencing : public app::ApplicationBase
{
public:
    SlangpyInferencing(app::DeviceManager* deviceManager) : ApplicationBase(deviceManager)
    {
    }

    bool Init()
    {
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/SlangpyInferencing" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

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
        std::filesystem::path textureFileName = dataPath / "slangstars.png";

        std::shared_ptr<engine::LoadedTexture> texture = m_TextureCache->LoadTextureFromFile(textureFileName, false, nullptr, m_commandList);
        if (texture->texture == nullptr)
        {
            log::error("Failed to load texture.");
            return false;
        }
        m_InputTexture = texture->texture;

        ////////////////////
        //
        // Create the Neural network class from json
        //
        ////////////////////
        m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
        m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);

        if (!m_neuralNetwork->InitialiseFromJson(*nativeFS, (dataPath / "weights.json").string()))
        {
            log::error("Failed to create a network.");
            return false;
        }

        // We are expecting 3 layers, validate
        assert(m_neuralNetwork->GetNetworkLayout().networkLayers.size() == 3);

        // Get a device optimized layout
        m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

        ////////////////////
        //
        // Create the shaders/buffers/textures for the Neural Training
        //
        ////////////////////
        m_ShaderCS = m_ShaderFactory->CreateShader("app/SlangpyInferencing_cpp", "inference_cs", nullptr, nvrhi::ShaderType::Compute);

        const auto& params = m_neuralNetwork->GetNetworkParams();

        nvrhi::BufferDesc paramsBufferDesc;
        paramsBufferDesc.byteSize = params.size();
        paramsBufferDesc.debugName = "MLPParamsHostBuffer";
        paramsBufferDesc.canHaveUAVs = true;
        paramsBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        paramsBufferDesc.keepInitialState = true;
        m_mlpHostBuffer = GetDevice()->createBuffer(paramsBufferDesc);

        // Create a buffer for a device optimized parameters layout
        paramsBufferDesc.structStride = sizeof(uint16_t); // Use 16-bit float for weights and biases
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

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
        GetDevice()->waitForIdle();

        auto inputTexDesc = m_InputTexture->getDesc();

        inputTexDesc.debugName = "InferenceTexture";
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
        // Create the pipeline for neural pass
        //
        ////////////////////
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_NeuralConstantBuffer),
            nvrhi::BindingSetItem::Texture_UAV(0, m_InferenceTexture),
            nvrhi::BindingSetItem::Texture_UAV(1, m_LossTexture),
            nvrhi::BindingSetItem::Texture_SRV(0, m_InputTexture),
        };
        {
            int i = 1;
            for (const auto& l : m_deviceNetworkLayout.networkLayers)
            {
                bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::RawBuffer_SRV(i++, m_mlpDeviceBuffer, nvrhi::BufferRange(l.weightOffset, l.weightSize)));
                bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::RawBuffer_SRV(i++, m_mlpDeviceBuffer, nvrhi::BufferRange(l.biasOffset, l.biasSize)));
            }
        }

        nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_BindingLayout, m_BindingSet);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { m_BindingLayout };
        pipelineDesc.CS = m_ShaderCS;
        m_Pipeline = GetDevice()->createComputePipeline(pipelineDesc);

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
        GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true);
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

        nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.f));

        ////////////////////
        //
        // Update the Constant buffer
        //
        ////////////////////
        NeuralConstants neuralConstants;
        neuralConstants.resolution.x = m_InferenceTexture->getDesc().width;
        neuralConstants.resolution.y = m_InferenceTexture->getDesc().height;
        m_commandList->writeBuffer(m_NeuralConstantBuffer, &neuralConstants, sizeof(neuralConstants));

        nvrhi::ComputeState state;

        // inference pass
        state.bindings = { m_BindingSet };
        state.pipeline = m_Pipeline;
        m_commandList->beginMarker("Inference");
        m_commandList->setComputeState(state);
        m_commandList->dispatch(dm::div_ceil(m_InferenceTexture->getDesc().width, 8), dm::div_ceil(m_InferenceTexture->getDesc().height, 8), 1);
        m_commandList->endMarker();

        ////////////////////
        //
        // Render the outputs
        //
        ////////////////////
        for (uint32_t viewIndex = 0; viewIndex < 3; ++viewIndex)
        {
            // Construct the viewport so that all viewports form a grid.
            const float width = float(g_ViewSize);
            const float height = float(fbinfo.height);
            const float left = float((g_ViewSize + g_ViewOffset) * viewIndex);
            const float top = 0;

            const nvrhi::Viewport viewport = nvrhi::Viewport(left, left + width, top, top + height, 0.f, 1.f);

            // Draw original image
            engine::BlitParameters blitParams;
            blitParams.targetFramebuffer = framebuffer;
            blitParams.targetViewport = viewport;
            blitParams.sourceTexture = viewIndex == 0 ? m_InputTexture : viewIndex == 1 ? m_InferenceTexture : m_LossTexture;
            m_CommonPasses->BlitTexture(m_commandList, blitParams, m_BindingCache.get());
        }

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
    }

private:
    std::shared_ptr<vfs::RootFileSystem> m_RootFS;

    nvrhi::CommandListHandle m_commandList;

    nvrhi::ShaderHandle m_ShaderCS;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSet;
    nvrhi::ComputePipelineHandle m_Pipeline;

    nvrhi::BufferHandle m_NeuralConstantBuffer;

    nvrhi::TextureHandle m_InputTexture;
    nvrhi::TextureHandle m_InferenceTexture;
    nvrhi::TextureHandle m_LossTexture;

    nvrhi::BufferHandle m_mlpHostBuffer;
    nvrhi::BufferHandle m_mlpDeviceBuffer;

    nvrhi::FramebufferHandle m_Framebuffer;

    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::unique_ptr<engine::Scene> m_Scene;
    std::shared_ptr<engine::DescriptorTableManager> m_DescriptorTableManager;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;
    std::unique_ptr<rtxns::HostNetwork> m_neuralNetwork;
    rtxns::NetworkLayout m_deviceNetworkLayout;
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
    deviceParams.backBufferWidth = g_ViewSize * 3 + g_ViewOffset * 2;
    deviceParams.backBufferHeight = g_ViewSize;

    deviceParams.enablePerMonitorDPI = true;
    deviceParams.swapChainFormat = nvrhi::Format::RGBA8_UNORM;

    ////////////////////
    //
    // Setup the CoopVector extensions.
    //
    ////////////////////
    SetCoopVectorExtensionParameters(deviceParams, graphicsApi, false, g_windowTitle);

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
        SlangpyInferencing example(deviceManager);
        if (example.Init())
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
