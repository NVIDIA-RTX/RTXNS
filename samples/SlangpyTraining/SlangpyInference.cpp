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

static const char* g_defaultTitle = "RTXNS Example: Showing pretrained model. Run SlangpyTraining.py to train new model";
static const char* g_loadedTitle = "RTXNS Example: Inference of Slangpy Training result (Ground Truth | Trained)";
static const char* g_windowTitle = g_defaultTitle;

struct UIData
{
    bool load = false;
    bool isPretrained = true;
    std::string fileName;
};

class SlangpyInference : public app::ApplicationBase
{

public:
    SlangpyInference(app::DeviceManager* deviceManager, UIData* ui) : ApplicationBase(deviceManager), m_uiParams(ui)
    {
    }

    bool Init()
    {
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/SlangpyTraining" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        m_RootFS = std::make_shared<vfs::RootFileSystem>();
        m_RootFS->mount("/shaders/donut", frameworkShaderPath);
        m_RootFS->mount("/shaders/app", appShaderPath);

        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), m_RootFS, "/shaders");
        m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_ShaderFactory);
        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());

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

        auto texDesc = m_InputTexture->getDesc();
        texDesc.debugName = "InferenceTexture";
        texDesc.format = nvrhi::Format::RGBA16_FLOAT;
        texDesc.isRenderTarget = true;
        texDesc.isUAV = true;
        texDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        texDesc.keepInitialState = true;
        m_InferenceTexture = GetDevice()->createTexture(texDesc);

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);

        return true;
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

    bool InitializeNetwork(const rtxns::HostNetwork& network)
    {
        ////////////////////
        //
        // Create the Neural network class and initialise it the hyper parameters from NetworkConfig.h.
        //
        ////////////////////
        m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);

        const auto& arch = network.GetNetworkArchitecture();
        if (!m_neuralNetwork->InitialiseFromNetwork(network))
        {
            log::error("Failed to create a network.");
            return false;
        }

        // Get a device optimized layout
        m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

        ////////////////////
        //
        // Create the shaders/buffers/textures for the Neural Training
        //
        ////////////////////
        m_InferencePass.m_ShaderCS = m_ShaderFactory->CreateShader("app/SlangpyInference", "main_cs", nullptr, nvrhi::ShaderType::Compute);

        nvrhi::BufferDesc paramsBufferDesc;
        paramsBufferDesc.byteSize = m_neuralNetwork->GetNetworkParams().size();
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
        paramsBufferDesc.debugName = "MLPParamsDeviceBuffer";
        paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        m_mlpDeviceBuffer = GetDevice()->createBuffer(paramsBufferDesc);


        m_TotalParamCount = (uint32_t)(paramsBufferDesc.byteSize / GetSize(arch.weightPrecision));

        m_commandList->open();
        m_commandList->writeBuffer(m_mlpHostBuffer, m_neuralNetwork->GetNetworkParams().data(), m_neuralNetwork->GetNetworkParams().size());

        // Set up the constant buffers
        m_NeuralConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(NeuralConstants), "NeuralConstantBuffer")
                                                               .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                               .setKeepInitialState(true));

        // Convert to GPU optimized layout
        m_networkUtils->ConvertWeights(m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);
        m_commandList->setBufferState(m_mlpDeviceBuffer, nvrhi::ResourceStates::ShaderResource);
        m_commandList->commitBarriers();

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

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
        GetDevice()->waitForIdle();

        return true;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        if (!m_ModelLoaded)
        {
            GetDeviceManager()->SetInformativeWindowTitle("No model loaded", true);
        }
        else
        {
            GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true);
        }

        ////////////////////
        //
        // Load the Neural network if required
        //
        ////////////////////
        if (!m_uiParams->fileName.empty())
        {
            if (m_uiParams->load)
            {
                rtxns::HostNetwork network(m_networkUtils);
                vfs::NativeFileSystem nativeFS;
                if (network.InitialiseFromJson(nativeFS, m_uiParams->fileName))
                {
                    m_ModelLoaded = InitializeNetwork(network);
                    if (!m_uiParams->isPretrained)
                    {
                        g_windowTitle = g_loadedTitle;
                    }
                }
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
        if (!m_ModelLoaded)
        {
            return;
        }

        const auto& fbinfo = framebuffer->getFramebufferInfo();

        m_commandList->open();

        nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.f));

        ////////////////////
        //
        // Update the Constant buffer
        //
        ////////////////////
        NeuralConstants neuralConstants = {};

        for (int i = 0; i < MAX_LAYER_COUNT; ++i)
        {
            uint32_t weightOffset = 0;
            uint32_t biasOffset = 0;
            if (i < m_deviceNetworkLayout.networkLayers.size())
            {
                weightOffset = m_deviceNetworkLayout.networkLayers[i].weightOffset;
                biasOffset = m_deviceNetworkLayout.networkLayers[i].biasOffset;
            }
            neuralConstants.weightOffsets[i / 4][i % 4] = weightOffset;
            neuralConstants.biasOffsets[i / 4][i % 4] = biasOffset;
        }

        neuralConstants.imageWidth = m_InferenceTexture->getDesc().width;
        neuralConstants.imageHeight = m_InferenceTexture->getDesc().height;
        m_commandList->writeBuffer(m_NeuralConstantBuffer, &neuralConstants, sizeof(neuralConstants));

        nvrhi::ComputeState state;
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
        for (uint32_t viewIndex = 0; viewIndex < 2; ++viewIndex)
        {
            // Construct the viewport so that all viewports form a grid.
            const float width = float(fbinfo.width) / 2;
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

    nvrhi::BufferHandle m_NeuralConstantBuffer;
    nvrhi::BufferHandle m_OptimisationConstantBuffer;

    nvrhi::TextureHandle m_InputTexture;
    nvrhi::TextureHandle m_InferenceTexture;

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

    uint m_TotalParamCount = 0;

    bool m_ModelLoaded = false;

    UIData* m_uiParams;
};

class UserInterface : public app::ImGui_Renderer
{
private:
    UIData* m_uiParams;

public:
    UserInterface(app::DeviceManager* deviceManager, UIData* uiParams) : ImGui_Renderer(deviceManager), m_uiParams(uiParams)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void load(const std::string& fname)
    {
        m_uiParams->fileName = fname;
        m_uiParams->load = true;
        m_uiParams->isPretrained = false;
    }

    void buildUI() override
    {
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
    deviceParams.backBufferWidth = 768 * 2;
    deviceParams.backBufferHeight = 768;

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
    if (!graphicsResources->GetCoopVectorFeatures().inferenceSupported && !graphicsResources->GetCoopVectorFeatures().fp16InferencingSupported)
    {
        log::fatal("Not all required Coop Vector features are available");
        return 1;
    }

    {
        UIData uiData;
        uiData.fileName = (GetLocalPath("assets/data") / "slangpy-weights.json").generic_string();
        uiData.load = true;

        SlangpyInference example(deviceManager, &uiData);
        UserInterface gui(deviceManager, &uiData);
        for (int i = 1; i < __argc; ++i)
        {
            std::string arg = __argv[i];
            if (arg.size() >= 5 && arg.substr(arg.size() - 5) == ".json")
            {
                gui.load(arg);
                break;
            }
        }
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
