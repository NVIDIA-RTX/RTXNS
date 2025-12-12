/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/app/UserInterfaceUtils.h>
#include "UserInterface.h"

UserInterface::UserInterface(donut::app::DeviceManager* deviceManager, UIConfig uiConfig /*= {}*/) : ImGui_Renderer(deviceManager), m_uiConfig(uiConfig)
{
}

UserInterface::~UserInterface()
{
    ImPlot::DestroyContext(m_implot);
}

void UserInterface::BackBufferResized(const uint32_t width, const uint32_t height, const uint32_t sampleCount)
{
    m_implot = ImPlot::CreateContext();
}

void UserInterface::buildUI()
{
    ImVec2 displaySize = ImGui::GetIO().DisplaySize;

    ImGui::SetNextWindowPos(m_uiConfig.windowPosition, 0);
    ImVec2 uiSize = ImVec2(m_uiConfig.uiSize.x, displaySize.y * 0.95f);
    ImGui::SetNextWindowSize(uiSize, ImGuiCond_Always);

    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_None);

    for (auto& widgets : m_widgets)
    {
        widgets->Draw();
        ImGui::Separator();
    }

    ImGui::End();
}

void UserInterface::AddWidget(IWidget* const widget)
{
    m_widgets.push_back(widget);
}
