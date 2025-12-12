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

#include <donut/app/imgui_renderer.h>
#include <imgui_internal.h>
#include <implot.h>
#include <vector>

#include "WidgetInterface.h"

struct UIConfig
{
    ImVec2 windowPosition = ImVec2(10.f, 10.f);
    ImVec2 uiSize = ImVec2(300, 300);
};

class UserInterface : public donut::app::ImGui_Renderer
{
public:
    UserInterface(donut::app::DeviceManager* deviceManager, UIConfig uiConfig = {});
    ~UserInterface();

    void BackBufferResized(const uint32_t width, const uint32_t height, const uint32_t sampleCount) override;
    void buildUI() override;
    void AddWidget(IWidget* const widget);

private:
    UIConfig m_uiConfig;
    ImPlotContext* m_implot = nullptr;
    std::vector<IWidget*> m_widgets;
};