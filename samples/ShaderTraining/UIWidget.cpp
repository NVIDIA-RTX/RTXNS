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
#include <donut/app/imgui_renderer.h>
#include <imgui_internal.h>

#include "UIWidget.h"

UIWidget::UIWidget(UIData& uiData) : m_uiData(uiData)
{
}

UIWidget::~UIWidget()
{
}

void UIWidget::Draw()
{
    ImGui::SliderFloat("Light Intensity", &m_uiData.lightIntensity, 0.f, 20.f);
    ImGui::SliderFloat("Specular", &m_uiData.specular, 0.f, 1.f);
    ImGui::SliderFloat("Roughness", &m_uiData.roughness, 0.3f, 1.f);
    ImGui::SliderFloat("Metallic", &m_uiData.metallic, 0.f, 1.f);

    ImGui::Text("Epochs : %d", m_uiData.epochs);
    ImGui::Text("Training Time : %.2f s", m_uiData.trainingTime);

    if (ImGui::Button(m_uiData.training ? "Disable Training" : "Enable Training"))
    {
        m_uiData.training = !m_uiData.training;
    }

    if (ImGui::Button("Reset Training"))
    {
        m_uiData.reset = true;
    }

    if (ImGui::Button("Load Model"))
    {
        std::string fileName;
        if (donut::app::FileDialog(true, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
        {
            m_uiData.fileName = fileName;
            m_uiData.load = true;
        }
    }

    if (ImGui::Button("Save Model"))
    {
        std::string fileName;
        if (donut::app::FileDialog(false, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
        {
            m_uiData.fileName = fileName;
            m_uiData.load = false;
        }
    }
}

void UIWidget::Reset()
{
    m_uiData.epochs = 0;
    m_uiData.trainingTime = 0.0f;
}
