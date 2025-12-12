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
    // Network options
    m_uiData.reset = ImGui::Combo("##networkTransform", (int*)&m_uiData.networkTransform,
                                  "1:1 Mapping\0"
                                  "Zoom\0"
                                  "X/Y Flip\0");
    ImGui::Text("Epochs : %d", m_uiData.epochs);
    ImGui::Text("Adam Steps : %d", m_uiData.adamSteps);
    ImGui::Text("Training Time : %.2f s", m_uiData.trainingTime);
    ImGui::Text("Learning Rate : %.9f", m_uiData.learningRate);

    if (ImGui::Button(m_uiData.training ? "Disable Training" : "Enable Training"))
    {
        m_uiData.training = !m_uiData.training;
    }
    m_uiData.reset |= (ImGui::Button("Reset Training"));

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
