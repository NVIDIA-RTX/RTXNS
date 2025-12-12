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
#include <implot.h>

#include "ResultsWidget.h"

using namespace donut::math;
#include "TrainingResults.h"

namespace
{
constexpr float UI_MAX_EPOCHS = 500.f;
constexpr float UI_LOSS_AXIS_MAX = 1.0f;
constexpr float UI_LOSS_AXIS_MIN = 1e-3f;
} // namespace

ResultsWidget::ResultsWidget()
{
}

void ResultsWidget::Draw()
{
    // Graphs
    if (!m_epochHistory.empty() && !m_averageL2LossHistory.empty())
    {
        ImPlot::SetNextAxisLimits(ImAxis_X1, m_epochHistory.front(), m_epochHistory.front() + UI_MAX_EPOCHS - 1, ImPlotCond_Always);
        if (ImPlot::BeginPlot("Training UI"))
        {
            ImPlot::SetupAxis(ImAxis_X1, "Epochs", ImPlotAxisFlags_LockMin | ImPlotAxisFlags_LockMax);
            ImPlot::SetupAxis(ImAxis_Y1, "L2 Loss", ImPlotAxisFlags_LockMin | ImPlotAxisFlags_LockMax);
            ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
            ImPlot::SetupAxisLimits(ImAxis_Y1, UI_LOSS_AXIS_MIN, UI_LOSS_AXIS_MAX, ImGuiCond_Always);
            ImPlot::PlotLine("Line", m_epochHistory.data(), m_averageL2LossHistory.data(), static_cast<int>(m_epochHistory.size()));

            ImPlot::EndPlot();
        }
    }
}

void ResultsWidget::Reset()
{
    m_epochHistory.clear();
    m_averageL2LossHistory.clear();
}

void ResultsWidget::Update(const TrainingResults& trainingResults)
{
    if (m_epochHistory.size() > UI_MAX_EPOCHS - 1)
    {
        m_epochHistory.erase(m_epochHistory.begin());
        m_averageL2LossHistory.erase(m_averageL2LossHistory.begin());
    }

    m_epochHistory.push_back(static_cast<float>(trainingResults.epoch));
    m_averageL2LossHistory.push_back(trainingResults.l2Loss);
}
