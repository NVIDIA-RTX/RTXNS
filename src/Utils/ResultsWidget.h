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

#include "WidgetInterface.h"

using namespace donut::math;
#include "TrainingResults.h"

class ResultsWidget : public IWidget
{
public:
    ResultsWidget();
    void Draw() override;
    void Reset();
    void Update(const TrainingResults& trainingResults);

private:
    std::vector<float> m_epochHistory;
    std::vector<float> m_averageL2LossHistory;
};
