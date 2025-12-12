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
#include "UIData.h"

class UIWidget : public IWidget
{
public:
    UIWidget(UIData& uiData);
    ~UIWidget();
    void Draw() override;
    void Reset();

private:
    UIData& m_uiData;
};
