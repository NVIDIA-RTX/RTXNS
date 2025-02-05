/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "DirectoryHelper.h"

// Get local path for subfolder, used for creating a standalone binary package
std::filesystem::path GetLocalPath(std::string subfolder)
{
    // Repository path
    std::filesystem::path candidateA = donut::app::GetDirectoryWithExecutable().parent_path() / subfolder;
    // Binary path, assuming the folder is under bin/
    std::filesystem::path candidateB = donut::app::GetDirectoryWithExecutable().parent_path().parent_path() / subfolder;

    if (std::filesystem::exists(candidateA))
    {
        return candidateA;
    }
    else
    {
        return candidateB;
    }
}