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

// Get local path for subfolder, used for creating a standalone binary package.
// Searches upwards from the executable's directory, so it works both for the repository layout
// (<repo>/bin/<platform>/<config>/app.exe with <repo>/assets) and for a flattened package
// (<package>/app.exe with <package>/assets).
std::filesystem::path GetLocalPath(std::string subfolder)
{
    constexpr int maxDepth = 5;

    std::filesystem::path dir = donut::app::GetDirectoryWithExecutable();
    std::filesystem::path candidate = dir / subfolder;
    for (int depth = 0; depth < maxDepth; ++depth)
    {
        candidate = dir / subfolder;
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }

        std::filesystem::path parent = dir.parent_path();
        if (parent.empty() || parent == dir)
        {
            break;
        }
        dir = parent;
    }

    // Not found: return the last candidate so callers report a meaningful path.
    return candidate;
}