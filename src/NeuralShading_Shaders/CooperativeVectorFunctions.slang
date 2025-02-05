/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Additional functions for use with the CoopVec type

namespace rtxns
{

// Relu function implementation
CoopVec<T, K> relu<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    return max(v, CoopVec<T, K>(T(0.)));
}

// Leaky relu function implementation
CoopVec<T, K> leakyReLU<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> p, no_diff T a)
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        p[i] = p[i] < T(0.) ? a * p[i] : p[i];
    }
    return p;
}

// Sigmoid function implementation
CoopVec<T, K> sigmoid<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    var c1 = CoopVec<T, K>(T(1.));
    return c1 / (c1 + exp(CoopVec<T, K>(T(-1.)) * v));
}



}