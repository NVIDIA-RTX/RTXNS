/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

import CooperativeVectorFunctions;

// Derivatives of the functions in the CooperativeVectorFunction module

namespace rtxns
{

// Derivative of leaky relu
CoopVec<T, K> leakyReLU_Derivative<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> p, T a, CoopVec<T, K> dResult)
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        p[i] = p[i] > T(0.) ? dResult[i] : a * dResult[i];
    }
    return p;
}

// Derivative of sigmoid 
CoopVec<T, K> sigmoid_Derivative<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> p, CoopVec<T, K> dResult)
{
    var sigmoidOut = sigmoid(p);
    return dResult * sigmoidOut * (CoopVec<T, K>(T(1.)) - sigmoidOut);
}

}