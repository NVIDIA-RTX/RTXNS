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
import CooperativeVectorDerivatives;

// Implementation to extend CoopVec to make it automatically differentiable (autodiff)

namespace rtxns
{

// Extension for builtin type CoopVec<T, K> to make it automatically differentiable (autodiff)
extension<T : __BuiltinFloatingPointType, let K : int> CoopVec<T, K> : IDifferentiable
{
    typealias Differential = CoopVec<T, K>;
};

typealias HCoopVec<let K : int> = CoopVec<half, K>;

////////////////////////
//
// Additional functions and their derivatives for using in actvation functions
// To support Slang autodiff, for each function its derivative should be defined
//
////////////////////////

// exp is builtin function, so we just need to define derivative for autodiff support
[BackwardDerivativeOf(exp)]
void exp_BackwardAutoDiff<T : __BuiltinFloatingPointType, let K : int>(inout DifferentialPair<CoopVec<T, K>> p0, CoopVec<T, K>.Differential dResult)
{
    p0 = diffPair(p0.p, dResult * exp(p0.p));
}

// Relu backward derivative
[BackwardDerivativeOf(relu)]
void relu_BackwardAutoDiff<T : __BuiltinFloatingPointType, let K : int>(inout DifferentialPair<CoopVec<T, K>> p0, CoopVec<T, K>.Differential dResult)
{
    CoopVec<T, K> d;

    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        d[i] = p0.p[i] > T(0.) ? dResult[i] : T(0.);
    }

    p0 = diffPair(p0.p, d);
}

// LeakyRelu backward derivative
[BackwardDerivativeOf(leakyReLU)]
void leakyRelu_BackwardAutoDiff<T : __BuiltinFloatingPointType, let K : int>(inout DifferentialPair<CoopVec<T, K>> p0, T a, CoopVec<T, K>.Differential dResult)
{
    p0 = diffPair(p0.p, leakyReLU_Derivative(p0.p, a, dResult));
}

// Sigmoid backward derivative
[BackwardDerivativeOf(sigmoid)]
void sigmoid_BackwardAutoDiff<T : __BuiltinFloatingPointType, let K : int>(inout DifferentialPair<CoopVec<T, K>> p0, CoopVec<T, K>.Differential dResult)
{
    p0 = diffPair(p0.p, sigmoid_Derivative(p0.p, dResult));
}

}