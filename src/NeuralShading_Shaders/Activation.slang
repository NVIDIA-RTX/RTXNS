/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

import CooperativeVectorAutoDiff;
import CooperativeVectorFunctions;

namespace rtxns
{
namespace mlp
{
    ////////////////////////
    //
    // Activation function interface and implementation for several activation functions
    // for using with classes in MLP module
    //
    ////////////////////////

    // Base interface for activation functions
    interface IActivation<T : __BuiltinFloatingPointType, let K : int>
    {
        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x);
    };

    // None activation function
    struct NoneAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            return x;
        }
    };

    // Linear activation function
    struct LinearAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        T a;

        __init(T a)
        {
            this.a = a;
        }

        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            return no_diff CoopVec<T, K>(a) * x;
        }
    };

    // Exponential activation function
    struct ExponentialAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            // Exponent is builtin function.
            return exp(x); 
        }
    };

    // Shifted exponential activation function
    struct ShiftedExponentialAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            return exp(x) - no_diff CoopVec<T, K>(T(1.));
        }
    };

    // ReLU activation function
    struct ReLUAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            return relu(x); 
        }
    };

    // Leaky ReLU activation function
    struct LeakyReLUAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        T a;
        
        __init(T a) 
        { 
            this.a = a; 
        }

        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            return leakyReLU(x, a);
        }
    };

    // Sigmoid activation function
    struct SigmoidAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            // Sigmoid function calculation. Compiler will infer the derivative automatically (autodiff)
            return sigmoid(x);
        }
    };

    // Swish activation function
    struct SwishAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            return x / (no_diff CoopVec<T, K>(T(1.)) + exp(no_diff CoopVec<T, K>(T(-1.)) * x));
        }
    };

    // Tanh activation function
    struct TanhAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>
    {
        [Differentiable]
        CoopVec<T, K> eval(CoopVec<T, K> x)
        {
            var c1 = no_diff CoopVec<T, K>(T(1.));
            return no_diff CoopVec<T, K>(T(2.)) / (c1 + exp(no_diff CoopVec<T, K>(T(-2.)) * x)) - c1;
        }
    };
}
}
