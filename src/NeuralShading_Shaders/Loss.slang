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
    // Loss function interface and implementation for several loss functions
    // for using with classes in MLP module
    //
    ////////////////////////

    // Base interface for activation functions
    interface ILoss<T : __BuiltinFloatingPointType, let K : int>
    {
        static vector<T, K> value(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale);
        static vector<T, K> deriv(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale);
    };

    // L1
    struct L1<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static vector<T, K> value(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return scale * abs(predicted - target);
        }

        static vector<T, K> deriv(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return copysign(scale, predicted - target);
        }
    };

    // Relative L1
    struct L1Relative<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static vector<T, K> value(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return scale * abs(predicted - target) / (abs(predicted) + vector<T, K>(0.01f));
        }

        static vector<T, K> deriv(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return copysign(scale, predicted - target) / (abs(predicted) + vector<T, K>(0.01f));
        }
    };

    // Mean absolute percentage error (MAPE)
    struct MAPE<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static vector<T, K> value(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return scale * abs(predicted - target) / (abs(target) + vector<T, K>(0.01f));
        }

        static vector<T, K> deriv(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return copysign(scale, predicted - target) / (abs(target) + vector<T, K>(0.01f));
        }
    };

    // Symmetric mean absolute percentage error (SMAPE)
    struct SMAPE<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static vector<T, K> value(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return scale * abs(predicted - target) / ((abs(target) + abs(predicted)) * T(0.5) + vector<T, K>(0.01f));
        }

        static vector<T, K> deriv(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return copysign(scale, predicted - target) / ((abs(target) + abs(predicted))*T(0.5) + vector<T, K>(0.01f));
        }
    };

    // L2 
    struct L2<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static vector<T, K> value(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return scale * (predicted - target) * (predicted - target);
        }

        static vector<T, K> deriv(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return T(2) * scale * (predicted - target);
        }
    };

    // Relative L2
    struct L2Relative<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static vector<T, K> value(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return scale * (predicted - target) * (predicted - target) / (predicted * predicted + vector<T, K>(0.01f));
        }

        static vector<T, K> deriv(vector<T, K> target, vector<T, K> predicted, vector<T, K> scale)
        {
            return T(2) * scale * (predicted - target) / (predicted * predicted + vector<T, K>(0.01f));
        }
    };
}
}
