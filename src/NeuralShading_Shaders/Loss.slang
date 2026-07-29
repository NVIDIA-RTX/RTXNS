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
import Utils;

#define LOSS_EPSILON 0.01f

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

    // Base interface for loss functions
    // Operates on cooperative vectors of 1 to 128 components 
    interface ILoss<T : __BuiltinFloatingPointType, let K : int>
    {
        static CoopVec<T, K> value(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon);
        static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon);
    };

    // Conversion functions for using standard float vectors of 1 to 4 components with the loss interface
    static vector<float, K> LossValueFromVector<
        let K : int, 
        Loss : ILoss<float, K>>
        (vector<float, K> target, vector<float, K> predicted, vector<float, K> scale, Optional<float> epsilon)
    {
        return VectorFromCoopVec<float, K>(Loss.value(
            CoopVecFromVector<float, K>(target),
            CoopVecFromVector<float, K>(predicted),
            CoopVecFromVector<float, K>(scale), 
            epsilon));
    }

    static vector<float, K> LossValueFromVector<
        let K : int, 
        Loss : ILoss<float, K>>
        (vector<float, K> target, vector<float, K> predicted, vector<float, K> scale)
    {
        return LossValueFromVector<K, Loss>(target, predicted, scale, none);
    }

    static vector<float, K> LossDerivFromVector<
        let K : int, 
        Loss : ILoss<float, K>>
        (vector<float, K> target, vector<float, K> predicted, vector<float, K> scale, Optional<float> epsilon)
    {
        return VectorFromCoopVec<float, K>(Loss.deriv(
            CoopVecFromVector<float, K>(target), 
            CoopVecFromVector<float, K>(predicted), 
            CoopVecFromVector<float, K>(scale), 
            epsilon));
    }

    static vector<float, K> LossDerivFromVector<
        let K : int, 
        Loss : ILoss<float, K>>
        (vector<float, K> target, vector<float, K> predicted, vector<float, K> scale)
    {
        return LossDerivFromVector<K, Loss>(target, predicted, scale, none);
    }

    // ILoss functions 
    // L1
    struct L1<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static CoopVec<T, K> value(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            return scale * abs(predicted - target);
        }

        static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            return copysign(scale, predicted - target);
        }
    };

    // Relative L1
    struct L1Relative<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static CoopVec<T, K> value(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            float eps = select(epsilon.hasValue, epsilon.value, LOSS_EPSILON);
            return scale * abs(predicted - target) / (abs(predicted) + CoopVec<T, K>(T(eps)));
        }

        static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            float eps = select(epsilon.hasValue, epsilon.value, LOSS_EPSILON);
            return copysign(scale, predicted - target) / (abs(predicted) + CoopVec<T, K>(T(eps)));
        }
    };

    // Mean absolute percentage error (MAPE)
    struct MAPE<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static CoopVec<T, K> value(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            float eps = select(epsilon.hasValue, epsilon.value, LOSS_EPSILON);
            return scale * abs(predicted - target) / (abs(target) + CoopVec<T, K>(T(eps)));
        }

        static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            float eps = select(epsilon.hasValue, epsilon.value, LOSS_EPSILON);
            return copysign(scale, predicted - target) / (abs(target) + CoopVec<T, K>(T(eps)));
        }
    };

    // Symmetric mean absolute percentage error (SMAPE)
    struct SMAPE<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static CoopVec<T, K> value(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            float eps = select(epsilon.hasValue, epsilon.value, LOSS_EPSILON);
            return scale * abs(predicted - target) / ((abs(target) + abs(predicted)) * T(0.5) + CoopVec<T, K>(T(eps)));
        }

        static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            float eps = select(epsilon.hasValue, epsilon.value, LOSS_EPSILON);
            return copysign(scale, predicted - target) / ((abs(target) + abs(predicted))*T(0.5) + CoopVec<T, K>(T(eps)));
        }
    };

    // L2 
    struct L2<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static CoopVec<T, K> value(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            return scale * (predicted - target) * (predicted - target);
        }

        static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            return T(2) * scale * (predicted - target);
        }
    };

    // Relative L2
    struct L2Relative<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
    {
        static CoopVec<T, K> value(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            float sum = select(epsilon.hasValue, epsilon.value, LOSS_EPSILON);
            [ForceUnroll]
            for (int i = 0; i < K; ++i)
                sum += target[i].toFloat() * target[i].toFloat();
            return scale * (predicted - target) * (predicted - target) * T(1 / sum);
        }

        static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale, Optional<float> epsilon)
        {
            float sum = select(epsilon.hasValue, epsilon.value, LOSS_EPSILON);
            [ForceUnroll]
            for (int i = 0; i < K; ++i)
                sum += target[i].toFloat() * target[i].toFloat();
            return scale * (predicted - target) * T(2.f / sum);
        }
    };
}
}
