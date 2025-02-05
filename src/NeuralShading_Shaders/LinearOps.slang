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
    // Linear (without activation function) forward step of MLP using Cooperative Vector extension functions
    // Weights matrix and biases vector are stored in matrixBiasBuffer at offsets matrixOffset and biasOffset
    CoopVec<T, M> LinearOp<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
        CoopVec<T, K> ip, 
        StructuredBuffer<T> matrixBiasBuffer, 
        uint matrixOffset, 
        int biasOffset, 
        constexpr CoopVecMatrixLayout matrixLayout, 
        constexpr CoopVecComponentType componentType)
    {
        return coopVecMatMulAdd<T, M>(
            ip, 
            componentType, 
            matrixBiasBuffer, 
            matrixOffset, 
            componentType, 
            matrixBiasBuffer, 
            biasOffset, 
            componentType, 
            matrixLayout, 
            false, 
            0
        );
    }

    // One linear backward step of MLP using Cooperative Vector extension functions
    // Weights matrix and biases vector are stored in matrixBiasBuffer at offsets matrixOffset and biasOffset
    // Derivates of weights matrix and derivatives of biases vector are stored in matrixBiasBufferDerivative at offsets matrixOffset and biasOffset
    CoopVec<T, K> LinearOp_Backward<T : __BuiltinFloatingPointType, let M : int, let K : int>(
        CoopVec<T, K> ip, 
        CoopVec<T, M> grad, 
        StructuredBuffer<T> matrixBiasBuffer, 
        RWStructuredBuffer<T> matrixBiasBufferDerivative, 
        uint matrixOffset, 
        int biasOffset, 
        constexpr CoopVecMatrixLayout matrixLayout, 
        constexpr CoopVecComponentType componentType)
    {
        coopVecOuterProductAccumulate(grad, ip, matrixBiasBufferDerivative, matrixOffset, 0, matrixLayout, componentType);
        coopVecReduceSumAccumulate(grad, matrixBiasBufferDerivative, biasOffset);

        return coopVecMatMul<T, K>(grad, componentType, matrixBiasBuffer, matrixOffset, componentType, matrixLayout, true, 0);
    }
}

namespace rtxns
{
namespace mlp
{
    // Structure to store derivatives of weights matrix and derivatives of biases vector
    // Extends IDifferentiablePtrType interface to support autodiff
    struct MatrixBiasBufferDifferential<T : __BuiltinFloatingPointType> : IDifferentiablePtrType
    {
        typealias Differential = MatrixBiasBufferDifferential<T>;

        __init(RWStructuredBuffer<T> buf) 
        { 
            buffer = buf;
        }

        RWStructuredBuffer<T> buffer;
    };

    // Structure to store weights matrix and biases vector
    // Extends IDifferentiablePtrType interface to support autodiff
    struct MatrixBiasBuffer<T : __BuiltinFloatingPointType> : IDifferentiablePtrType
    {
        typealias Differential = MatrixBiasBufferDifferential<T>;

        __init(StructuredBuffer<T> buf) 
        { 
            buffer = buf;
        }

        StructuredBuffer<T> buffer;
    };

    // Linear forward step of MLP using MatrixBiasBuffer structure to store weights and biases
    CoopVec<T, M> LinearOp<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
        CoopVec<T, K> ip, 
        MatrixBiasBuffer<T> MatrixBiasBuffer, 
        uint2 offsets,
        constexpr CoopVecMatrixLayout matrixLayout, 
        constexpr CoopVecComponentType componentType)
    {
        return LinearOp<T, M, K>(
            ip, 
            MatrixBiasBuffer.buffer, 
            offsets[0], 
            offsets[1], 
            matrixLayout, 
            componentType
        );
    }

    // Linear backward step of MLP using MatrixBiasBuffer and MatrixBiasBufferDifferential
    [BackwardDerivativeOf(LinearOp)]
    void LinearOp_BackwardAutoDiff<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
        inout DifferentialPair<CoopVec<T, K>> ip, 
        DifferentialPtrPair<MatrixBiasBuffer<T>> MatrixBiasBuffer, 
        uint2 offsets,
        constexpr CoopVecMatrixLayout matrixLayout, 
        constexpr CoopVecComponentType componentType, CoopVec<T, M>.Differential grad)
    {
        ip = diffPair(
            ip.p, 
            LinearOp_Backward<T, M, K>(
                ip.p, 
                grad, 
                MatrixBiasBuffer.p.buffer, 
                MatrixBiasBuffer.d.buffer, 
                offsets[0], 
                offsets[1], 
                matrixLayout, 
                componentType
            )
        );
    }
}
}