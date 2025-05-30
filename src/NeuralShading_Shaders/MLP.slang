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
import LinearOps;
import Activation;

namespace rtxns
{
namespace mlp
{
    // Structure to store MLP layers. Implements full forward step for inference
    // MLP is defined by number of hidden layers and number of inputs, outputs and elements in hidden layers
    struct InferenceMLP<
        T : __BuiltinFloatingPointType, 
        let HIDDEN_LAYERS : int, 
        let INPUTS : int, 
        let HIDDEN : int, 
        let OUTPUTS : int, 
        let matrixLayout : CoopVecMatrixLayout, 
        let componentType : CoopVecComponentType
    >
    {
        // Initialized from buffer with weights and biases and two vectors of offsets
        __init(ByteAddressBuffer buf, uint matrixOffset[HIDDEN_LAYERS+1], uint biasOffset[HIDDEN_LAYERS+1]) 
        {
            parameters = MatrixBiasBuffer(buf);

            [ForceUnroll]
            for (int i = 0; i <= HIDDEN_LAYERS; ++i)
                layerOffsets[i] = uint2(matrixOffset[i], biasOffset[i]);
        }

        // Full MLP forward step using one activation function for input and hidden layers and another for output
        // Returns MLP output
        CoopVec<T, OUTPUTS> forward<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(CoopVec<T, INPUTS> inputParams, Act act, FinalAct finalAct)
        {
            var params = act.eval(LinearOp<T, HIDDEN, INPUTS>(inputParams, parameters, layerOffsets[0], matrixLayout, componentType));
            
            [ForceUnroll]
            for(int i = 1; i < HIDDEN_LAYERS; ++i)
                params = act.eval(LinearOp<T, HIDDEN, HIDDEN>(params, parameters, layerOffsets[i], matrixLayout, componentType));

            return finalAct.eval(LinearOp<T, OUTPUTS, HIDDEN>(params, parameters, layerOffsets[HIDDEN_LAYERS], matrixLayout, componentType));
        }

        MatrixBiasBuffer parameters;
        uint2 layerOffsets[HIDDEN_LAYERS+1];
    }

    // Structure to store MLP layers and derivatives. Implements full forward step and backward steps
    // MLP is defined by number of hidden layers and number of inputs, outputs and elements in hidden layers
    struct TrainingMLP<
        T : __BuiltinFloatingPointType, 
        let HIDDEN_LAYERS : int, 
        let INPUTS : int, 
        let HIDDEN : int, 
        let OUTPUTS : int, 
        let matrixLayout : CoopVecMatrixLayout, 
        let componentType : CoopVecComponentType
    >
    {
        // Initialized from buffer with weights and biases, buffer to store derivatives and two vectors of offsets
        __init(
            ByteAddressBuffer matrixBuffer, 
            RWByteAddressBuffer derivativeBuffer, 
            uint matrixOffset[HIDDEN_LAYERS+1], 
            uint biasOffset[HIDDEN_LAYERS+1]
        ) 
        {
            parameters = MatrixBiasBuffer(matrixBuffer);
            derivatives = MatrixBiasBufferDifferential(derivativeBuffer);
            
            [ForceUnroll]
            for (int i = 0; i <= HIDDEN_LAYERS; ++i)
                layerOffsets[i] = uint2(matrixOffset[i], biasOffset[i]);
        }

        // Full MLP forward step using one activation function for input and hidden layers and another for output
        // Implemented as static function to support autodiff
        // Input parameter inputParams is no_diff to skip derivative calculation for inputs as we interested in weights and biases derivatives only
        // Returns MLP output
        [Differentiable]
        static CoopVec<T, OUTPUTS> forward_s<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(
            CoopVec<T, INPUTS> inputParams, 
            MatrixBiasBuffer parameters,
            uint2 layerOffsets[HIDDEN_LAYERS + 1],
            no_diff Act act,
            no_diff FinalAct finalAct
        )
        {
            var params = act.eval(LinearOp<T, HIDDEN, INPUTS>(inputParams, parameters, layerOffsets[0], matrixLayout, componentType));
            
            [ForceUnroll]
            for(int i = 1; i < HIDDEN_LAYERS; ++i)
                params = act.eval(LinearOp<T, HIDDEN, HIDDEN>(params, parameters, layerOffsets[i], matrixLayout, componentType));

            return finalAct.eval(LinearOp<T, OUTPUTS, HIDDEN>(params, parameters, layerOffsets[HIDDEN_LAYERS], matrixLayout, componentType));
        }

        // Forward step member function
        CoopVec<T, OUTPUTS> forward<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(CoopVec<T, INPUTS> inputParams, Act act, FinalAct finalAct)
        {
            return forward_s<Act, FinalAct>(inputParams, parameters, layerOffsets, act, finalAct);
        }

        // Full MLP backward step calculation is infered automatically by Slang autodiff from forward function
        void backward<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(DifferentialPair<CoopVec<T, INPUTS>> dInputParams, Act act, FinalAct finalAct, CoopVec<T, OUTPUTS> loss)
        {
            bwd_diff(forward_s<Act, FinalAct>)( dInputParams, DifferentialPtrPair<MatrixBiasBuffer>(parameters, derivatives), layerOffsets, act, finalAct, loss);
        }
        void backward<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(CoopVec<T, INPUTS> inputParams, Act act, FinalAct finalAct, CoopVec<T, OUTPUTS> loss)
        {
            var dInputParams = diffPair(inputParams);
            backward(dInputParams, act, finalAct, loss);
        }

        MatrixBiasBuffer parameters;
        MatrixBiasBufferDifferential derivatives;
        uint2 layerOffsets[HIDDEN_LAYERS+1];
    }
}
}