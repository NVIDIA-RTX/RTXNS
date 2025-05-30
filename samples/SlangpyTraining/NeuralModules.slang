/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
__exported import CooperativeVectorDerivatives;
__exported import CooperativeVectorFunctions;
__exported import CooperativeVectorAutoDiff;
__exported import Optimizers;
__exported import Utils;
__exported import LinearOps;
__exported import MLP;
__exported import Activation;

namespace rtxns
{
    ////////////////////////
    //
    // Root interface for neural modules and implementations of several
    // Takes a CoopVec of type T with NumInput elements and returns NumOutputs elements
    // Several RTXNS classes and functions are wrapped here to conform to the IModule interface
    // This lets you build network architectures with generic types
    //
    ////////////////////////
    interface IModule<T : __BuiltinFloatingPointType, let NumInputs : int, let NumOutputs : int>
    {
        [BackwardDifferentiable]
        CoopVec<T, NumOutputs> forward(CoopVec<T, NumInputs> inputParams);
    }

    // Chain two modules together, i.e. pass the output of the first to the second
    // Can be nested arbitrarily
    struct ModuleChain<
        T : __BuiltinFloatingPointType,
        let NumInputs : int,
        let NumHidden : int,
        let NumOutputs : int,
        First : IModule<T, NumInputs, NumHidden>,
        Second : IModule<T, NumHidden, NumOutputs>
    > : IModule<T, NumInputs, NumOutputs>
    {
        First first;
        Second second;

        [BackwardDifferentiable]
        CoopVec<T, NumOutputs> forward(CoopVec<T, NumInputs> inputParams)
        {
            CoopVec<T, NumHidden> middle = first.forward(inputParams);
            return second.forward(middle);
        }
    }

    struct TrainableMLPModule<
        T : __BuiltinFloatingPointType,
        let NumHiddenLayers : int,
        let InputNeurons : int,
        let HiddenNeurons : int,
        let OutputNeurons : int,
        let ComponentType : CoopVecComponentType,
        HiddenAct : mlp::IActivation<T, HiddenNeurons>,
        OutputAct : mlp::IActivation<T, OutputNeurons>
    > : IModule<T, InputNeurons, OutputNeurons>
    {
        ByteAddressBuffer parameters;
        RWByteAddressBuffer derivatives; 
        uint matrixOffsets[NumHiddenLayers + 1];
        uint biasOffsets[NumHiddenLayers + 1];

        HiddenAct hiddenAct;
        OutputAct outputAct;

        [BackwardDerivative(backward)]
        CoopVec<T, OutputNeurons> forward(CoopVec<T, InputNeurons> inputParams)
        {
            var mlp = mlp::TrainingMLP<
                T, 
                NumHiddenLayers, 
                InputNeurons, 
                HiddenNeurons, 
                OutputNeurons, 
                CoopVecMatrixLayout::TrainingOptimal, 
                ComponentType
            >(parameters, derivatives, matrixOffsets, biasOffsets);
            return mlp.forward(inputParams, hiddenAct, outputAct);
        }

        void backward(inout DifferentialPair<CoopVec<T, InputNeurons>> inputParams, const CoopVec<T, OutputNeurons> dOutputActivated)
        {
            var mlp = mlp::TrainingMLP<
                T, 
                NumHiddenLayers, 
                InputNeurons, 
                HiddenNeurons, 
                OutputNeurons, 
                CoopVecMatrixLayout::TrainingOptimal, 
                ComponentType
            >(parameters, derivatives, matrixOffsets, biasOffsets);
            
            mlp.backward(inputParams, hiddenAct, outputAct, dOutputActivated);
        }
    }

    struct InferenceMLPModule<
        T : __BuiltinFloatingPointType,
        let NumHiddenLayers : int,
        let InputNeurons : int,
        let HiddenNeurons : int,
        let OutputNeurons : int,
        let ComponentType : CoopVecComponentType,
        HiddenAct : mlp::IActivation<T, HiddenNeurons>,
        OutputAct : mlp::IActivation<T, OutputNeurons>
    > : IModule<T, InputNeurons, OutputNeurons>
    {
        ByteAddressBuffer parameters;
        uint matrixOffsets[NumHiddenLayers + 1];
        uint biasOffsets[NumHiddenLayers + 1];

        HiddenAct hiddenAct;
        OutputAct outputAct;

        [TreatAsDifferentiable]
        CoopVec<T, OutputNeurons> forward(CoopVec<T, InputNeurons> inputParams)
        {
            var mlp = mlp::InferenceMLP<
                T, 
                NumHiddenLayers, 
                InputNeurons, 
                HiddenNeurons, 
                OutputNeurons, 
                CoopVecMatrixLayout::InferencingOptimal, 
                ComponentType
            >(parameters, matrixOffsets, biasOffsets);
            return mlp.forward(inputParams, hiddenAct, outputAct);
        }
    }

    struct FrequencyEncoding<T : __BuiltinFloatingPointType, let NumInputs : int, let NumScales : int> : IModule<T, NumInputs, NumScales * NumInputs * 2>
    {
        [BackwardDifferentiable]
        CoopVec<T, NumScales * NumInputs * 2> forward(CoopVec<T, NumInputs> inputParams)
        {
            return rtxns::EncodeFrequencyN<T, NumInputs, NumScales>(inputParams);
        }
    }
}
