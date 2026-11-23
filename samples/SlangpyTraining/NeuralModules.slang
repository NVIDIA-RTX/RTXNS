/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
__exported import CooperativeVectorFunctions;
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
    // Modules follow the same pattern as TrainingMLP: they are differentiable pointer types
    // (IDifferentiablePtrType). A module holds its parameter buffers and its Differential type holds
    // the matching gradient buffers, so Slang can synthesize the backward pass of forward() and
    // accumulate the gradients into those buffers. bwd_diff of a function taking an IModule receives
    // the two halves as a DifferentialPtrPair; getDifferential() builds the gradient half from the
    // module itself (see TrainTexture in SlangpyTraining.slang).
    //
    ////////////////////////
    interface IModule<T : __BuiltinFloatingPointType, let NumInputs : int, let NumOutputs : int> : IDifferentiablePtrType
    {
        [Differentiable]
        CoopVec<T, NumOutputs> forward(CoopVec<T, NumInputs> inputParams);

        // The gradient storage paired with this module when differentiating with bwd_diff
        This.Differential getDifferential();
    }

    // Placeholder differential for modules without trainable parameters. An empty struct would do from
    // Slang 2026.20 on, but 2026.12 through 2026.19 fail to emit code for an empty IDifferentiablePtrType
    // (shader-slang/slang#13453). The placeholder stays because this module is compiled by the Slang
    // bundled in slangpy (0.43.1 bundles 2026.12), not by the SDK's pinned slangc.
    struct EmptyModuleDifferential : IDifferentiablePtrType
    {
        typealias Differential = EmptyModuleDifferential;
        uint unused;
    }

    // Gradient half of a ModuleChain: the differentials of both chained modules
    struct ModuleChainDifferential<First : IDifferentiablePtrType, Second : IDifferentiablePtrType> : IDifferentiablePtrType
    {
        typealias Differential = ModuleChainDifferential<First, Second>;

        First.Differential first;
        Second.Differential second;
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
        typealias Differential = ModuleChainDifferential<First, Second>;

        [DerivativeMember(ModuleChainDifferential<First, Second>.first)]
        First first;
        [DerivativeMember(ModuleChainDifferential<First, Second>.second)]
        Second second;

        Differential getDifferential()
        {
            return { first.getDifferential(), second.getDifferential() };
        }

        [Differentiable]
        CoopVec<T, NumOutputs> forward(CoopVec<T, NumInputs> inputParams)
        {
            CoopVec<T, NumHidden> middle = first.forward(inputParams);
            return second.forward(middle);
        }
    }

    // Gradient half of a TrainableMLPModule: the differential of its weight and bias buffer
    struct TrainableMLPModuleDifferential : IDifferentiablePtrType
    {
        typealias Differential = TrainableMLPModuleDifferential;

        mlp::MatrixBiasBufferDifferential parameters;
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
        typealias Differential = TrainableMLPModuleDifferential;

        // Weights and biases, wrapped like TrainingMLP does. Their gradients accumulate into
        // derivatives, which getDifferential() pairs with parameters for the backward pass.
        [DerivativeMember(TrainableMLPModuleDifferential.parameters)]
        mlp::MatrixBiasBuffer parameters;
        no_diff mlp::MatrixBiasBufferDifferential derivatives;
        uint matrixOffsets[NumHiddenLayers + 1];
        uint biasOffsets[NumHiddenLayers + 1];

        HiddenAct hiddenAct;
        OutputAct outputAct;

        Differential getDifferential()
        {
            return { derivatives };
        }

        // Slang synthesizes the backward pass from this forward pass; LinearOp supplies the
        // derivative of each linear step and accumulates into the paired gradient buffer.
        [Differentiable]
        CoopVec<T, OutputNeurons> forward(CoopVec<T, InputNeurons> inputParams)
        {
            uint2 layerOffsets[NumHiddenLayers + 1];

            [ForceUnroll]
            for (int i = 0; i <= NumHiddenLayers; ++i)
                layerOffsets[i] = uint2(matrixOffsets[i], biasOffsets[i]);

            return mlp::TrainingMLP<
                T,
                NumHiddenLayers,
                InputNeurons,
                HiddenNeurons,
                OutputNeurons,
                CoopVecMatrixLayout::TrainingOptimal,
                ComponentType
            >::forward_s(inputParams, parameters, layerOffsets, hiddenAct, outputAct);
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
        // Inference only: there are no gradients
        typealias Differential = EmptyModuleDifferential;

        ByteAddressBuffer parameters;
        uint matrixOffsets[NumHiddenLayers + 1];
        uint biasOffsets[NumHiddenLayers + 1];

        HiddenAct hiddenAct;
        OutputAct outputAct;

        Differential getDifferential()
        {
            return { 0 };
        }

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
        // No trainable parameters
        typealias Differential = EmptyModuleDifferential;

        Differential getDifferential()
        {
            return { 0 };
        }

        [Differentiable]
        CoopVec<T, NumScales * NumInputs * 2> forward(CoopVec<T, NumInputs> inputParams)
        {
            return rtxns::EncodeFrequencyN<T, NumInputs, NumScales>(inputParams);
        }
    }
}
