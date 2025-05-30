// SPDX-License-Identifier: Apache-2.0
// clang-format off

/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

__exported import NeuralModules;
__exported import Loss;
__exported import Optimizers;

struct RNG
{
    uint state;

    __init(uint state) { this.state = state; }

    [mutating]
    float next()
    {
        float r = (state >> 8) * 0x1p-24;
        state = state * 2739110765U + 2739110765U;
        return r;
    }
}

// An example of adding a custom activation to your network
// This implements the Sigmoid Linear Unit (SiLU)
struct SiLUActivation<T : __BuiltinFloatingPointType, let K : int> : rtxns::mlp::IActivation<T, K>
{
    [Differentiable]
    CoopVec<T, K> eval(CoopVec<T, K> x)
    {
        return x * no_diff CoopVec<T, K>(T(1.)) / (no_diff CoopVec<T, K>(T(1.)) + exp(-x));
    }
}

// Utility function for nearest-neighbor sampling of texture
T SampleTexture<T : ITexelElement>(Texture2D<T> tex, float2 uv)
{
    float2 size;
    tex.GetDimensions(size[0], size[1]);
    uint2 xy = uint2(uv * size);
    return tex[xy];
}

// Take one step with the adam optimizer
void OptimizerStep(
    RWBuffer<float> moments1,
    RWBuffer<float> moments2,
    RWBuffer<float> paramF,
    RWBuffer<half> paramH,
    RWBuffer<half> grad,
    uint idx,
    float learningRate,
    float gradScale,
    int iteration)
{
    var optimizer = optimizers::Adam(moments1, moments2, learningRate, gradScale);

    // Parameters are converted to FP16 for computing gradients,
    // but we keep the FP32 originals around so we don't accumulate
    // rounding errors
    float parameter = paramF[idx];
    float gradient = (float)grad[idx];

    parameter = optimizer.step(parameter, idx, gradient, iteration);

    // Update the reference FP32 parameter, and convert the new value back to FP16
    paramF[idx] = parameter;
    paramH[idx] = (half)parameter;
    // Zero out gradients
    grad[idx] = 0.0h;
}

void TrainTexture<Model : rtxns::IModule<half, 2, 3>, Loss : rtxns::mlp::ILoss<float, 3>>(Model model, inout RNG rng, Texture2D<float4> targetTex, float lossScale)
{
    // Get a random uv coordinate for the input
    float2 inputUV = clamp(float2(rng.next(), rng.next()), 0.0, 1.0);

    // Sample the target texture at the generated UV
    float3 targetRGB = SampleTexture(targetTex, inputUV).rgb;

    // Evaluate the current output of the model
    float3 predictedRGB = EvalModel(model, inputUV);

    // Evaluate the loss gradient
    float3 lossGradient = Loss.deriv(targetRGB, predictedRGB, lossScale);

    // Backpropragate gradient through network parameters
    bwd_diff(EvalModel)(model, inputUV, lossGradient);
}

// Convenience functions for evaluating the model from vector inputs
// Converts to/from CoopVec internally
[Differentiable]
float3 EvalModel<Model: rtxns::IModule<half, 2, 3>>(Model model, no_diff float2 inputUV)
{
    var inputVec = rtxns::CoopVecFromVector<half>(inputUV);

    var result = model.forward(inputVec);

    return rtxns::VectorFromCoopVec(result);
}

// Computes the loss between the predicted RGB at a given UV coordinate and a reference texture
float3 EvalLoss<Loss : rtxns::mlp::ILoss<float, 3>>(float2 inputUV, float3 predictedRGB, Texture2D<float4> targetTex)
{
    float3 targetRGB = SampleTexture(targetTex, inputUV).rgb;

    return Loss.value(targetRGB, predictedRGB, 1.0f);
}

// Computes the difference between the predicted RGB at a given UV coordinate and a reference texture
// for visualization
float3 TextureDifference(float2 inputUV, float3 predictedRGB, Texture2D<float4> targetTex, float scale)
{
    float3 targetRGB = SampleTexture(targetTex, inputUV).rgb;

    return (predictedRGB - targetRGB) * scale + 0.5f;
}

// Convenience function to convert from half to float params
float ConvertToFloat(half paramH)
{
    return (float)paramH;
}
