/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Additional functions for use with the CoopVec type.
//
// CoopVec is natively differentiable as of Slang 2026.12, so Slang synthesizes the derivatives of the
// [Differentiable] helpers below; nothing is registered by hand. The HLSL target needs Slang 2026.20 or
// newer for the derivatives of the CoopVec operators (see Known Issues in the README). no_diff constants
// are bound to locals so that no_diff scopes only to the constant. sigmoid (and swish and tanh through
// it) is written in the overflow-free form so that the synthesized derivative stays finite in half
// precision.

namespace rtxns
{

typealias HCoopVec<let K : int> = CoopVec<half, K>;

// Relu: max() is natively differentiable
[Differentiable]
CoopVec<T, K> relu<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    return max(v, no_diff CoopVec<T, K>(T(0.)));
}

// Leaky relu: max(p, 0) + a * min(p, 0)
[Differentiable]
CoopVec<T, K> leakyReLU<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> p, no_diff T a)
{
    var zero = no_diff CoopVec<T, K>(T(0.));
    var slope = no_diff CoopVec<T, K>(a);
    return max(p, zero) + min(p, zero) * slope;
}

// Sigmoid, in the overflow-free form exp(min(v, 0)) / (1 + exp(-|v|)):
//   v >= 0: 1 / (1 + exp(-v))        v < 0: exp(v) / (1 + exp(v))
// The plain 1 / (1 + exp(-v)) overflows exp() for v < -11 in half precision, and the derivative Slang
// synthesizes for it then multiplies inf by 0 and produces NaN, while its 1 / (1 + exp(-v))^2 term
// underflows to zero well before that. Here every exp() argument is <= 0 and the denominator stays in
// (1, 2], so the synthesized derivative is finite and accurate over the whole half range.
[Differentiable]
CoopVec<T, K> sigmoid<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    var zero = no_diff CoopVec<T, K>(T(0.));
    var c1 = no_diff CoopVec<T, K>(T(1.));
    var t = exp(min(v, zero));   // exp(v) for v < 0, else 1
    var u = exp(min(-v, zero));  // exp(-v) for v > 0, else 1;  t * u = exp(-|v|)
    return t / (c1 + t * u);
}

// Shifted exponential: exp(v) - 1
[Differentiable]
CoopVec<T, K> shiftedExp<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    var c1 = no_diff CoopVec<T, K>(T(1.));
    return exp(v) - c1;
}

// Swish: v * sigmoid(v)
[Differentiable]
CoopVec<T, K> swish<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    return v * sigmoid(v);
}

// Tanh: 2 * sigmoid(2v) - 1, inheriting sigmoid's overflow-free evaluation
[Differentiable]
CoopVec<T, K> tanhCoopVec<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    var c1 = no_diff CoopVec<T, K>(T(1.));
    var c2 = no_diff CoopVec<T, K>(T(2.));
    return sigmoid(v * c2) * c2 - c1;
}

// copysign function implementation
CoopVec<T, K> copysign<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> a, CoopVec<T, K> b)
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        a[i] = copysign(a[i], b[i]);
    }
    return a;
}

// abs functions implementation
CoopVec<T, K> abs<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
     [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        v[i] = abs(v[i]);
    }
    return v;
}

// Scale: x * s
[Differentiable]
CoopVec<T, K> scale<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> x, no_diff T s)
{
    var cs = no_diff CoopVec<T, K>(s);
    return x * cs;
}

}
