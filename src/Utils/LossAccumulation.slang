/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

module LossAccumulation;

public static const uint LOSS_ACCUM_MAX_COMPONENTS = 128;

 // Configuration 
 public struct LossConfig
 {
    uint componentCount;
    uint baseByteOffset;

    public uint GetComponentCount()
    { 
        return min(componentCount, LOSS_ACCUM_MAX_COMPONENTS);
    }
 }

public LossConfig MakeLossConfig(uint requestedComponents, uint baseByteOffset)
{
    LossConfig config;
    
    uint components = requestedComponents;
    components = clamp(components, 1u, LOSS_ACCUM_MAX_COMPONENTS);

    config.componentCount = components;
    config.baseByteOffset = baseByteOffset;
    return config;
}

// CAS-based atomic float add on RWByteAddressBuffer.
float AtomicAddFloat(RWByteAddressBuffer buffer, uint byteOffset, float valueToAdd)
{
    uint expected = buffer.Load(byteOffset);

    while (true)
    {
        float oldVal = asfloat(expected);
        float newVal = oldVal + valueToAdd;
        uint desired = asuint(newVal);

        uint original;
        buffer.InterlockedCompareExchange(byteOffset, expected, desired, original);

        if (original == expected)
        {
            // Success — return the previous value
            return oldVal;
        }

        // Retry with latest observed value
        expected = original;
    }
}

public float AtomicAddF32Portable(
    RWByteAddressBuffer buffer,
    uint byteOffset,
    float value)
{
#if defined(DXIL) && !defined(NVAPI_INIT)
    return AtomicAddFloat(buffer, byteOffset, value);
#else
    float oldValue;
    buffer.InterlockedAddF32(byteOffset, value, oldValue);
    return oldValue;
#endif
}

public void Zero<let N : int>(out float components[N])
{
    [unroll]
    for (int i = 0; i < N; i++)
        components[i] = 0.f;
}

public void AddInPlace<let N : int>(
    inout float dst[N],
    float src[N],
    uint componentCount)
{
    [unroll]
    for (int i = 0; i < N; i++)
    {
        if (i >= componentCount) 
            break;
        dst[i] += src[i];
    }
}

public void AccumulateComponents<let N : int>(
    RWByteAddressBuffer buffer,
    float components[N],
    LossConfig config)
{
    [unroll]
    for (int i = 0; i < N; i++)
    {
        if (i >= config.GetComponentCount())
            break;
        
        uint byteOffset = config.baseByteOffset + uint(i) * 4u;
        AtomicAddF32Portable(buffer, byteOffset, components[i]);
    }
}