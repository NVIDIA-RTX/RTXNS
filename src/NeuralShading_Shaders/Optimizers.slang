/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define ADAM_BETA1                               0.9f
#define ADAM_BETA2                               0.999f
#define ADAM_EPSILON                             1E-8f

namespace optimizers
{
    float sanitize(float x, float fallback = 0.0f)
    {
        return (isnan(x) || isinf(x)) ? fallback : x;
    }

    // Common interface for buffer accessors
    interface IBufferAccessor
    {
        float get(uint index);
        void set(uint index, float value);
    };

    // Accessor for RWBuffer
    struct RWBufferAccessor : IBufferAccessor
    {
        RWBuffer<float> m_buffer;
        
        __init(RWBuffer<float> buffer)
        {
            m_buffer = buffer;
        }

        float get(uint index)
        {
            return m_buffer[index];
        }

        void set(uint index, float value)
        {
            m_buffer[index] = value;
        }
    };

    // Common interface for optimizers
    interface IOptimizer
    {
        float step(float weightBias, uint parameterID, float gradient, const float currentStep);
    };

    // Adam optimizer using a generic buffer accessor
    // This allows using different buffer types (e.g., RWBuffer, RWNDBuffer
    struct AdamAccessor<T : IBufferAccessor> : IOptimizer
    {
        T m_moments1;
        T m_moments2;
        float m_learningRate;
        float m_lossScale;
        float m_beta1;
        float m_beta2;
        float m_epsilon;

        // Initializes from two moments buffers and optimizations parameters
        __init(
            T moments1, 
            T moments2,
            float learningRate, 
            float lossScale,
            float beta1 = ADAM_BETA1,
            float beta2 = ADAM_BETA2,
            float epsilon = ADAM_EPSILON)
        {
            m_moments1 = moments1;
            m_moments2 = moments2;
            m_learningRate = learningRate;
            m_lossScale = lossScale;
            m_beta1 = beta1;
            m_beta2 = beta2;
            m_epsilon = epsilon;
        }

        // Optimization step for one MLP parameter
        float step(in float weightBias, uint parameterID, float gradient, const float currentStep)
        {
            gradient /= m_lossScale;
            gradient = sanitize(gradient);
            
            float gradient_sq = gradient * gradient;
            float moment1 = m_moments1.get(parameterID) * m_beta1 + gradient * (1 - m_beta1);
            float moment2 = m_moments2.get(parameterID) * m_beta2 + gradient_sq * (1 - m_beta2);

            float bias_correction1 = 1 - pow(m_beta1, (float) currentStep);
            float bias_correction2 = 1 - pow(m_beta2, (float) currentStep);

            float denom = sqrt(moment2) * rsqrt(bias_correction2) + m_epsilon;
            float step_size = m_learningRate / bias_correction1;

            float adjustedWeightbias = weightBias - (moment1 / denom) * step_size;

            m_moments1.set(parameterID, moment1);
            m_moments2.set(parameterID, moment2);

            return adjustedWeightbias;
        }
    };

    // Adam optimizer using RWBuffer specifically
    struct Adam : IOptimizer
    {
        AdamAccessor<RWBufferAccessor> m_accessor;

        __init(
            RWBuffer<float> moments1, 
            RWBuffer<float> moments2,
            float learningRate, 
            float lossScale,
            float beta1 = ADAM_BETA1,
            float beta2 = ADAM_BETA2,
            float epsilon = ADAM_EPSILON)
        {
            m_accessor = AdamAccessor<RWBufferAccessor>(RWBufferAccessor(moments1), RWBufferAccessor(moments2), learningRate, lossScale, beta1, beta2, epsilon);
        }

        float step(in float weightBias, uint parameterID, float gradient, const float currentStep)
        {
            return m_accessor.step(weightBias, parameterID, gradient, currentStep);
        }
    };
};