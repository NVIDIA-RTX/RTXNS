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
    // Common interface for optimizers
    interface IOptimizer
    {
        float step(float weightBias, uint parameterID, float gradient, const float currentStep);
    };

    // Adam optimizer 
    struct Adam : IOptimizer
    {
        RWStructuredBuffer<float> m_moments1;
        RWStructuredBuffer<float> m_moments2;
        float m_learningRate;
        float m_lossScale;
        float m_beta1;
        float m_beta2;
        float m_epsilon;

        // Initializes from two moments buffers and optimizations parameters
        __init(
            RWStructuredBuffer<float> moments1, 
            RWStructuredBuffer<float> moments2,
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
            float gradient_sq = gradient * gradient;
            float moment1 = m_moments1[parameterID] * m_beta1 + gradient * (1 - m_beta1);
            float moment2 = m_moments2[parameterID] * m_beta2 + gradient_sq * (1 - m_beta2);

            float bias_correction1 = 1 - pow(m_beta1, (float) currentStep);
            float bias_correction2 = 1 - pow(m_beta2, (float) currentStep);

            float denom = sqrt(moment2) * rsqrt(bias_correction2) + m_epsilon;
            float step_size = m_learningRate / bias_correction1;

            float adjustedWeightbias = weightBias - (moment1 / denom) * step_size;

            m_moments1[parameterID] = moment1;
            m_moments2[parameterID] = moment2;
            return adjustedWeightbias;
        }
    };
};