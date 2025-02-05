/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define MAX_LAYER_COUNT 8
#define MAX_LAYER_COUNT_ALIGN4 ((MAX_LAYER_COUNT + 3) / 4)

// These defines will be overriden by texture-training.py with the
// chosen network architecture. However, if we compile this file
// from scratch, we provide a default architexture here so the sample
// runs. We provide the trained weights for this network under
// assets/data/slangpy-weights.json
#ifndef MODEL_TYPE
#define MODEL_TYPE                                                                                                                                                                 \
    rtxns::ModuleChain<half, 2, 12, 3, rtxns::FrequencyEncoding<half, 2, 3>,                                                                                                       \
                       rtxns::InferenceMLPModule<half, 4, 12, 32, 3, CoopVecComponentType::Float16, rtxns::mlp::LeakyReLUAct<half, 32>, rtxns::mlp::SigmoidAct<half, 3>>>

#define MODEL_INITIALIZER                                                                                                                                                          \
    {                                                                                                                                                                              \
        {},                                                                                                                                                                        \
        {                                                                                                                                                                          \
            weights, { wo[0], wo[1], wo[2], wo[3], wo[4] }, { bo[0], bo[1], bo[2], bo[3], bo[4] }, { 0.01h },                                                                      \
            {                                                                                                                                                                      \
            }                                                                                                                                                                      \
        }                                                                                                                                                                          \
    }
#define VECTOR_FORMAT half
#endif

struct NeuralConstants
{
    uint4 weightOffsets[MAX_LAYER_COUNT_ALIGN4];
    uint4 biasOffsets[MAX_LAYER_COUNT_ALIGN4];

    uint32_t imageWidth;
    uint32_t imageHeight;
};
