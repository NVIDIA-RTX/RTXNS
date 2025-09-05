/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define INPUT_FEATURES 2
#define INPUT_NEURONS (INPUT_FEATURES * 6) // Frequency encoding increases the input by 6 for each input
#define OUTPUT_NEURONS 3

#define HIDDEN_NEURONS 64
#define NUM_HIDDEN_LAYERS 4

#define BASE_LEARNING_RATE 0.001f
#define MIN_LEARNING_RATE 0.0001f
#define WARMUP_LEARNING_STEPS 0
#define FLAT_LEARNING_STEPS 1000000
#define DECAY_LEARNING_STEPS 1000000

#define NUM_TRANSITIONS (NUM_HIDDEN_LAYERS + 1)
#define NUM_TRANSITIONS_ALIGN4 ((NUM_TRANSITIONS + 3) / 4)
#define LOSS_SCALE 1024.0
#define RELU_LEAK 0.01h

#define VECTOR_FORMAT half
#define TYPE_INTERPRETATION CoopVecComponentType::Float16
#define NETWORK_PRECISION rtxns::Precision::F16

#define MATRIX_LAYOUT CoopVecMatrixLayout::TrainingOptimal

#define BATCH_COUNT 128
#define BATCH_SIZE_X 32
#define BATCH_SIZE_Y 32

enum class NetworkTransform
{
    Identity,
    Zoom,
    Flip
};

struct NeuralConstants
{
    uint4 weightOffsets[NUM_TRANSITIONS_ALIGN4];
    uint4 biasOffsets[NUM_TRANSITIONS_ALIGN4];

    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t maxParamSize;
    float learningRate;

    uint32_t currentStep;
    uint32_t batchSizeX;
    uint32_t batchSizeY;
    NetworkTransform networkTransform;
};