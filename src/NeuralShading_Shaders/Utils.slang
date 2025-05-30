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

namespace rtxns
{
    ////////////////////////
    //
    // Encoders for input parameters
    // Using encoding of input parameters often significantly improves MLP convergence speed and results quality
    //
    ////////////////////////

    // Frequency encoding expands input count 
    static const int FREQUENCY_ENCODING_COUNT = 6;  

    // Encode every input as sequence sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x), sin(4*pi*x), cos(4*pi*x), ...
    // The output parameter count must equal 6 * PARAMS_COUNT
    CoopVec<T, PARAMS_COUNT * FREQUENCY_ENCODING_COUNT> EncodeFrequency<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
    {
        var output = CoopVec<T, PARAMS_COUNT * FREQUENCY_ENCODING_COUNT>(T(1.));

        [ForceUnroll] 
        for (int i = 0; i < PARAMS_COUNT; ++i)
        {
            float sn, cn;
            sincos(parameters[i] * 3.14159265358979323846, sn, cn);

            int i1 = i * FREQUENCY_ENCODING_COUNT;
            output[i1 + 0] = T(sn);
            output[i1 + 1] = T(cn);
            output[i1 + 2] = T(2.) * output[i1 + 0] * output[i1 + 1];
            output[i1 + 3] = T(2.) * output[i1 + 1] * output[i1 + 1] - T(1.);
            output[i1 + 4] = T(2.) * output[i1 + 2] * output[i1 + 3];
            output[i1 + 5] = T(2.) * output[i1 + 3] * output[i1 + 3] - T(1.);
        }

        return output;
    }

    // Generic version that allows arbitrary number of scales
    [Differentiable]
    CoopVec<T, NUM_SCALES * PARAMS_COUNT * 2> EncodeFrequencyN<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int, let NUM_SCALES : int>(CoopVec<T, PARAMS_COUNT> inputParams)
    {
        var inputVec = VectorFromCoopVec(inputParams);
        float[NUM_SCALES * PARAMS_COUNT * 2] outputVec;

        [ForceUnroll]
        for (int i = 0; i < PARAMS_COUNT; ++i)
        {
            const int base = i * NUM_SCALES * 2;

            float sn, cn;
            sincos(inputVec[i] * 3.141592653589793238f, sn, cn);
            outputVec[base + 0] = sn;
            outputVec[base + 1] = cn;
            [ForceUnroll]
            for (int j = 1; j < NUM_SCALES; ++j) {
                sn = 2.0f * sn * cn;
                cn = 2.0f * cn * cn - 1.0f;
                outputVec[base + j * 2 + 0] = sn;
                outputVec[base + j * 2 + 1] = cn;
            }
        }

        return CoopVecFromArray<T, NUM_SCALES * PARAMS_COUNT * 2>(outputVec);
    }

    // Triangle encoding expands input count 
    static const int TRIANGLE_ENCODING_COUNT = 6;  

    // Similar to frequence encoder, but use triangle waves to improve performance
    // The output parameter count must equal 6 * PARAMS_COUNT
    CoopVec<T, PARAMS_COUNT * TRIANGLE_ENCODING_COUNT> EncodeTriangle<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
    {
        var output = CoopVec<T, PARAMS_COUNT * TRIANGLE_ENCODING_COUNT>(T(1.));

        [ForceUnroll] 
        for (int i = 0; i < PARAMS_COUNT; ++i)
        {
            int i1 = i * TRIANGLE_ENCODING_COUNT;
            float p = parameters[i];
            float s = 0.5f, k = 0.f;
            [ForceUnroll] for (int j = 0; j < TRIANGLE_ENCODING_COUNT; ++j)
            {
                float r = p * s + k;
                output[i1 + j] = T(abs(r - floor(r) - 0.5f) * 4.f - 1.f);
                s *= 2.f;
                k += 0.25f;
            }
        }

        return output;
    }

    ////////////////////////
    //
    // Packing functions 
    //
    ////////////////////////

    // Unpacks a uint4 array to a uint array 
    // Used for storing weight and bias buffer offsets 
    uint[NUM_UNPACKED] UnpackArray<let NUM_PACKED4 : int, let NUM_UNPACKED : int>(uint4 ps[NUM_PACKED4])
    {
        uint res[NUM_UNPACKED];
        
        [ForceUnroll]
        for(int i = 0; i < NUM_UNPACKED; ++i)
            res[i] = ps[i/4][i%4];
        
        return res;
    }

    ////////////////////////
    //
    // Conversion functions 
    //
    ////////////////////////

    // Construct a CoopVec from a float array
    [BackwardDerivative(CoopVecFromArray_Backward)]
    CoopVec<T, PARAMS_COUNT> CoopVecFromArray<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
    {
        var output = CoopVec<T, PARAMS_COUNT>(T(0.));
        
        [ForceUnroll] 
        for(int i = 0; i < PARAMS_COUNT; i++)
        {
            output[i] = T(parameters[i]);
        }
        return output;
    }
    
    void CoopVecFromArray_Backward<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(inout DifferentialPair<float[PARAMS_COUNT]> parameters, CoopVec<T, PARAMS_COUNT> grad)
    {
        float gradArray[PARAMS_COUNT];

        [ForceUnroll]
        for (int i = 0; i < PARAMS_COUNT; ++i)
        {
            gradArray[i] = grad[i].toFloat();
        }
        parameters = diffPair(parameters.p, gradArray);
    }

    // Construct a CoopVec from a float vector of up to 4 parameters
    [BackwardDerivative(CoopVecFromVector_Backward)]
    CoopVec<T, PARAMS_COUNT> CoopVecFromVector<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(vector<float, PARAMS_COUNT> parameters)
    {
        var output = CoopVec<T, PARAMS_COUNT>(T(0.));
        
        [ForceUnroll] 
        for(int i = 0; i < PARAMS_COUNT; i++)
        {
            output[i] = T(parameters[i]);
        }
        return output;
    }
    
    void CoopVecFromVector_Backward<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(inout DifferentialPair<vector<float, PARAMS_COUNT>> parameters, CoopVec<T, PARAMS_COUNT> grad)
    {
        vector<float, PARAMS_COUNT> gradVec;

        [ForceUnroll]
        for (int i = 0; i < PARAMS_COUNT; ++i)
        {
            gradVec[i] = grad[i].toFloat();
        }
        parameters = diffPair(parameters.p, gradVec);
    }

    // Convert a CoopVec to a float vector
    [BackwardDerivative(VectorFromCoopVec_Backward)]
    vector<float, PARAMS_COUNT> VectorFromCoopVec<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(CoopVec<T, PARAMS_COUNT> parameters)
    {
        vector<float, PARAMS_COUNT> result;

        [ForceUnroll]
        for (int i = 0; i < PARAMS_COUNT; ++i)
        {
            result[i] = parameters[i].toFloat();
        }
        return result;
    }
    void VectorFromCoopVec_Backward<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(inout DifferentialPair<CoopVec<T, PARAMS_COUNT>> parameters, vector<float, PARAMS_COUNT> grad)
    {
        CoopVec<T, PARAMS_COUNT> gradParams;

        [ForceUnroll]
        for (int i = 0; i < PARAMS_COUNT; ++i)
        {
            gradParams[i] = T(grad[i]);
        }

        parameters = diffPair(parameters.p, gradParams);
    }
}
