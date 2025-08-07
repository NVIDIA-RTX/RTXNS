#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

#define PI 3.14159265358979323846f

struct NetworkParameters<int Inputs, int Outputs, int WeightReg, int BiasReg>
{
    static const CoopVecComponentType ComponentType = CoopVecComponentType.Float16;

    ByteAddressBuffer weights : REGISTER_SRV(WeightReg, 0);
    ByteAddressBuffer biases : REGISTER_SRV(BiasReg, 0);

    CoopVec<half, Outputs> forward(CoopVec<half, Inputs> x)
    {
        return coopVecMatMulAdd<half, Outputs>(
            x, ComponentType,
            weights, 0, ComponentType,
            biases, 0, ComponentType,
            CoopVecMatrixLayout.InferencingOptimal, false, 0
        );
    }
}

struct Network {
    NetworkParameters<16, 32, 0, 1> layer0;
    NetworkParameters<32, 32, 2, 3> layer1;
    NetworkParameters<32, 3, 4, 5>  layer2;

    float3 eval(no_diff float2 uv)
    {
        CoopVec<half, 16> inputs;
        [ForceUnroll]
        for (int i = 0; i < 4; ++i) {
            float scale = float(2 << i);
            inputs[i * 4 + 0] = half(sin(uv.x * PI * scale));
            inputs[i * 4 + 1] = half(cos(uv.x * PI * scale));
            inputs[i * 4 + 2] = half(sin(uv.y * PI * scale));
            inputs[i * 4 + 3] = half(cos(uv.y * PI * scale));
        }

        var output0 = layer0.forward(inputs);
        output0 = leakyReLU(output0);
        var output1 = layer1.forward(output0);
        output1 = leakyReLU(output1);
        var output2 = layer2.forward(output1);
        output2 = exp(output2);
        return float3(output2[0], output2[1], output2[2]);
    }
}

CoopVec<half, N> activation<int N>(CoopVec<half, N> x)
{
    return max(x, CoopVec<half, N>(0.0h));
}

CoopVec<half, N> leakyReLU<int N>(CoopVec<half, N> x)
{
    return max(x, CoopVec<half, N>(0.0h)) + min(x, CoopVec<half, N>(0.0h)) * 0.01h;
}

// Render full res BRDF from given inputs.
float3 inference(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}

float3 loss(int2 pixel, int2 resolution, no_diff float3 reference, Network network)
{
    float3 color = inference(pixel, resolution, network);
    float3 error = color - reference;
    return 1 - abs(error);
}

DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
RWTexture2D<float4> inferenceTexture : REGISTER_UAV(0, 0);
RWTexture2D<float4> lossTexture : REGISTER_UAV(1, 0);

Texture2D<float4> inputTexture : REGISTER_SRV(0, 0);

Network network;

[shader("compute")]
[numthreads(8, 8, 1)]
void inference_cs(uint3 pixel: SV_DispatchThreadID)
{
    inferenceTexture[pixel.xy].rgb = inference(pixel.xy, gConst.resolution, network);
    lossTexture[pixel.xy].rgb = loss(pixel.xy, gConst.resolution, inputTexture[pixel.xy].rgb, network);
}
