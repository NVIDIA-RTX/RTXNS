// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;

// Helper function to turn pixel coordinates into normalized UVs.
float2 pixel_to_uv(int2 pixel, int2 resolution)
{
    return (float2(pixel) + 0.5f) / float2(resolution);
}

float3 tonemap_aces_film(float3 input)
{
    float3 x = input.xyz;
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    float3 col = saturate((x * (a * x + b)) / (x * (c * x + d) + e));
    return col;
}

void blit(
    int2 pixel,
    int2 size,
    int2 offset,
    bool tonemap,
    bool bilinear,
    Tensor<float3, 2> input,
    RWTexture2D<float4> output,
)
{
    uint2 output_dimensions;
    output.GetDimensions(output_dimensions.x, output_dimensions.y);

    float2 uv = pixel_to_uv(pixel, size);

    uint[2] shape = input.shape;

    float3 col = bilinear ?
        input.sample(float2(uv.x, uv.y)) :
        input.getv(uint2(shape[1]*uv.x, shape[0]*uv.y));

    col = abs(col);

    if(tonemap)
    {
        col = tonemap_aces_film(col);
    }

    if(any(isnan(col.x)))
    {
        output[offset + pixel] = float4(1,0,1,1);
    }
    else
    {
        output[offset + pixel] = float4(col, 1.0f);

    }

}
