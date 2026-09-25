// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pv_mid(
    float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2, float* __restrict__ data3,
    float* __restrict__ data4, float* __restrict__ data5, float* __restrict__ data6,
    float* __restrict__ data7, float* __restrict__ data8, float* __restrict__ data9,
    float* __restrict__ data10, float* __restrict__ data11, const int dummy)
{
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x + (blockIdx.z << 7);
  const __half* V_g = (const __half*)data4 + ((size_t)(g / 6) << 20) + 8388608;
  float acc = 0.0f;
  for (int p = 0; p < 64; ++p) acc += __half2float(V_g[((size_t)p << 8) + d]);
  const float gate = data11[d + ((size_t)g << 9) + 256 + (size_t)r * 12288];
  data0[d + ((size_t)g << 8) + (size_t)r * 6144] = acc * (1.0f/(1.0f+exp2f(-gate*1.4426950216293334f)));
}
