// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pv8kh(
    float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2, float* __restrict__ data3,
    float* __restrict__ data4, float* __restrict__ data5, float* __restrict__ data6,
    float* __restrict__ data7, float* __restrict__ data8, float* __restrict__ data9,
    float* __restrict__ data10, float* __restrict__ data11, const int dummy)
{
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x + (blockIdx.z << 7);
  const float* P  = (r == 0) ? data8 : (r == 1) ? data5 : data1;
  const float  mx = (r == 0) ? data9[g]  : (r == 1) ? data6[g]  : data2[g];
  const float  rs = (r == 0) ? data10[g] : (r == 1) ? data7[g] : data3[g];
  const float* P_g = P + ((size_t)g << 13);
  const __half* V_g = (const __half*)data4 + (((size_t)(g / 6)) << 21) + 8388608;
  float acc = 0.0f;
  for (int p = 0; p < 8192; ++p)
    acc += exp2f((P_g[p] - mx) * 1.4426950216293334f) * __half2float(V_g[((size_t)p << 8) + d]);
  const float gate = data11[d + ((size_t)g << 9) + 256 + (size_t)r * 12288];
  data0[d + ((size_t)g << 8) + (size_t)r * 6144] =
    (acc / rs) * (1.0f / (1.0f + exp2f(-gate * 1.4426950216293334f)));
}
