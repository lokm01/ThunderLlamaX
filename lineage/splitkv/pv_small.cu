// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pv_small(
    float* __restrict__ data0, const __half* __restrict__ data4,
    float* __restrict__ data11, const int dummy)
{
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x + (blockIdx.z << 7);
  if (r == 0 && g == 0 && d == 0) data0[0] = 777.0f;
  const __half* V_g = data4 + ((size_t)(g / 6) << 21) + 8388608;
  float acc = 0.0f;
  for (int p = 0; p < 64; ++p) acc += __half2float(V_g[((size_t)p << 8) + d]);
  const float gate = data11[d + ((size_t)g << 9) + 256 + (size_t)r * 12288];
  data0[d + ((size_t)g << 8) + (size_t)r * 6144] = acc * (1.0f/(1.0f+exp2f(-gate*1.4426950216293334f)));
}
