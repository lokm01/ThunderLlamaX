// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pvA(
    float* __restrict__ out, float* __restrict__ P, float* __restrict__ mx, float* __restrict__ sm,
    half* __restrict__ V, float* __restrict__ gate, const int dummy)
{
  const int s = blockIdx.x, d = threadIdx.x + (blockIdx.y * 128);
  float m[3][6], si[3][6], acc[3][6];
  #pragma unroll
  for (int r = 0; r < 3; r++) {
    #pragma unroll
    for (int j = 0; j < 6; j++) {
      const int h = s*6 + j;
      m[r][j] = mx[h*3 + r]; si[r][j] = sm[h*3 + r]; acc[r][j] = 0.0f;
    }
  }
  const float* PE = P;
  const __half* Vg = (const __half*)V + ((size_t)s * 25690112) + 102760448;
  for (int p = 0; p < 100352; ++p) {
    const float v = __half2float(Vg[((size_t)p << 8) + d]);
    #pragma unroll
    for (int r = 0; r < 3; r++)
      #pragma unroll
      for (int j = 0; j < 6; j++)
        acc[r][j] += __ldg(PE + ((size_t)(s*6+j) * 301056) + ((size_t)r * 100352) + p) * v;
  }
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int j = 0; j < 6; j++) {
      const int h = s*6 + j;
      const float gt = gate[d + ((size_t)h << 9) + 256 + (size_t)r * 12288];
      out[d + ((size_t)h << 8) + (size_t)r * 6144] = (acc[r][j] / si[r][j]) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));
    }
}
