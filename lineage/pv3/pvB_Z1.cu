// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) pvB(
    float* __restrict__ out, float* __restrict__ P, float* __restrict__ mx, float* __restrict__ sm,
    half* __restrict__ V, float* __restrict__ gate, const int dummy)
{
  const int s = blockIdx.x, r = blockIdx.z, d = threadIdx.x + (blockIdx.y * 256);
  float si[6], acc[6];
  #pragma unroll
  for (int j = 0; j < 6; j++) {
    const int h = s*6 + j;
    si[j] = sm[h*3 + r]; acc[j] = 0.0f;
  }
  const __half* Vg = (const __half*)V + ((size_t)s * 25690112) + 102760448;
  for (int p = 0; p < 100352; ++p) {
    const float v = __half2float(Vg[((size_t)p << 8) + d]);
    #pragma unroll
    for (int j = 0; j < 6; j++)
      acc[j] += __ldg(P + ((size_t)(s*6+j) * 301056) + ((size_t)r * 100352) + p) * v;
  }
  #pragma unroll
  for (int j = 0; j < 6; j++) {
    const int h = s*6 + j;
    const float gt = gate[d + ((size_t)h << 9) + 256 + (size_t)r * 12288];
    out[d + ((size_t)h << 8) + (size_t)r * 6144] = (acc[j] / si[j]) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));
  }
}
