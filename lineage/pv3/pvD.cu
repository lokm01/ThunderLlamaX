// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pvD(
    float* __restrict__ out, float* __restrict__ P, float* __restrict__ mx, float* __restrict__ sm,
    half* __restrict__ V, float* __restrict__ gate, const int dummy)
{
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x + (blockIdx.z << 7);
  const float si = sm[g*3 + r];
  const float* PE_g = P + ((size_t)g * 301056) + ((size_t)r * 100352);
  const __half* V_g = (const __half*)V + (((size_t)(g/6)) * 25690112LL) + 102760448LL;
  float acc = 0.0f;
  for (int p = 0; p < 100352; ++p)
    acc += PE_g[p] * __half2float(V_g[((size_t)p << 8) + d]);
  const float gt = gate[d + ((size_t)g << 9) + 256 + (size_t)r * 12288];
  out[d + ((size_t)g << 8) + (size_t)r * 6144] = (acc / si) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));
}
