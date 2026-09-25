// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
#ifndef LMAX
#define LMAX 100352
#endif
#ifndef PS
#define PS 100349
#endif
#ifndef S
#define S 24
#endif
#define C (((LMAX + S - 1) / S))
extern "C" __global__ void __launch_bounds__(256) k1sub2(
    float* __restrict__ P, const __half* __restrict__ KV, float* __restrict__ ws, const int sp)
{
  const int s = blockIdx.x >> 2;
  const int g = blockIdx.x & 3;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int start = s * C;
  const int p = start + warp;                 // tile 0, position = warp
  const float pv = P[(size_t)(g * 6 + 0) * PS + p];
  ws[((size_t)(s * 4 + g) * 6 + 0) * 256 + lane] = pv + (float)sp * 0.0f;
}
