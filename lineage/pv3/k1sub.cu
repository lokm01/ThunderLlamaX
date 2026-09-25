// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) k1sub(
    float* __restrict__ P, const __half* __restrict__ KV, float* __restrict__ ws, const int sp)
{
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < 64) ws[i] = P[i] + (float)sp * 0.0f + (float)KV[0] * 0.0f;
}
