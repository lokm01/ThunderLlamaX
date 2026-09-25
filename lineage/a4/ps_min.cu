// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) ps_min(float* __restrict__ a, float* __restrict__ b, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i] + 1.0f;
}
