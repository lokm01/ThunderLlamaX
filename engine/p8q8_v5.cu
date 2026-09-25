// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// v5: ONE store to sx only
#include <cuda_fp16.h>
#define NWARP 8
#define NTHR (NWARP * 32)
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const __half* __restrict__ x, signed char* __restrict__ xq,
    float* __restrict__ sx, int* __restrict__ rs) {
  const int m = blockIdx.x;
  if (threadIdx.x == 0) sx[(size_t)m * 40] = (float)m;
}
