// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// v9: SIX params (pad), store to param-3 (sx) — tests arg-count layout effect
#include <cuda_fp16.h>
#define NWARP 8
#define NTHR (NWARP * 32)
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const __half* __restrict__ x, signed char* __restrict__ xq,
    float* __restrict__ sx, int* __restrict__ rs,
    float* __restrict__ d5, float* __restrict__ d6) {
  const int m = blockIdx.x;
  if (threadIdx.x == 0) sx[(size_t)m * 40] = (float)m;
}
