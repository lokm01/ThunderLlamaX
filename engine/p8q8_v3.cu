// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// v3: shfl tree ONLY (no loads) + guarded store
#include <cuda_fp16.h>
#define NWARP 8
#define NTHR (NWARP * 32)
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const __half* __restrict__ x, signed char* __restrict__ xq,
    float* __restrict__ sx, int* __restrict__ rs) {
  const int m = blockIdx.x;
  float am = (float)(threadIdx.x & 31);
  _Pragma("unroll")
  for (int o = 16; o > 0; o >>= 1)
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  if (threadIdx.x == 0) sx[(size_t)m * 40] = am;
}
