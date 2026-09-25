// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// v2: half2 loads + ONE guarded store (no shfl, no rint)
#include <cuda_fp16.h>
#define NWARP 8
#define NTHR (NWARP * 32)
#define KDIM 5120
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const __half* __restrict__ x, signed char* __restrict__ xq,
    float* __restrict__ sx, int* __restrict__ rs) {
  const int m = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int c = warp * 5;
  const int k0 = c * 128 + lane * 4;
  const __half2 h01 = *(const __half2*)(x + (size_t)m * KDIM + k0);
  const __half2 h23 = *(const __half2*)(x + (size_t)m * KDIM + k0 + 2);
  float f = __half2float(__low2half(h01)) + __half2float(__high2half(h23));
  if (threadIdx.x == 0) sx[(size_t)m * 40] = f;
}
