// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// wB: 6 params; shfl tree + rint + store to param-3 (NO loads)
#include <cuda_fp16.h>
#define NWARP 8
#define NTHR (NWARP * 32)
#define KDIM 5120
#define NCH (KDIM/128)
#define CPC (NCH/NWARP)
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const __half* __restrict__ x, signed char* __restrict__ xq,
    float* __restrict__ sx, int* __restrict__ rs,
    const float* __restrict__ pad5, const float* __restrict__ pad6) {
  (void)pad5; (void)pad6;
  const int m = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  _Pragma("unroll")
  for (int j = 0; j < CPC; ++j) {
    const int c = warp * CPC + j;
    float am = (float)lane;
    _Pragma("unroll")
    for (int o = 16; o > 0; o >>= 1)
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float s = fmaxf(am, 1e-6f) * (1.0f / 127.0f);
    int q0 = (int)rintf((float)lane / s);
    int rsum = q0;
    _Pragma("unroll")
    for (int o = 16; o > 0; o >>= 1)
      rsum += __shfl_xor_sync(0xffffffffu, rsum, o);
    if (lane == 0) { sx[(size_t)m * NCH + c] = s; rs[(size_t)m * NCH + c] = rsum; }
  }
}
