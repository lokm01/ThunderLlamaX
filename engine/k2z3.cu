// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2F L4: k2z3 = the zz RMS-norm + gated z3 tail of k2s3, as its own kernel
// (grid 48, one warp) so k2s3v can grid-split. ONE warp computes zz with the
// EXACT original reduction order (i=lane..128 stride 32 + full butterfly) and
// every z3 element with the original expression -> BIT-IDENTICAL to k2s3.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define EPS_N 1e-6f

extern "C" __global__ void __launch_bounds__(32) k2z3(
    const float* __restrict__ core, const __half* __restrict__ gate3,
    const float* __restrict__ snw, __half* __restrict__ z3)
{
  const int h = blockIdx.x;
  const int lane = threadIdx.x;
  for (int t = 0; t < 3; ++t) {
    float zz = 0.f;
    for (int i = lane; i < 128; i += 32) { const float c = core[h*128+i]; zz += c*c; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) zz += __shfl_xor_sync(FULL, zz, o);
    const float rz = rsqrtf(zz/128 + EPS_N);
    #pragma unroll
    for (int s2 = 0; s2 < 4; ++s2) {
      const int j = lane + s2*32;
      const __half g = gate3[t*6144 + h*128 + j];
      z3[t*6144 + h*128 + j] = __float2half((core[h*128+j]*rz*snw[j]) * __half2float(
        __hmul(g, hrcp((__half)1.0f + hexp2(__hmul(g, __float2half(-1.4423828125f)))))));
    }
  }
}
