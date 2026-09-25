// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R2c M=128 split-plane FFN epilogue: gact = hmul(hsilu(ag), au) over
// [128][17408] halves. BIT-IDENTICAL to the fused FFN epilogue by construction:
// the fused kernel writes __hmul(hsilu_h((half)accG), (half)accU); the plain
// GEMMs write exactly (half)accG / (half)accU, and hsilu_hs here is the SAME
// function — identical half values, identical op order. (16B-aligned 8-half
// chunks per thread; MM*FFN_N % 2048 == 0 so no tail.)
#include <cuda_fp16.h>
#define FFN_N 17408
#define MM 128

__device__ __forceinline__ __half hsilu_hs(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

extern "C" __global__ void __launch_bounds__(256) pfk_smul128(
    const __half* __restrict__ ag, const __half* __restrict__ au, __half* __restrict__ gact)
{
  const int i = (blockIdx.x * 256 + threadIdx.x) * 8;   // 8 halves (4 half2) per thread
  const __half2* a2 = (const __half2*)(ag + i);
  const __half2* u2 = (const __half2*)(au + i);
  __half2* g2 = (__half2*)(gact + i);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half2 hg = a2[j], hu = u2[j];
    g2[j] = __halves2half2(__hmul(hsilu_hs(hg.x), hu.x), __hmul(hsilu_hs(hg.y), hu.y));
  }
}
