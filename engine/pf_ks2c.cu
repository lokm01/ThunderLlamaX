// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P11 G1: fixed-order K-split-2 combine (see pf_gemm.cu KS=2 twins).
// Reads fp32 partials p[2][32][NDIM]; adds p0+p1 elementwise in a FIXED order
// (+ res16 -> out32 for the RES class; else half out16). 4 elem/thread (16B);
// grid = 32*NDIM/(4*NTHR) = 160 CTAs at NDIM=5120, NTHR=256.
#include <cuda_fp16.h>
#ifndef NTHR
#define NTHR 256
#endif
#if KS2C
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const float* __restrict__ p,
#if KS2C_RES
    const float* __restrict__ res16,
    float* __restrict__ out32
#else
    __half* __restrict__ out16
#endif
    )
{
  const int i0 = (blockIdx.x * NTHR + threadIdx.x) * 4;
  const float4 a = *(const float4*)(p + i0);
  const float4 b = *(const float4*)(p + 32*NDIM + i0);
#if KS2C_RES
  const float4 r = *(const float4*)(res16 + i0);
  *(float4*)(out32 + i0) = make_float4(r.x + a.x + b.x, r.y + a.y + b.y,
                                       r.z + a.z + b.z, r.w + a.w + b.w);
#else
  *(__half2*)(out16 + i0)     = __floats2half2_rn(a.x + b.x, a.y + b.y);
  *(__half2*)(out16 + i0 + 2) = __floats2half2_rn(a.z + b.z, a.w + b.w);
#endif
}
#endif  // KS2C
