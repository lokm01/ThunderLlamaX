// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2-MTP: draft-chain + accept/commit kernels. Draft = blk.64 (all Q4_0,
// repacked two-region per row: [qs NGRP*16B][d NGRP*8*2B] — all loads naturally
// aligned per the ALIGNMENT LAW). Draft numerics are heuristic-only (no bit-exact
// contract); the ACCEPT kernel implements the exact mtp_v3 emission contract.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define INNER 6144
#define FFN_N 17408
#define EPS_N 1e-6f
#define NH 24
#define CTXK 2304
#define SLICE 40960

__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- dnorm2: enorm(e) || hnorm(hm) -> cat[10240] halves ----
extern "C" __global__ void __launch_bounds__(256) dnorm2(
    const float* __restrict__ e, const float* __restrict__ hm,
    const float* __restrict__ enw, const float* __restrict__ hnw,
    __half* __restrict__ cat)
{
  const int lane = threadIdx.x & 31;
  float ss0 = 0.f, ss1 = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v0 = e[i]; ss0 += v0*v0; const float v1 = hm[i]; ss1 += v1*v1; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { ss0 += __shfl_xor_sync(FULL, ss0, o); ss1 += __shfl_xor_sync(FULL, ss1, o); }
  const float r0 = rsqrtf(ss0/DIM + EPS_N), r1 = rsqrtf(ss1/DIM + EPS_N);
  for (int i = threadIdx.x; i < DIM; i += 256) {
    cat[i] = __float2half(e[i]*r0*enw[i]);
    cat[DIM + i] = __float2half(hm[i]*r1*hnw[i]);
  }
}

// ---- dfgu: draft FFN gate+up Q4 GEMVs + silu-mul ----
