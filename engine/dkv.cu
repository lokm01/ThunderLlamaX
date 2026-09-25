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
extern "C" __global__ void __launch_bounds__(256) dkv(
    const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv,
    const __half* __restrict__ xh, __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const bool isk = (warp < 1024);
  const int r = isk ? warp : warp - 1024;
  const unsigned char* rowp = (isk ? wk : wv) + (size_t)r * 2880u;
  float acc = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    const float4 xf = *(const float4*)(xh + koff);
    const __half2* hx = (const __half2*)&xf;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    const int subi = (b<<3) + (lane>>2);
    const unsigned long long qs8 = *(const unsigned long long*)(rowp + subi*16 + ((lane&1)<<3));
    const float d = __half2float(*((const __half*)(rowp + 2560 + subi*2)));
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float qv = (float)(((qs8 >> (8*j)) >> (((lane>>1)&1)<<2)) & 0xFu);
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(d*(qv - 8.f))));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) { if (isk) krow[r] = (__half)acc; else vrow[r] = (__half)acc; }
}

// ---- aattn_d: draft attention (a_attn clone, CTX=2304, pos from arg buffer) ----
