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
extern "C" __global__ void __launch_bounds__(256) samx(
    const __half* __restrict__ slogits, const int* __restrict__ stab, int* __restrict__ out_tok)
{
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  __shared__ int sm[16];
  float best = -1e30f; int bidx = 0;
  for (int i = tid; i < SLICE; i += 256) {
    const float v = __half2float(slogits[i]);
    if (v > best || (v == best && i < bidx)) { best = v; bidx = i; }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(FULL, best, o);
    const int oi = __shfl_down_sync(FULL, bidx, o);
    if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
  }
  if (lane == 0) { sm[warp] = __float_as_int(best); sm[8+warp] = bidx; }
  __syncthreads();
  if (tid == 0) {
    for (int w = 1; w < 8; ++w)
      if (__int_as_float(sm[w]) > __int_as_float(sm[0]) || (__int_as_float(sm[w]) == __int_as_float(sm[0]) && sm[8+w] < sm[8])) { sm[0] = sm[w]; sm[8] = sm[8+w]; }
    out_tok[0] = stab[sm[8]];
  }
}

// ---- dposadd: dst[0] = src[0] + 1 ----
