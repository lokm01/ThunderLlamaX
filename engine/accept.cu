// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2-MTP + M1-A: draft-chain + accept/commit kernels. Draft = blk.64 (all Q4_0,
// repacked two-region per row: [qs NGRP*16B][d NGRP*8*2B] — all loads naturally
// aligned per the ALIGNMENT LAW). Draft numerics are heuristic-only (no bit-exact
// contract); the ACCEPT kernel implements the exact mtp_v3 emission contract.
// M1-A additions (new outputs ONLY — every legacy write bit-identical):
//   emit[8]   per-cycle outbox: {pos_new, m, tok0..2, stop_flag(0), cyc, rsv}
//   dhd_seed  committed-position draft hidden: hd_d0 if m==0 else hd_d1
//   (= draft hidden after the last COMMITTED fed token; fill_draft resume seed.)
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
extern "C" __global__ void __launch_bounds__(256) accept(
    const int* __restrict__ amds, const int* __restrict__ dring0, const int* __restrict__ dring1, const float* __restrict__ xA,
    int* __restrict__ m_slot, int* __restrict__ m_hist, int* __restrict__ cyc_slot,
    int* __restrict__ pos_slot, int* __restrict__ cur_slot, int* __restrict__ tok_hist,
    float* __restrict__ h_seed,
    const float* __restrict__ hd_d0, const float* __restrict__ hd_d1,
    float* __restrict__ dhd_seed, int* __restrict__ emit)
{
  __shared__ int ms;
  if (threadIdx.x == 0) {
    int m = 0;
    if (amds[0] == dring0[0]) { m = 1; if (amds[1] == dring1[0]) m = 2; }
    const int pos = pos_slot[0];
    for (int t = 0; t <= m; ++t) tok_hist[pos + t] = amds[t];
    cur_slot[0] = amds[m];
    pos_slot[0] = pos + m + 1;
    m_slot[0] = m;
    m_hist[cyc_slot[0]] = m;
    emit[0] = pos + m + 1;          // pos_new
    emit[1] = m;
    emit[2] = amds[0]; emit[3] = amds[1]; emit[4] = amds[2];
    emit[5] = 0;                    // stop_flag (reserved; M2 stop_ids)
    emit[6] = cyc_slot[0];          // this cycle's index (pre-increment)
    emit[7] = 0;                    // reserved
    cyc_slot[0] = cyc_slot[0] + 1;
    ms = m;
  }
  __syncthreads();
  const int m = ms;
  const float* dh = (m == 0) ? hd_d0 : hd_d1;
  for (int i = threadIdx.x; i < DIM; i += 256) { h_seed[i] = xA[m*DIM + i]; dhd_seed[i] = dh[i]; }
}

// ---- acceptsel: copy per-block rec/conv slot m -> live slot 4 ----
// rec4: [48][5][786432] f32; conv4: [48][5][30720] f32 (block-major); grid 48 CTAs.
