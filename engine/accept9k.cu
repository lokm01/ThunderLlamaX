// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R7a K=8 accept9k: the DEEP (K=8) accept — m-ladder to 8 over dring0..7.
// 20-word emit record: {pos_new, m, tok0..8, stop, cyc, hit, rsv...}; DecodeSession
// reads layout-aware (20 when the K=8 deep set ran this cycle). emit[13] = l_hist[cyc]
// (9 = 8-gram hit) = the selection flag. h_seed = xA[m*DIM] (m <= 8 — T=9 rows).
// dhd_seed law unchanged (draft numerics carry no exactness contract).
#include <cuda_fp16.h>
#define DIM 5120
extern "C" __global__ void __launch_bounds__(256) accept9k(
    const int* __restrict__ amds, const int* __restrict__ dring0, const int* __restrict__ dring1,
    const int* __restrict__ dring2, const int* __restrict__ dring3, const int* __restrict__ dring4,
    const int* __restrict__ dring5, const int* __restrict__ dring6, const int* __restrict__ dring7,
    const float* __restrict__ xA,
    int* __restrict__ m_slot, int* __restrict__ m_hist, int* __restrict__ cyc_slot,
    int* __restrict__ pos_slot, int* __restrict__ cur_slot, int* __restrict__ tok_hist,
    float* __restrict__ h_seed,
    const float* __restrict__ hd_d0, const float* __restrict__ hd_d1,
    float* __restrict__ dhd_seed, int* __restrict__ emit, const int* __restrict__ l_hist)
{
  __shared__ int ms;
  if (threadIdx.x == 0) {
    int m = 0;
    if (amds[0] == dring0[0]) { m = 1;
      if (amds[1] == dring1[0]) { m = 2;
        if (amds[2] == dring2[0]) { m = 3;
          if (amds[3] == dring3[0]) { m = 4;
            if (amds[4] == dring4[0]) { m = 5;
            if (amds[5] == dring5[0]) { m = 6;
            if (amds[6] == dring6[0]) { m = 7;
            if (amds[7] == dring7[0]) m = 8; } } } } } } }
    const int pos = pos_slot[0];
    for (int t = 0; t <= m; ++t) tok_hist[pos + t] = amds[t];
    cur_slot[0] = amds[m];
    pos_slot[0] = pos + m + 1;
    m_slot[0] = m;
    const int cyc = cyc_slot[0];
    m_hist[cyc] = m;
    emit[0] = pos + m + 1;
    emit[1] = m;
    emit[2] = amds[0]; emit[3] = amds[1]; emit[4] = amds[2]; emit[5] = amds[3];
    emit[6] = amds[4]; emit[7] = amds[5]; emit[8] = amds[6]; emit[9] = amds[7];
    emit[10] = amds[8];
    emit[11] = 0;                   // stop_flag
    emit[12] = cyc;
    emit[13] = l_hist[cyc];         // hit flag (9 = 8-gram hit)
    cyc_slot[0] = cyc + 1;
    ms = m;
  }
  __syncthreads();
  const int m = ms;
  const float* dh = (m == 0) ? hd_d0 : hd_d1;
  for (int i = threadIdx.x; i < DIM; i += 256) { h_seed[i] = xA[m*DIM + i]; dhd_seed[i] = dh[i]; }
}
