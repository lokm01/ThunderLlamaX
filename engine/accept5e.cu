// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P10-dense rung 2 accept5e: the K=4 EAGLE accept — accept5k verbatim except
// the dhd_seed mapping over the 4-step chain's DEDICATED hidden buffers
// (hd_d0=h(cur), hd_d1=h(dring0), hd_d2=h(dring1), hd_d3=h(dring2)): EXACT
// for m<=3; m=4 commits amds[4] which no chain step processed -> hd_d3 (1
// stale — the documented heuristic-only class; the probe verifies everything,
// draft numerics carry no exactness contract). h_seed = xA[m*DIM] (m<=4 — T=5
// probe rows). emit layout IDENTICAL to accept5k (16-word) — the DecodeSession
// readout is shared with the acceptk/accept5k class.
#include <cuda_fp16.h>
#define DIM 5120
extern "C" __global__ void __launch_bounds__(256) accept5e(
    const int* __restrict__ amds, const int* __restrict__ dring0, const int* __restrict__ dring1,
    const int* __restrict__ dring2, const int* __restrict__ dring3, const float* __restrict__ xA,
    int* __restrict__ m_slot, int* __restrict__ m_hist, int* __restrict__ cyc_slot,
    int* __restrict__ pos_slot, int* __restrict__ cur_slot, int* __restrict__ tok_hist,
    float* __restrict__ h_seed,
    const float* __restrict__ hd_d0, const float* __restrict__ hd_d1,
    const float* __restrict__ hd_d2, const float* __restrict__ hd_d3,
    float* __restrict__ dhd_seed, int* __restrict__ emit, const int* __restrict__ l_hist)
{
  __shared__ int ms;
  if (threadIdx.x == 0) {
    int m = 0;
    if (amds[0] == dring0[0]) { m = 1;
      if (amds[1] == dring1[0]) { m = 2;
        if (amds[2] == dring2[0]) { m = 3;
          if (amds[3] == dring3[0]) m = 4; } } }
    const int pos = pos_slot[0];
    for (int t = 0; t <= m; ++t) tok_hist[pos + t] = amds[t];
    cur_slot[0] = amds[m];
    pos_slot[0] = pos + m + 1;
    m_slot[0] = m;
    const int cyc = cyc_slot[0];
    m_hist[cyc] = m;
    emit[0] = pos + m + 1;
    emit[1] = m;
    emit[2] = amds[0]; emit[3] = amds[1]; emit[4] = amds[2]; emit[5] = amds[3]; emit[6] = amds[4];
    emit[7] = 0;                    // stop_flag
    emit[8] = cyc;
    emit[9] = l_hist[cyc];          // hit flag (9 = 8-gram hit)
    cyc_slot[0] = cyc + 1;
    ms = m;
  }
  __syncthreads();
  const int m = ms;
  const float* dh = (m == 0) ? hd_d0 : (m == 1) ? hd_d1 : (m == 2) ? hd_d2 : hd_d3;
  for (int i = threadIdx.x; i < DIM; i += 256) { h_seed[i] = xA[m*DIM + i]; dhd_seed[i] = dh[i]; }
}
