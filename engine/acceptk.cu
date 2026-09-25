// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R4 acceptk: the K=2 accept (emit machinery IDENTICAL to accept.cu) extended
// to the 16-word emit record: {pos_new, m, tok0..4, stop, cyc, hit, rsv...}.
// emit[9] = l_hist[cyc] (the lookup match len+1; 9 = full 8-gram hit) — the
// host's per-cycle graph-set selector reads it (readout-order law: the DECISION
// for the next cycle reads THIS cycle's completed emit). l_hist exists whenever
// LOOKUP/LOOKUP_K is on. Legacy emit[2..4] tok slots unchanged; 5..6 zero.
#include <cuda_fp16.h>
#define DIM 5120
extern "C" __global__ void __launch_bounds__(256) acceptk(
    const int* __restrict__ amds, const int* __restrict__ dring0, const int* __restrict__ dring1, const float* __restrict__ xA,
    int* __restrict__ m_slot, int* __restrict__ m_hist, int* __restrict__ cyc_slot,
    int* __restrict__ pos_slot, int* __restrict__ cur_slot, int* __restrict__ tok_hist,
    float* __restrict__ h_seed,
    const float* __restrict__ hd_d0, const float* __restrict__ hd_d1,
    float* __restrict__ dhd_seed, int* __restrict__ emit, const int* __restrict__ l_hist)
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
    const int cyc = cyc_slot[0];
    m_hist[cyc] = m;
    emit[0] = pos + m + 1;
    emit[1] = m;
    emit[2] = amds[0]; emit[3] = amds[1]; emit[4] = amds[2];
    emit[5] = 0; emit[6] = 0;
    emit[7] = 0;                    // stop_flag
    emit[8] = cyc;
    emit[9] = l_hist[cyc];          // R4: hit flag (9 = 8-gram hit)
    cyc_slot[0] = cyc + 1;
    ms = m;
  }
  __syncthreads();
  const int m = ms;
  const float* dh = (m == 0) ? hd_d0 : hd_d1;
  for (int i = threadIdx.x; i < DIM; i += 256) { h_seed[i] = xA[m*DIM + i]; dhd_seed[i] = dh[i]; }
}
