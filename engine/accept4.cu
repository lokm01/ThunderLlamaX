// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2D-L2: K=3 accept kernel (m-ladder to 3; 12 args — count on both sides!)
#include <cuda_fp16.h>
#define DIM 5120
extern "C" __global__ void __launch_bounds__(256) accept4(
    const int* __restrict__ amds, const int* __restrict__ dring0, const int* __restrict__ dring1, const int* __restrict__ dring2,
    const float* __restrict__ xA, int* __restrict__ m_slot, int* __restrict__ m_hist, int* __restrict__ cyc_slot,
    int* __restrict__ pos_slot, int* __restrict__ cur_slot, int* __restrict__ tok_hist, float* __restrict__ h_seed)
{
  __shared__ int ms;
  if (threadIdx.x == 0) {
    int m = 0;
    if (amds[0] == dring0[0]) { m = 1; if (amds[1] == dring1[0]) { m = 2; if (amds[2] == dring2[0]) m = 3; } }
    const int pos = pos_slot[0];
    for (int t = 0; t <= m; ++t) tok_hist[pos + t] = amds[t];
    cur_slot[0] = amds[m];
    pos_slot[0] = pos + m + 1;
    m_slot[0] = m;
    m_hist[cyc_slot[0]] = m;
    cyc_slot[0] = cyc_slot[0] + 1;
    ms = m;
  }
  __syncthreads();
  const int m = ms;
  for (int i = threadIdx.x; i < DIM; i += 256) h_seed[i] = xA[m*DIM + i];
}
