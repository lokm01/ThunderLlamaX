// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R5 K=6 LOOKUP7 drafter: deep-K (K=6) n-gram proposal kernel.
// Same contract as lookup5_nw32 (R4) with the DEEP-K SCAN-RANGE LAW at K=5:
// iend = pos - 8 - K = pos - 14 (all 6 proposal tokens hist[i+8..i+13] are
// WRITTEN ids <= hist[pos-1] = cur). Writes dring0..5 only on hit (l >= 8);
// any proposal source is Tier-1-safe (the probe verifies the target model).
// Single 1024-thread CTA (nw32 name law); thread-strided flat scan.
extern "C" __global__ void __launch_bounds__(1024) lookup7_nw32(
    const int* __restrict__ tok_hist, const int* __restrict__ pos_slot,
    const int* __restrict__ cur_slot, int* __restrict__ dring0, int* __restrict__ dring1,
    int* __restrict__ dring2, int* __restrict__ dring3, int* __restrict__ dring4, int* __restrict__ dring5,
    int* __restrict__ l_hist, const int* __restrict__ cyc_slot)
{
  __shared__ int bkey;
  const int tid = threadIdx.x;
  if (tid == 0) bkey = 0;
  __syncthreads();
  const int pos = pos_slot[0];
  const int cur = cur_slot[0];
  int lkey = 0;
  if (pos >= 14) {   // suffix defined (pos>=8) and scan [0,pos-14] non-empty
    const int s0 = tok_hist[pos-8], s1 = tok_hist[pos-7], s2 = tok_hist[pos-6], s3 = tok_hist[pos-5];
    const int s4 = tok_hist[pos-4], s5 = tok_hist[pos-3], s6 = tok_hist[pos-2];
    const int s7 = tok_hist[pos-1]; (void)cur;
    const int iend = pos - 14;
    for (int i = tid; i <= iend; i += 1024) {
      int l = 0;
      if (tok_hist[i] == s0) { l = 1;
        if (tok_hist[i+1] == s1) { l = 2;
          if (tok_hist[i+2] == s2) { l = 3;
            if (tok_hist[i+3] == s3) { l = 4;
              if (tok_hist[i+4] == s4) { l = 5;
                if (tok_hist[i+5] == s5) { l = 6;
                  if (tok_hist[i+6] == s6) { l = 7;
                    if (tok_hist[i+7] == s7) { l = 8; } } } } } } } }
      if (l >= 1) { const int k = (l << 20) | i; if (k > lkey) lkey = k; }
    }
  }
  if (lkey) atomicMax(&bkey, lkey);
  __syncthreads();
  if (tid == 0) {
    const int key = bkey;
    const int l = key >> 20;
    l_hist[cyc_slot[0]] = (pos < 14) ? (1000000 + pos) : (l + 1);   // 1..8 miss; 9 hit
    if (l >= 8) {
      const int i = key & 0xFFFFF;
      dring0[0] = tok_hist[i + 8];
      dring1[0] = tok_hist[i + 9];
      dring2[0] = tok_hist[i + 10];
      dring3[0] = tok_hist[i + 11];
      dring4[0] = tok_hist[i + 12];
      dring5[0] = tok_hist[i + 13];
    }
  }
}
