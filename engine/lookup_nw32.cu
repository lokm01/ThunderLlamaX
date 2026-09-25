// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R3 LOOKUP drafter: in-graph n-gram suffix-match proposal kernel.
// Appended to the END of the draft graph: runs after the MTP draft chain has
// written dring0/dring1, and OVERWRITES them when a long enough n-gram match
// of the recent suffix is found in tok_hist. The probe verifies the target
// model -> Tier-1 exactness preserved by construction (accept.cu compares
// amds vs the same dring buffers; m<K handled).
// Match contract (identical to the offline analyzer lut_offline.py):
//   pos = pos_slot[0] (fed count), cur = cur_slot[0] (predicted, unfed);
//   S[0..7] = hist[pos-8..pos-1] (hist[pos-1]==cur per the engine stream law);  window W(i) = hist[i..i+7];
//   l(i) = leading match length; scan i in [0, pos-11] (both proposal
//   tokens hist[i+l], hist[i+l+1] are FED); best = max l, tie -> max i
//   (most recent occurrence) via packed key (l<<20)|i, unique -> max is
//   order-independent = deterministic. HIT = l >= 6 (LMIN; offline: l>=6
//   gives alpha1 = 1.000 on the 100k continuation; l in [4,5] is
//   spurious-match-contaminated, alpha1 0.824 < MTP's 0.892).
// Single 1024-thread CTA (name carries nw32 for the gcycle LS law);
// hardcoded sizes (dext law: blockDim/gridDim read 0); flat strided scan
// i += 1024 is THREAD-strided (blockDim hardcoded), not grid-stride.
// Outputs: dring0[0], dring1[0] (only on hit), l_hist[cyc] = l (always,
// instrumentation; 0 = miss; LMIN=8 = full-window match).
extern "C" __global__ void __launch_bounds__(1024) lookup_nw32(
    const int* __restrict__ tok_hist, const int* __restrict__ pos_slot,
    const int* __restrict__ cur_slot, int* __restrict__ dring0, int* __restrict__ dring1,
    int* __restrict__ l_hist, const int* __restrict__ cyc_slot)
{
  __shared__ int bkey;
  const int tid = threadIdx.x;
  if (tid == 0) bkey = 0;
  __syncthreads();
  const int pos = pos_slot[0];
  const int cur = cur_slot[0];
  int lkey = 0; int iter0 = -1;
  if (pos >= 11) {   // suffix fully defined (pos>=8) and scan range [0,pos-11] non-empty (pos>=11)
    // ENGINE STREAM LAW: accept writes hist[p+t]=amds[t] incl. the bonus -> hist[pos-1] == cur
    // (and out[0] never enters hist). The suffix IS hist[pos-8..pos-1] — reading cur
    // separately duplicates it ([..,cur,cur] = impossible 8-gram; the 50x l=7 signature).
    const int s0 = tok_hist[pos-8], s1 = tok_hist[pos-7], s2 = tok_hist[pos-6], s3 = tok_hist[pos-5];
    const int s4 = tok_hist[pos-4], s5 = tok_hist[pos-3], s6 = tok_hist[pos-2];
    const int s7 = tok_hist[pos-1]; (void)cur;
    const int iend = pos - 11;
    for (int i = tid; i <= iend; i += 1024) { if (tid == 0) ++iter0;
      int l = 0;
      if (tok_hist[i] == s0) { l = 1;
        if (tok_hist[i+1] == s1) { l = 2;
          if (tok_hist[i+2] == s2) { l = 3;
            if (tok_hist[i+3] == s3) { l = 4;
              if (tok_hist[i+4] == s4) { l = 5;
                if (tok_hist[i+5] == s5) { l = 6;
                  if (tok_hist[i+6] == s6) { l = 7;
                    if (tok_hist[i+7] == s7) { l = 8; } } } } } } } }
      if (l >= 1) { const int k = (l << 20) | i; if (k > lkey) lkey = k; }  // R3 DIAG: record best PARTIAL (overwrite still gated l>=8)
    }
  }
  if (lkey) atomicMax(&bkey, lkey);
  __syncthreads();
  if (tid == 0) {
    const int key = bkey;
    const int l = key >> 20;
    l_hist[cyc_slot[0]] = (pos < 11) ? (1000000 + pos) : (l + 1);   // 1..7 miss; 9 hit; 0 unwritten; >=1000000 = pos-guard diag
    l_hist[500001] = pos;                                   // R3 DIAG: raw pos as seen
    l_hist[500002] = (pos >= 11) ? tok_hist[pos-7] : -2;    // R3 DIAG: s0 as seen
    l_hist[500003] = (tid == 0) ? iter0 : l_hist[500003]; l_hist[500004] = key & 0xFFFFF;  // R3 DIAG: tid0 iters, best i
    if (l >= 8) {
      const int i = key & 0xFFFFF;
      dring0[0] = tok_hist[i + l];
      dring1[0] = tok_hist[i + l + 1];
    }
  }
}
