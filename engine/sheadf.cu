// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// TLX P10-dense (Route A rung 1): the FULL-VOCAB draft head. Verbatim port of
// shead.cu (the Q5_K slice-head GEMV) with SLICE -> VOCAB: every chain step
// proposes over the whole 248320-row output.weight instead of the 40960-row
// prompt-frequency slice (the A.2 finding: 89.8% of novel-prose targets were
// out-of-slice -> m=0 -> the alpha-death T=1 fallback at 20.6 tok/s).
// Draft numerics stay heuristic-only (probe-verified proposals; no bit-exact
// contract). Reads the SAME resident W[("head",0)] the trunk probe head uses;
// writes logits3 row 0 (dead scratch at draft time — the probe head overwrites
// rows 0..RM-1 before amx3 reads, timeline-ordered).
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define VOCAB 248320

extern "C" __global__ void __launch_bounds__(256) sheadf(
    const unsigned char* __restrict__ wq5, const __half* __restrict__ xh, __half* __restrict__ slogits)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= VOCAB) return;
  const unsigned char* rowp = wq5 + (size_t)warp * 3520u;
  float acc = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + b*176;
    const float d = __half2float(*((const __half*)blk));
    const float dm = __half2float(*((const __half*)(blk+2)));
    const int s = lane >> 2;
    float sc, mn;
    if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
    else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
           mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
    const unsigned long long qs8 = *(const unsigned long long*)(blk + 48 + ((lane >> 3) << 5) + ((lane & 3) << 3));
    const unsigned long long qh8 = *(const unsigned long long*)(blk + 16 + ((lane & 3) << 3));
    const int nsh = ((lane >> 2) & 1) << 2;
    const int koff = (b << 8) + (lane << 3);
    const float4 xf0 = *(const float4*)(xh + koff);
    const __half2* h0 = (const __half2*)&xf0;
    float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
           f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int qv = (int)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu);
      qv += (int)((((unsigned int)(qh8 >> (8*j)) >> s) & 1u) << 4);
      const float w = d*sc*(float)qv - dm*mn;
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) slogits[warp] = (__half)acc;
}
