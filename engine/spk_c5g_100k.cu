// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W3-100k: split-KV attention for ctx 100k. Replaces aattn3 (T=3 probe) and
// a_attn/aattn_d (T=1 trunk/draft) when the l-loop becomes BW-bound. THREE kernels:
//   KPRE: per-head q-norm + partial-rope -> qw workspace (fp32, qe*0.0625 folded);
//         k-norm + rope + fp16 KV append at pos..pos+ROWS-1 (verbatim aattn3 phase-1).
//   K1S : grid (4 kv-groups x S splits). GQA-shared: ONE KV read serves all 6 q-heads
//         x ROWS rows. 8 warps lockstep over l (stride 1, tile-unrolled); warp w owns
//         rows w, w+8, (w+16); per-row online softmax in registers; partials
//         (m, s, acc[256]) to workspace. Empty split -> identity partials.
//   K2S : grid 24 heads; combine S partials sequentially (fixed order, same epilogue
//         structure as aattn3), sigmoid gate, write ao.
// Per-row op order matches aattn3/a_attn exactly (Tier-1: T=3 rows bit-identical to
// T=1 rows). Laws: sequential loops, FULL-mask shuffles only on full warps, flat
// indexing, naturally-aligned float4 loads (KV rows are 512B-aligned, lane*16B).
// -DKNAME -DCTXK -DROWS (3|1) -DS (splits, power of 2) -DCH (chunk = CTXK/S) -DUNROLL
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define EPS_N 1e-6f
#define RMAX (6*ROWS)

#define LDH8(NM, P) float NM[8]; { \
  const float4 xa = *(const float4*)((P)); \
  const __half2* hx = (const __half2*)&xa; \
  float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]); \
  NM[0]=f0.x; NM[1]=f0.y; NM[2]=f1.x; NM[3]=f1.y; NM[4]=f2.x; NM[5]=f2.y; NM[6]=f3.x; NM[7]=f3.y; }

// ============================== KPRE ==============================
extern "C" __global__ void __launch_bounds__(256) K2S(
    const float* __restrict__ pm, const float* __restrict__ ps, const float* __restrict__ pA,
    const __half* __restrict__ qrow, __half* __restrict__ ao)
{
  const int h = blockIdx.x;
  const int g = h / 6, hl = h % 6;
  const int d = threadIdx.x;
  for (int t = 0; t < ROWS; ++t) {
    const int r = hl*ROWS + t;
    const size_t pb = (size_t)(g*S)*RMAX + r;
    float M = -1e30f;
    for (int s2 = 0; s2 < S; ++s2) M = fmaxf(M, pm[pb + (size_t)s2*RMAX]);
    float out = 0.f, Ssum = 0.f;
    for (int s2 = 0; s2 < S; ++s2) {
      const float ex = expf(pm[pb + (size_t)s2*RMAX] - M);
      Ssum += ps[pb + (size_t)s2*RMAX] * ex;
      out += pA[(pb + (size_t)s2*RMAX)*256 + d] * ex;
    }
    out /= Ssum;
    const float gf = __half2float(qrow[(ROWS==1 ? 0 : t*12288) + h*512 + 256 + d]);
    const float sg = 1.0f / (1.0f + expf(-gf));
    ao[(ROWS==1 ? 0 : t*6144) + h*256 + d] = __float2half(out * sg);
  }
}
