// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P7-F probe: dp4a int8 QK dot on the ENGINE kv8 layout (real KV data).
// Layout (K half of engine kv{i} slab): [4 groups][CTXK][256] biased-u8
// (+128), scales per 32-ch group. Head h of 8: group g=h>>1, within-row
// byte base (h&1)*128. K' = K-128 PRECOMPUTED (the kpre trick) as s8 quads.
// Q quantized per-(row,head) 128-ch (absmax/127), packed s8 quads int32.
// out[t][h][pos] = s_q * sum_j s_k[pos][j] * dp4a_partial_j   (fp16 out)
//
// Structure: warp per (t,h,L-segment); LANE PER POSITION (no cross-lane
// reduce); fixed per-warp range (NO grid-stride); full-warp masks; q quads
// hoisted into registers via unrolled compile-time loops (no runtime-indexed
// locals).
#include <cuda_fp16.h>

#define CTXK 100352
#define QQUADS 32   // 128 ch = 32 int32 quads per (t,h)

extern "C" __global__ void pfk_dp4a(
    const int*   __restrict__ kp,     // K' quads: [4*CTXK*64] int32 (256 ch/row)
    const float* __restrict__ scp,    // scales fp32: [4*CTXK*8]
    const int*   __restrict__ qp,     // Q quads:   [T*8*32] int32
    const float* __restrict__ sqp,    // Q per-(t,h) scale: [T*8]
    __half*      __restrict__ out,    // [T*8*CTXK]
    const int*   __restrict__ np)     // [1] = T*8 (warps per L-segment)
{
  const int npairs = np[0];
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) + (blockIdx.x << 3);  // 8 warps/CTA
  const int seg  = warp / npairs;          // L-segment index
  const int p    = warp - seg * npairs;    // pair index = t*8+h
  const int t    = p >> 3, h = p & 7;
  const int g    = h >> 1;                 // kv group
  const int qrow = (h & 1) * 32;           // within-row quad base (128 ch = 32 quads)
  const int scb  = (h & 1) * 4;            // within-row scale base (4 groups of 32ch)

  // per-warp fixed L range (no grid-stride)
  // P7E7 fix: gridDim.x reads 0 on the dext (hardcode law) => nseg was 0 =>
  // empty L-ranges => the silent no-op. nseg is architecturally NSEG=8 (the
  // harness always launches NSEG*npairs warps).
  const int nseg = 8;
  const int l0 = seg * (CTXK / nseg);
  const int l1 = l0 + (CTXK / nseg);

  // hoist Q quads into registers (compile-time indices only)
  int qv[QQUADS];
  #pragma unroll
  for (int i = 0; i < QQUADS; i++) qv[i] = qp[p * QQUADS + i];
  const float sq = sqp[p];

  const int* krow = kp + (size_t)g * CTXK * 64;
  const float* srow = scp + (size_t)g * CTXK * 8;
  __half* orow = out + (size_t)p * CTXK;

  for (int pos = l0 + lane; pos < l1; pos += 32) {
    const int* kq = krow + (size_t)pos * 64 + qrow;
    const float* s4 = srow + (size_t)pos * 8 + scb;
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;   // 4 group partials (32 ch each)
    #pragma unroll
    for (int u = 0; u < 8; u++) {
      a0 = __dp4a(qv[0 * 8 + u], kq[0 * 8 + u], a0);
      a1 = __dp4a(qv[1 * 8 + u], kq[1 * 8 + u], a1);
      a2 = __dp4a(qv[2 * 8 + u], kq[2 * 8 + u], a2);
      a3 = __dp4a(qv[3 * 8 + u], kq[3 * 8 + u], a3);
    }
    float sc = (float)a0 * s4[0] + (float)a1 * s4[1] + (float)a2 * s4[2] + (float)a3 * s4[3];
    orow[pos] = __float2half(sc * sq);
  }
}
