// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-b: attention blocks (16x, Qwen3.8-27B). T=1 decode.
// dims: 24 q-heads x 256, 4 kv-heads x 256, gate interleaved [q|gate] per head,
// qk RMSNorm per head (eps 1e-6), NeoX rope (pairs (i, i+128)), GQA h->h/6,
// scale 1/16, fp16 KV cache [2][4][2048][256], sigmoid gate, o_proj 6144->5120.
// Weight types: q = Q6_K or IQ3_XXS (per block), k = IQ3_XXS, v = Q4_K, o = IQ3_S.
// HARD RULES: flat indexing, no gridDim reads, sequential loops, single-array smem,
// byte/uint16 loads on Q-buffers, per-kernel cubins.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define NH 24
#define NKV 4
#define HD 256
#define CTX 2048
#define EPS_N 1e-6f
#define QOUT 12288
#define KVOUT 1024

// ---- a_q6: Q6_K GEMV [12288 rows x 5120 in], row 4200B (20 blocks x 210B) ----
extern "C" __global__ void __launch_bounds__(256) a_kv(
    const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv,
    const float* __restrict__ gridf, const __half* __restrict__ xh,
    __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  float acc = 0.f;
  if (warp < KVOUT) {
    // IQ3_XXS row (identical math to k1_iq3)
    const unsigned char* rowp = wk + (size_t)warp * 1960u;
    #pragma unroll 4
    for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + (size_t)b * 98u;
      const float d = __half2float(*((const __half*)blk));
      const unsigned short* scw = (const unsigned short*)(blk + 66);
      const unsigned int sw = ((unsigned int)scw[2*(lane>>2)]) | (((unsigned int)scw[2*(lane>>2)+1]) << 16);
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu;
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
      const unsigned int q = ((unsigned int)blk[2 + 2*lane]) | (((unsigned int)blk[3 + 2*lane]) << 8);
      const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2)));
      const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2)));
      const int koff = (b << 8) + (lane << 3);
      const float4 xa = *(const float4*)(xh + koff);
      const __half2* hx = (const __half2*)&xa;
      float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
      const float xvv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
      const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3,
                            db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(wv[j])));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) krow[warp] = (__half)acc;
  } else {
    // Q4_K row (2880B = 20 blocks x 144B); scale decode identical to Q5_K, no qh, qs at +16
    const int r = warp - KVOUT;
    const unsigned char* rowp = wv + (size_t)r * 2880u;
    #pragma unroll 4
    for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + b*144;
      const float d = __half2float(*((const __half*)blk));
      const float dm = __half2float(*((const __half*)(blk+2)));
      const int s = lane >> 2;
      float sc, mn;
      if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
      else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
             mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
      const unsigned char* qsb = blk + 16 + ((lane >> 3) << 5) + ((lane & 3) << 3);
      const int nsh = ((lane >> 2) & 1) << 2;
      const int koff = (b << 8) + (lane << 3);
      const float4 xf0 = *(const float4*)(xh + koff);
      const __half2* h0 = (const __half2*)&xf0;
      float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
             f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
      const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float qv = (float)((qsb[j] >> nsh) & 0xF);
        const float w = d*sc*qv - dm*mn;
        acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) vrow[r] = (__half)acc;
  }
}

// ---- a_attn: qk-norm + rope + KV append + gated softmax attention. grid=(24,) ----
// CTA = q head h; kv head kvh = h/6 (each CTA re-derives its kv head's k/v; only
// h%6==0 stores -> no cross-CTA dependency on the freshly written row).
// smem (single array): [0..15] reduce partials/bcast; [256..511] qn then kn stash;
// [512..767] qe (x 0.0625); [1024..3071] per-warp partial out (8x256);
// [3072..3087] per-warp (m, s). All write-after-read reuses are syncthreads-separated.
