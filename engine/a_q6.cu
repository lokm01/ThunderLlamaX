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
extern "C" __global__ void __launch_bounds__(256) a_q6(
    const unsigned char* __restrict__ wq6, const __half* __restrict__ xh, __half* __restrict__ qrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const unsigned char* rowp = wq6 + (size_t)warp * 4200u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + b*210;
    const float d = __half2float(*((const __half*)(blk+208)));
    // element e = b*256 + lane*8 + j;  h = e>>7, i = e&127
    const int lo_off = 64*(lane>>4) + (lane&7)*8;
    const bool nib_hi = ((lane&15) >= 8);
    const int c2 = (lane>>2)&3;   // 2-bit chunk = i>>5 = (lane*8+j)>>5, j<8 -> (lane>>2), mask to 0..3
    const unsigned char* qhp = blk + 128 + (lane>>4)*32 + (lane&3)*8;
    const int sc8 = (signed char)blk[192 + (lane>>1)];
    const int koff = (b << 8) + (lane << 3);
    const float4 xf0 = *(const float4*)(xh + koff);
    const __half2* h0 = (const __half2*)&xf0;
    float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
           f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int lo_byte = blk[lo_off + j];
      const int xl = nib_hi ? (lo_byte >> 4) : (lo_byte & 0xF);
      const int xh = ((qhp[j] >> (c2<<1)) & 3) << 4;
      const float w = d * (float)sc8 * (float)((signed char)((xl | xh) - 32));
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) qrow[warp] = (__half)acc;
}

// ---- a_kv: k GEMV (IQ3_XXS 1024 rows) + v GEMV (Q4_K 1024 rows) in one launch ----
