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
extern "C" __global__ void __launch_bounds__(256) a_o(
    const unsigned char* __restrict__ wo, const float* __restrict__ grid512,
    const __half* __restrict__ ao_in, __half* __restrict__ attn_out)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wo + (size_t)warp * 2640u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 24; ++b) {
    const unsigned char* blk = rowp + (size_t)b * 110u;
    const float d = __half2float(*((const __half*)blk));
    // element e = b*256 + lane*8 + j
    const int g0 = lane*2, g1 = lane*2 + 1;
    const int sraw = lane >> 2;                     // e>>5 constant per lane
    const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
    const unsigned char* sgnb = blk + 74 + lane;    // sign byte index = e>>3 mod 256
    const int koff = (b << 8) + (lane << 3);
    const float4 xa = *(const float4*)(ao_in + koff);
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int g = (j < 4) ? g0 : g1;
      const unsigned int q = (unsigned int)blk[2 + g] + ((((unsigned int)blk[66 + (g>>3)] >> (g&7)) & 1u) << 8);
      const float gv = grid512[(q << 2) + (j & 3)];
      const float sgn = ((*sgnb >> j) & 1) ? -1.f : 1.f;
      const float w = d * sc * gv * sgn;
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) attn_out[warp] = (__half)acc;
}
