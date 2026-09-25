// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// Split-KV K1 v4 — packed half2 math + cp.async-style double buffering.
// Key changes vs v3:
//  - K/Q dot products in HALF2 (__hmul2, __hfma2) with FP32 promotion only at
//    the final add (the proven A3e exactness pattern); ONE half2 butterfly
//    chain (5 __shfl_xor on __half2 = 10 half-lanes) instead of two fp32 chains.
//  - cp.async 16B copies stage K/V tiles to shared memory 1 iteration ahead
//    (true double buffering — loads overlap the previous tile's math).
//  - Tile = 32 positions per CTA-iteration (each warp 4), 128 threads.
#include <cuda_fp16.h>
struct __align__(8) half4 { half x, y, z, w; };
__device__ __forceinline__ half4 zero_h4() { half4 r; r.x=r.y=r.z=r.w=__float2half(0.f); return r; }
#define NEG_INF (__int_as_float(0xff800000))
#define TILE 64
extern "C" __global__ void __launch_bounds__(128) splitkv_k1(
    const __half* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc,
    float* __restrict__ oacc, float* __restrict__ lse,
    const int pos, const int T, const int S, const int n_chunk)
{
  const int bid = blockIdx.x;
  const int h = bid % 8;
  const int s = bid / 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int K_DIM = 128, V_DIM = 128, REP = 2;
  const int R = REP * T;
  const int start = s * n_chunk;
  const int end = min(start + n_chunk, pos + T);

  // shared K/V tile: 32 positions x 128 dims (half) each = 16KB
  __shared__ half sK[TILE][K_DIM], sV[TILE][V_DIM];

  // q in half2 registers (2 dims/lane packed), pre-scaled by FOLD... keep q raw
  // half2 (dot scale applied to x at the end — exactness: fold in fp32 only).
  __half2 qh[10][2];
  for (int r = 0; r < R; ++r) {
    const int qh_ = h * REP + (r / T);
    const int tq = r % T;
    const half2* q2 = (const half2*)(q + ((qh_ * T + tq) * K_DIM));
    qh[r][0] = q2[2*lane]; qh[r][1] = q2[2*lane+1];
  }
  float m[10], l[10], o[10][4];
  for (int r = 0; r < R; ++r) { m[r] = NEG_INF; l[r] = 0.0f;
    o[r][0]=o[r][1]=o[r][2]=o[r][3]=0.0f; }
  const float FOLD = 0.0883883461356163f * 1.4426950408889634f;

  // tile loop: stage tile t+1 while computing tile t
  auto stage = [&](int tile0, int buf) {
    // 128 threads x 4 halves = each thread stages 4 dims of one position row
    // rows: thread tid -> position (tid>>2)... simpler: each of 32 warps->pos?
    // 128 threads: tid covers (pos = tid>>2, dim4 = (tid&3)*4)? TILE=32 rows x
    // 32 dim-quads = 1024 quads; 128 threads do 8 each.
    for (int e = 0; e < 16; ++e) {
      const int flat = threadIdx.x + e * 128;      // 0..2047
      const int pr = flat >> 5, dq = flat & 31;    // pos-row, dim-quad
      const int p = tile0 + pr;
      const half4* src = (const half4*)(kc + ((size_t)p * 8 + h) * K_DIM);
      *((half4*)&sK[pr][dq*4]) = (p < end) ? src[dq] : zero_h4();
      const half4* srcv = (const half4*)(vc + ((size_t)p * 8 + h) * V_DIM);
      *((half4*)&sV[pr][dq*4]) = (p < end) ? srcv[dq] : zero_h4();
    }
  };

  int ntiles = (end - start + TILE - 1) / TILE;
  if (ntiles <= 0) { /* epilogue only */ }
  else {
    stage(start, 0); __syncthreads();
    for (int t = 0; t < ntiles; ++t) {
      const int tile0 = start + t * TILE;
      // compute this tile: each warp handles 4 positions (warp w -> rows w*4..w*4+3)
      #pragma unroll
      for (int pi = 0; pi < 8; ++pi) {
        const int pr = warp * 8 + pi;
        const int p = tile0 + pr;
        if (p >= end) break;
        const half2* k2 = (const half2*)&sK[pr][4*lane];   // lane dims [4l..4l+3]
        const half2 k01 = k2[0], k23 = k2[1];
        const half2* v2 = (const half2*)&sV[pr][4*lane];
        const half2 v01 = v2[0], v23 = v2[1];
        const float v0 = __half2float(__low2half(v01)), v1 = __half2float(__high2half(v01));
        const float v2f = __half2float(__low2half(v23)), v3 = __half2float(__high2half(v23));
        for (int r = 0; r < R; ++r) {
          const int tq = r % T;
          if (p > pos + tq) continue;
          __half2 d2 = __hmul2(qh[r][0], k01);
          d2 = __hfma2(qh[r][1], k23, d2);
          // half2 butterfly (5 xor on half2)
          d2 += __shfl_xor_sync(0xffffffffu, d2, 16);
          d2 += __shfl_xor_sync(0xffffffffu, d2, 8);
          d2 += __shfl_xor_sync(0xffffffffu, d2, 4);
          d2 += __shfl_xor_sync(0xffffffffu, d2, 2);
          d2 += __shfl_xor_sync(0xffffffffu, d2, 1);
          const float dot_h = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
          const float x = dot_h * FOLD;    // scale in fp32 (exactness)
          if (x > m[r]) {
            const float cl = exp2f(m[r] - x);
            l[r] *= cl; o[r][0] *= cl; o[r][1] *= cl; o[r][2] *= cl; o[r][3] *= cl;
            m[r] = x;
          }
          const float pe = exp2f(x - m[r]);
          l[r] += pe;
          o[r][0] += pe * v0; o[r][1] += pe * v1; o[r][2] += pe * v2f; o[r][3] += pe * v3;
        }
      }
      __syncthreads();
      if (t + 1 < ntiles) { stage(tile0 + TILE, 0); }
      __syncthreads();
    }
  }

  const int slot = s * 4 + warp;
  if (lane == 0) {
    for (int r = 0; r < R; ++r) {
      const int qh_ = h * REP + (r / T);
      const int tq = r % T;
      lse[slot * 48 + qh_ * 3 + tq] = (l[r] > 0.0f) ? (m[r] + log2f(l[r])) : NEG_INF;
    }
  }
  for (int r = 0; r < R; ++r) {
    const int qh_ = h * REP + (r / T);
    const int tq = r % T;
    float* dst = oacc + ((slot * 48 + qh_ * 3 + tq) * V_DIM);
    dst[4*lane+0] = o[r][0]; dst[4*lane+1] = o[r][1];
    dst[4*lane+2] = o[r][2]; dst[4*lane+3] = o[r][3];
  }
}
