// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// T=1 draft-PV split-KV. out[d + h*256] = (sum_{p<sp+1} P[h][p] * V[g][p][d]) * sigmoid(gate[d + h*512 + 256])
// P=[24][100349] row-major (pre-normalized), V slab [(h/6)*L*256 + L*1024 + p*256 + d].
// K1: CTA=(split s, group g) x 256 threads (8 warps, warp->position of an 8-tile);
// 6 heads; plain partial sums to PER-WARP ws slots (the missing piece — all warps
// writing one slot was a lost-update race). No softmax (P already normalized).
#include <cuda_fp16.h>
#ifndef LMAX
#define LMAX 100352
#endif
#ifndef PS
#define PS 100349
#endif
#ifndef S
#define S 24
#endif
#define C (((LMAX + S - 1) / S))
extern "C" __global__ void __launch_bounds__(256) k1t1(
    float* __restrict__ P, const __half* __restrict__ KV, float* __restrict__ ws, const int sp)
{
  const int s = blockIdx.x >> 2;
  const int g = blockIdx.x & 3;
  const int start = s * C;
  const int cend = min(start + C, sp + 1);
  const __half* Vg = KV + (size_t)g * (LMAX * 256) + (size_t)LMAX * 1024;
  __shared__ __align__(16) __half sV[8][256];
  const int tid = threadIdx.x, lane = tid & 31;
  const int spr = tid >> 5, sdc = (tid & 31) << 3;
  struct __align__(16) u4 { unsigned x, y, z, w; };
  float acc[6][8];
  #pragma unroll
  for (int h6 = 0; h6 < 6; h6++)
    #pragma unroll
    for (int j = 0; j < 8; j++) acc[h6][j] = 0.0f;
  const int end = cend - start;
  for (int t0 = 0; t0 < end; t0 += 8) {
    if (t0 + spr < end) *(u4*)&sV[spr][sdc] = *(const u4*)(Vg + (size_t)(start + t0 + spr) * 256 + sdc);
    __syncthreads();
    const int npos = min(8, end - t0);
    const int pi = tid >> 5;
    if (pi < npos) {
      const int p = start + t0 + pi;
      #pragma unroll
      for (int h6 = 0; h6 < 6; h6++) {
        const float pv = P[(size_t)(g * 6 + h6) * PS + p];
        #pragma unroll
        for (int j = 0; j < 8; j++)
          acc[h6][j] += pv * __half2float(sV[pi][(lane << 3) + j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int h6 = 0; h6 < 6; h6++) {
    float* w = ws + ((((size_t)(s * 4 + g) * 6) + h6) * 8 + (tid >> 5)) * 256;
    #pragma unroll
    for (int j = 0; j < 8; j++) w[(lane << 3) + j] = acc[h6][j];
  }
}
