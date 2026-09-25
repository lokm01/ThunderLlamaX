// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// T=1 draft-PV fused split-KV kernel (Route A: last-CTA combine).
// Replaces stock r_12_16_16_2_28mtp_sp2B129 (5-arg fused PV) 1:1.
// Phase 1 (all CTAs): partial sums per (s,g,h6,warp) to ws (baked VA).
// Phase 2 (last CTA only, via self-resetting atomic counter): combine all
// heads x dims + sigmoid gate -> out. ws layout: (S*4*6*8)*256 floats + [0] counter at the end.
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
extern "C" __global__ void __launch_bounds__(256) k1t1f(
    float* __restrict__ out, float* __restrict__ P, const __half* __restrict__ KV,
    float* __restrict__ gate, float* __restrict__ ws, const int sp)
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
  // ---- last-CTA combine ----
  float* counter = ws + (size_t)S * 4 * 6 * 8 * 256;
  __threadfence();
  __shared__ unsigned is_last;
  if (tid == 0) is_last = (atomicAdd(counter, 1u) == (unsigned)(S * 4 - 1)) ? 1u : 0u;
  __syncthreads();
  if (!is_last) return;
  if (tid == 0) *counter = 0u;   // self-reset for the next launch
  // combine: 24 heads x 256 dims over S*8 partials each; 256 threads stride dims
  for (int h = 0; h < 24; h++) {
    const int g2 = h / 6, h6 = h % 6;
    float a = 0.0f;
    for (int s2 = 0; s2 < S; s2++)
      #pragma unroll 8
      for (int wv = 0; wv < 8; wv++)
        a += ws[(((size_t)(s2 * 4 + g2) * 6) + h6) * 8 * 256 + (size_t)wv * 256 + tid];
    const float gt = gate[tid + (size_t)h * 512 + 256];
    out[tid + (size_t)h * 256] = a * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293335f)));
  }
}
