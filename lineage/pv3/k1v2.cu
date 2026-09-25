// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// Split-KV K1 v2 — 100k probe attention (24 q-heads / 4 kv-groups GQA 6:1, 256d, T=3).
// CTA = (split s, gr = g*3+r). Writes raw masked scores to P; online-softmax
// partials (m2,l,o[256]) per (warp,h6) to ws for the K2 combine.
// Single-buffered smem tiles (cp.async variant device-faults on the dext).
// All bounds baked (no gridDim).
#include <cuda_fp16.h>
#define NEG_INF (__int_as_float(0xff800000))
#ifndef LMAX
#define LMAX 100352
#endif
#ifndef S
#define S 20
#endif
#ifndef TILE
#define TILE 32
#endif
#define C (((LMAX + S - 1) / S))
#define LOG2E 1.4426950216293335f
extern "C" __global__ void __launch_bounds__(256) k1v2(
    float* __restrict__ P, const float* __restrict__ Q, const __half* __restrict__ KV,
    float* __restrict__ ws, const int sp)
{
  const int s  = blockIdx.x / 12;
  const int gr = blockIdx.x % 12;
  const int g = gr / 3, r = gr % 3;
  const int start = s * C;
  const int cend = min(start + C, LMAX);
  const int end_r = min(cend, sp + r + 1);
  const int h0 = g * 6;
  const __half* Kg = KV + (size_t)g * (LMAX * 256);
  const __half* Vg = Kg + (size_t)LMAX * 1024;

  __shared__ __align__(16) __half sK[TILE][256];
  __shared__ __align__(16) __half sV[TILE][256];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nwarp = 256 >> 5;                  // 8

  float q[6][8];
  #pragma unroll
  for (int h6 = 0; h6 < 6; h6++)
    #pragma unroll
    for (int j = 0; j < 8; j++)
      q[h6][j] = Q[(size_t)(h0 + h6) * 768 + r * 256 + lane * 8 + j];

  float m[6], l[6], o[6][8];
  #pragma unroll
  for (int h6 = 0; h6 < 6; h6++) { m[h6] = NEG_INF; l[h6] = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) o[h6][j] = 0.0f; }

  for (int t0 = start; t0 < cend; t0 += TILE) {
    // stage TILE positions x 256 dims of K and V; thread -> row tid/8, 32-dim col
    {
      const int pr = tid >> 3, dc = (tid & 7) << 5;
      const int pbase = t0 + pr;
      if (pbase < cend) {
        const uint4* ksrc = (const uint4*)(Kg + (size_t)pbase * 256 + dc);
        const uint4* vsrc = (const uint4*)(Vg + (size_t)pbase * 256 + dc);
        uint4* kd = (uint4*)&sK[pr][dc];
        uint4* vd = (uint4*)&sV[pr][dc];
        kd[0]=ksrc[0]; kd[1]=ksrc[1]; kd[2]=ksrc[2]; kd[3]=ksrc[3];
        vd[0]=vsrc[0]; vd[1]=vsrc[1]; vd[2]=vsrc[2]; vd[3]=vsrc[3];
      }
    }
    __syncthreads();
    const int npos = min(TILE, cend - t0);
    for (int pi = warp; pi < npos; pi += nwarp) {
      const int p = t0 + pi;
      const bool valid = p < end_r;
      #pragma unroll
      for (int h6 = 0; h6 < 6; h6++) {
        float dpt = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++)
          dpt += q[h6][j] * __half2float(sK[pi][lane * 8 + j]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dpt += __shfl_xor_sync(0xffffffffu, dpt, off);
        const float raw = dpt * 0.0625f;
        if (lane == 0) P[(size_t)(h0 + h6) * (3 * LMAX) + (size_t)r * LMAX + p] = valid ? raw : NEG_INF;
        if (valid) {
          const float s2 = raw * LOG2E;
          if (s2 > m[h6]) {
            const float sc = exp2f(m[h6] - s2);
            l[h6] *= sc;
            #pragma unroll
            for (int j = 0; j < 8; j++) o[h6][j] *= sc;
            m[h6] = s2;
          }
          const float e = exp2f(s2 - m[h6]);
          l[h6] += e;
          #pragma unroll
          for (int j = 0; j < 8; j++) o[h6][j] += e * __half2float(sV[pi][lane * 8 + j]);
        }
      }
    }
    __syncthreads();
  }
  // epilogue -> workspace (per-warp slots; K2 merges all 8 warps x S splits)
  #pragma unroll
  for (int h6 = 0; h6 < 6; h6++) {
    float* w = ws + ((((size_t)(s * 12 + gr) * 6) + h6) * 8 + warp) * 258;
    #pragma unroll
    for (int j = 0; j < 8; j++) w[lane * 8 + j] = o[h6][j];
    if (lane == 0) { w[256] = m[h6]; w[257] = l[h6]; }
  }
}
