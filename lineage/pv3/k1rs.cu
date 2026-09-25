// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// Split-KV K1-RS (row-shared) — CTA = (split s, group g), covers ALL 18 rows (6 heads x T=3).
// K/V tiles read ONCE per (s,g) = 1x unique traffic (was 3x with per-row CTAs).
// Two-phase tiles: warps score positions into smem, then row-owning warps run the
// online softmax + o accumulation (no cross-warp races). Lane dims = lane + 32*j
// (bank-conflict-free q/o/sc access). q/o live in smem (fp32 exactness kept).
#include <cuda_fp16.h>
#define NEG_INF (__int_as_float(0xff800000))
#ifndef LMAX
#define LMAX 100352
#endif
#ifndef S
#define S 41
#endif
#define C (((LMAX + S - 1) / S))
#define LOG2E 1.4426950216293335f
extern "C" __global__ void __launch_bounds__(256) k1rs(
    float* __restrict__ P, const float* __restrict__ Q, const __half* __restrict__ KV,
    float* __restrict__ ws, const int sp)
{
  const int s = blockIdx.x >> 2;
  const int g = blockIdx.x & 3;
  const int start = s * C;
  const int cend = min(start + C, LMAX);
  const int h0 = g * 6;
  const __half* Kg = KV + (size_t)g * (LMAX * 256);
  const __half* Vg = Kg + (size_t)LMAX * 1024;

  __shared__ __align__(16) float sQ[18][256];
  __shared__ __align__(16) float sO[18][256];
  __shared__ __align__(16) float sC2[18][8];
  __shared__ __align__(16) __half sK[8][256];
  __shared__ __align__(16) __half sV[8][256];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // load q: row = h6*3 + r for (h6, r); dims = lane + 32j
  {
    for (int i = tid; i < 18 * 256; i += 256) {
      const int row = i >> 8, d = i & 255;
      const int h6 = row / 3, r = row % 3;
      sQ[row][d] = Q[(size_t)(h0 + h6) * 768 + r * 256 + d];
      sO[row][d] = 0.0f;
    }
  }
  // row ownership: warp w owns rows w, w+8, (w+16 if < 18); per-row online state in registers
  float mrow[3], lrow[3];
  const int nown = (warp < 2) ? 3 : 2;
  #pragma unroll
  for (int i = 0; i < 3; i++) { mrow[i] = NEG_INF; lrow[i] = 0.0f; }

  // register prefetch of tile 0 (16B/thread/tensor)
  const int spr = tid >> 5, sdc = (tid & 31) << 3;
  unsigned kp, vp;
  bool kvalid = start + spr < cend;
  if (kvalid) {
    kp = *(const unsigned*)(Kg + (size_t)(start + spr) * 256 + sdc);
    vp = *(const unsigned*)(Vg + (size_t)(start + spr) * 256 + sdc);
  }

  __syncthreads();

  for (int t0 = start; t0 < cend; t0 += 8) {
    // store prefetched 16B (8 halfs)
    if (kvalid) {
      *(unsigned*)&sK[spr][sdc] = kp;
      *(unsigned*)&sV[spr][sdc] = vp;
    }
    __syncthreads();
    const int tn = t0 + 8;
    kvalid = tn + spr < cend;
    if (kvalid) {
      kp = *(const unsigned*)(Kg + (size_t)(tn + spr) * 256 + sdc);
      vp = *(const unsigned*)(Vg + (size_t)(tn + spr) * 256 + sdc);
    }
    const int npos = min(8, cend - t0);
    // ---- phase 1: score (warp w -> position w): all 18 rows ----
    const int pi = warp;
    if (pi < npos) {
      const int p = t0 + pi;
      #pragma unroll
      for (int row = 0; row < 18; row++) {
        const int r = row % 3;
        const bool valid = p < sp + r + 1;
        float dpt = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          const int d = lane + (j << 5);
          dpt += sQ[row][d] * __half2float(sK[pi][d]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dpt += __shfl_xor_sync(0xffffffffu, dpt, off);
        const float raw = dpt * 0.0625f;
        if (lane == 0) {
          sC2[row][pi] = valid ? raw : NEG_INF;
          P[(size_t)(h0 + row / 3) * (3 * LMAX) + (size_t)r * LMAX + p] = valid ? raw : NEG_INF;
        }
      }
    }
    __syncthreads();
    // ---- phase 2: row-owning warps update online state + o ----
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      if (i >= nown) break;
      const int row = warp + (i << 3);
      const int r = row % 3, h6 = row / 3;
      const int end_r = min(cend, sp + r + 1);
      #pragma unroll 4
      for (int pi2 = 0; pi2 < 8; pi2++) {
        if (pi2 >= npos) break;
        const int p = t0 + pi2;
        if (p >= end_r) continue;
        const float s2 = sC2[row][pi2] * LOG2E;
        if (s2 > mrow[i]) {
          const float sc = exp2f(mrow[i] - s2);
          lrow[i] *= sc;
          #pragma unroll
          for (int j = 0; j < 8; j++) {
            const int d = lane + (j << 5);
            sO[row][d] *= sc;
          }
          mrow[i] = s2;
        }
        const float e = exp2f(s2 - mrow[i]);
        lrow[i] += e;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          const int d = lane + (j << 5);
          sO[row][d] += e * __half2float(sV[pi2][d]);
        }
      }
    }
    __syncthreads();
  }
  // ---- epilogue: ws[(s*4+g)*18 + row] = [o 256 | m | l] ----
  {
    float* wbase = ws + ((size_t)(s * 4 + g) * 18) * 258;
    for (int i = tid; i < 18 * 256; i += 256) {
      const int row = i >> 8, d = i & 255;
      wbase[(size_t)row * 258 + d] = sO[row][d];
    }
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      if (i >= nown) break;
      const int row = warp + (i << 3);
      if (lane == 0) {
        wbase[(size_t)row * 258 + 256] = mrow[i];
        wbase[(size_t)row * 258 + 257] = lrow[i];
      }
    }
  }
}
