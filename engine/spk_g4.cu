// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W4 SKV-G4: FAT-CTA split-KV attention K1 (G3 + warp-count parameter).
// Measured context (see W2C): the dext appears hard-limited to 1 CTA/SM
// (carveout override changed nothing at true S; G3 3-CTA-capable ~= G2), and
// G3-LB3 vs LB0 (1.31 vs 1.86 ms, SAME regs/smem) shows the kernel is
// latency-chain-bound -> the occupancy lever on a 1-CTA/SM dext is MORE WARPS
// PER CTA. G4 = G3 (P in registers + K xor-swizzle pitch 512) with:
//   -DNW warps/CTA (8 -> 256 thr control, 16 -> 512 thr fat CTA)
//   row ownership generalized to a sequential loop: warp w owns rows
//   w, w+NW, w+2NW... (RPMAX = ceil(RMAX/NW) register sets per warp)
//   STAGE iters = TILE/NW (threads scale, bytes constant)
// Numerics: per-row op sequence IDENTICAL to G2/G3 (dot d ascending 32x8
// chunks, tile butterfly 16,8,4,2,1, p = valid ? expf(sc-mn) : 0, PV i
// ascending, mask (l<l1)&&(l<=pos+t)); rows are warp-independent so partials
// are bit-identical regardless of the row->warp map. Empty-split identity
// partials unchanged. K2S/KPRE interfaces unchanged.
// LAWS: single __shared__ array + compile-time 16B-aligned offsets; no
// blockDim/gridDim reads; flat guards; full-mask shuffles in warp-convergent
// code; sequential loops; TILE=32 (lane-per-l).
// -DKNAME -DCTXK -DROWS (3|1) -DS -DCH -DTILE (32) -DNW (8|16)
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define RMAX (6*ROWS)
#ifndef NW
  #define NW 8
#endif
#define NTHR (NW*32)
#define RPMAX ((RMAX + NW - 1) / NW)
#define LBOUNDS __launch_bounds__(NTHR)

// ---------------- smem layout (bytes, compile-time) ----------------
#define SM_K0   0
#define SM_V0   (TILE*512)
#define SM_BYTES (TILE*1024)
#define KSZ(R, C)  (((R) * 512) + (((C) ^ ((R) & 7)) * 16))   // K swizzled 16B slot
#define VSZ(R, C)  (((R) * 512) + ((C) * 16))                 // V plain 16B slot

#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )

// ---------------- STAGE: KV tile -> smem, 1x DRAM, coalesced 16B ----------------
#define STAGE(BASEL, KO, VO) _Pragma("unroll") \
  for (int i = 0; i < TILE/NW; ++i) { \
    const int c = i*NTHR + tid; \
    const int row = c >> 5, c16 = c & 31; \
    const int l = (BASEL) + row; \
    float4 kk = make_float4(0.f, 0.f, 0.f, 0.f), vv = make_float4(0.f, 0.f, 0.f, 0.f); \
    if (l < l1) { \
      kk = *(const float4*)(Kc + (size_t)l*256 + c16*8); \
      vv = *(const float4*)(Vc + (size_t)l*256 + c16*8); \
    } \
    *(float4*)(SM + (KO) + KSZ(row, c16)) = kk; \
    *(float4*)(SM + (VO) + VSZ(row, c16)) = vv; \
  }

#define BUTTERFLY_MAX(V) _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) (V) = fmaxf((V), __shfl_xor_sync(FULL, (V), o));
#define BUTTERFLY_SUM(V) _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) (V) += __shfl_xor_sync(FULL, (V), o);

// G4 per-owned-row QK + tile softmax: p kept in register PR (lane's own l).
#define G4_QKROW(R, Mv, Sv, A8, PR) { \
  const int t_ = (R) % ROWS; \
  const bool vld = (lpl < l1) && (lpl <= pos + t_); \
  float sc = 0.f; \
  const float* qb = qw + (size_t)QROWIDX(R)*256; \
  _Pragma("unroll") \
  for (int c = 0; c < 32; ++c) { \
    const float4 kf4 = *(const float4*)(SM + kcur + KSZ(lane, c)); \
    const __half2* hk = (const __half2*)&kf4; \
    const float2 k0 = __half22float2(hk[0]), k1 = __half22float2(hk[1]), k2 = __half22float2(hk[2]), k3 = __half22float2(hk[3]); \
    const float4 qa = *(const float4*)(qb + c*8); \
    const float4 qcx = *(const float4*)(qb + c*8 + 4); \
    const float qq[8] = {qa.x, qa.y, qa.z, qa.w, qcx.x, qcx.y, qcx.z, qcx.w}; \
    const float kk[8] = {k0.x, k0.y, k1.x, k1.y, k2.x, k2.y, k3.x, k3.y}; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) sc += qq[j] * kk[j]; \
  } \
  float scm = vld ? sc : -1e30f; \
  BUTTERFLY_MAX(scm) \
  const float mn = fmaxf((Mv), scm); \
  const float cor = expf((Mv) - mn); \
  (Sv) *= cor; \
  _Pragma("unroll") for (int j = 0; j < 8; ++j) (A8)[j] *= cor; \
  (Mv) = mn; \
  const float p = vld ? expf(sc - mn) : 0.f; \
  float ps = p; \
  BUTTERFLY_SUM(ps) \
  (Sv) += ps; \
  (PR) = p; \
}

// G4 PV for one owned row: i ascending; p broadcast from lane i2's register.
#define G4_PVROW(A8, PR) _Pragma("unroll") \
  for (int i2 = 0; i2 < TILE; ++i2) { \
    const float p = __shfl_sync(FULL, (PR), i2); \
    const float4 vf4 = *(const float4*)(SM + vcur + VSZ(i2, lane)); \
    const __half2* hv = (const __half2*)&vf4; \
    const float2 g0 = __half22float2(hv[0]), g1 = __half22float2(hv[1]), g2 = __half22float2(hv[2]), g3 = __half22float2(hv[3]); \
    const float vv[8] = {g0.x, g0.y, g1.x, g1.y, g2.x, g2.y, g3.x, g3.y}; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) (A8)[j] += p * vv[j]; \
  }

extern "C" __global__ void LBOUNDS KNAME(
    const __half* __restrict__ kv, const float* __restrict__ qw, const int* __restrict__ pos_slot,
    float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
{
  const int g = blockIdx.x / S;
  const int s = blockIdx.x % S;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int pos = pos_slot[0];                       // read ONCE
  const int l0 = s * CH;
  const int l1 = (l0 + CH) < (pos + ROWS) ? (l0 + CH) : (pos + ROWS);
  const __half* Kc = kv + (size_t)g * (CTXK*256);
  const __half* Vc = kv + (size_t)(4 + g) * (CTXK*256);
  __shared__ __align__(16) char SM[SM_BYTES];

  float msv[RPMAX], ssv[RPMAX], aav[RPMAX][8], prv[RPMAX];
  _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {
    msv[ri] = -1e30f; ssv[ri] = 0.f; prv[ri] = 0.f;
    _Pragma("unroll") for (int j = 0; j < 8; ++j) aav[ri][j] = 0.f;
  }

  if (l0 < l1) {                                     // CTA-uniform (barriers inside)
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    const int kcur = SM_K0, vcur = SM_V0;
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      const int lpl = tile + lane;
      __syncthreads();                               // prev tile readers done
      STAGE(tile, SM_K0, SM_V0)
      __syncthreads();                               // stage visible
      _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {   // QK all owned rows
        const int row = warp + ri*NW;
        if (row < RMAX) G4_QKROW(row, msv[ri], ssv[ri], aav[ri], prv[ri])
      }
      _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {   // PV all owned rows
        const int row = warp + ri*NW;
        if (row < RMAX) G4_PVROW(aav[ri], prv[ri])
      }
    }
  }

  // ---- partial writes: IDENTICAL to K1S (empty split -> identity partials) ----
  const size_t pb0 = (size_t)(g*S + s)*RMAX;
  _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {
    const int row = warp + ri*NW;
    if (row < RMAX) {
      const size_t b = pb0 + row;
      if (lane == 0) { pm[b] = msv[ri]; ps[b] = ssv[ri]; }
      _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = aav[ri][j];
    }
  }
}
