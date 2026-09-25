// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W4 SKV-G3: occupancy-first split-KV attention K1 (G2 minus SM_P + K-swizzle).
// G2 measured 291.6 GB/s @1 CTA/SM (36.4KB smem -> fork carveout min-pick 64KB);
// forcing the 100KB carveout gave 484.8 @S=128 with the SAME 36.4KB cubin (2
// CTAs/SM) -> occupancy was the wall. G3 pushes to 3 CTAs/SM:
//   1. SM_P removed: p stays in a per-lane REGISTER after QK; the PV phase
//      broadcasts it with __shfl_sync(FULL, p, i2) (i2 warp-uniform -> full
//      mask, convergent; the QK butterfly already relies on this class).
//      Removes 3KB smem, the smem P round-trip, and the __syncwarp.
//   2. K pitch 512 + xor-swizzle (col ^ (row & 7)) replaces the 528 padding:
//      same 4-phase conflict-free QK reads, 512B smaller. V stays pitch-512
//      plain (PV pattern is naturally 4-phase).
// SM = 32*512 + 32*512 = 32768 B -> shmem_usage = round_up(0x400+32768,128)
// = 33792 -> 3 CTAs = 101376 <= 102400 (100KB carveout, via fork env
// NV_SMEM_CFG=100 NV_SMEM_CFG_NAMES=spk_g3).
// NUMERICS: arithmetic identical to G2/K1S contracts (dot d ascending via
// 32x8 chunk, tile butterfly max 16,8,4,2,1, p = valid ? expf(sc-mn) : 0,
// PV i ascending, mask (l<l1)&&(l<=pos+t), empty-split identity partials)
// -> T=3 rows bit-identical to T=1 rows; partials byte-compatible with K2S.
// -DKNAME -DCTXK -DROWS (3|1) -DS -DCH -DTILE (32 only: lane-per-l) -DLB
// (launch_bounds minBlocksPerMultiprocessor; 0 = unset)
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define RMAX (6*ROWS)
#if LB > 0
  #define LBOUNDS __launch_bounds__(256, LB)
#else
  #define LBOUNDS __launch_bounds__(256)
#endif

// ---------------- smem layout (bytes, compile-time) ----------------
// [K TILE*512 swizzled][V TILE*512 plain]
#define SM_K0   0
#define SM_V0   (TILE*512)
#define SM_BYTES (TILE*1024)
#define KSZ(R, C)  (((R) * 512) + (((C) ^ ((R) & 7)) * 16))   // K swizzled 16B slot
#define VSZ(R, C)  (((R) * 512) + ((C) * 16))                 // V plain 16B slot

#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )

// ---------------- STAGE: KV tile -> smem, 1x DRAM, coalesced 16B ----------------
// chunk c = i*256 + tid; row = c>>5 (l row within tile); c16 = c&31 (16B slot).
// rows are 512B (256 halves); l >= l1 -> zero fill, NO global touch.
#define STAGE(BASEL, KO, VO) _Pragma("unroll") \
  for (int i = 0; i < TILE/8; ++i) { \
    const int c = i*256 + tid; \
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

// G3 per-owned-row QK + tile softmax: p kept in register PR (lane's own l).
#define G3_QKROW(R, Mv, Sv, A8, PR) { \
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

// G3 PV for one owned row: i ascending; p broadcast from lane i2's register.
#define G3_PVROW(A8, PR) _Pragma("unroll") \
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

  const int rr0 = warp;
  float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f;
  float s0 = 0.f, s1 = 0.f, s2 = 0.f;
  float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0}, a2[8] = {0,0,0,0,0,0,0,0};
  float pr0 = 0.f, pr1 = 0.f, pr2 = 0.f;             // lane's own-l p per owned row

  if (l0 < l1) {                                     // CTA-uniform (barriers inside)
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    const int kcur = SM_K0, vcur = SM_V0;
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      const int lpl = tile + lane;
      __syncthreads();                               // prev tile readers done
      STAGE(tile, SM_K0, SM_V0)
      __syncthreads();                               // stage visible
      // ---- QK + tile-granularity online softmax, p in registers ----
      if (rr0 < RMAX)         G3_QKROW(rr0,      m0, s0, a0, pr0)
      if (rr0 + 8 < RMAX)     G3_QKROW(rr0 + 8,  m1, s1, a1, pr1)
      if (rr0 + 16 < RMAX)    G3_QKROW(rr0 + 16, m2, s2, a2, pr2)
      // ---- PV: p broadcast via shuffle (warp-uniform i2, full mask) ----
      if (rr0 < RMAX)         G3_PVROW(a0, pr0)
      if (rr0 + 8 < RMAX)     G3_PVROW(a1, pr1)
      if (rr0 + 16 < RMAX)    G3_PVROW(a2, pr2)
    }
  }

  // ---- partial writes: IDENTICAL to K1S (empty split -> identity partials) ----
  const size_t pb0 = (size_t)(g*S + s)*RMAX;
  if (rr0 < RMAX) {
    const size_t b = pb0 + rr0;
    if (lane == 0) { pm[b] = m0; ps[b] = s0; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a0[j];
  }
  if (rr0 + 8 < RMAX) {
    const size_t b = pb0 + rr0 + 8;
    if (lane == 0) { pm[b] = m1; ps[b] = s1; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a1[j];
  }
  if (rr0 +16 < RMAX) {
    const size_t b = pb0 + rr0 + 16;
    if (lane == 0) { pm[b] = m2; ps[b] = s2; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a2[j];
  }
}
