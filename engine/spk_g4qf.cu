// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2E KV8-F: int8-KV split-KV attention K1 with FLOAT smem tiles (64KB).
// Same storage layout as spk_g4q.cu (biased uint8 kv + per-(row,32ch) fp16
// scales). Difference: STAGE dequantizes to FLOAT in smem (2x float4 slots
// per row-thread) so the QK/PV hot loops read float4s directly — ZERO unpack
// converts in the dot phase (the measured bottleneck: dot ~0.84ms of the
// 1.31ms launch; stage is only ~0.4ms memory-limited at ~545 GB/s).
// Dequant: PRMT 0x6400-trick -> half2 q = u-128 (exact); __hmul2 by scale
// (single rounding, same as the fp16 path's float-mul-then-f2h); __half22float2
// (exact) -> float4 pairs. 64KB smem/CTA: 1 CTA/SM hard anyway; launch may
// need the fork carveout env (NV_SMEM_CFG=100 NV_SMEM_CFG_NAMES=<name-substr>).
// LAWS: single __shared__ array, 16B-aligned compile-time offsets, no
// blockDim/gridDim reads, full-mask shuffles, sequential loops, TILE=32.
// -DKNAME -DCTXK -DROWS (3|1) -DS -DCH -DTILE (32) -DNW (32)
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define RMAX (6*ROWS)
#ifndef NW
  #define NW 8
#endif
#define NTHR (NW*32)
#define RPMAX ((RMAX + NW - 1) / NW)
#define LBOUNDS __launch_bounds__(NTHR)

// ---------------- smem layout: K FLOAT tile (32KB) + V HALF tile (16KB) = 48KB max
// K rows are 1KB (bank-neutral stride!) -> xor-swizzle at float4-slot
// granularity: KSLOT(R,F) = R*64 + (F ^ (R&7)); V plain 16B half slots.
#define SM_K0   0
#define SM_V0   (TILE*1024)
#define SM_BYTES (TILE*1536)
#define KSLOT(R, F) ( (((R) * 64) + ((F) ^ ((R) & 7))) * 16 )
#define VSLOT(R, C)  ( ((R) * 512) + ((C) * 16) )

#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )

// biased-uint8 x8 + scale -> 8 floats (two float4)
__device__ __forceinline__ void dq8f(const int2 iv, const __half s, float4& a, float4& b) {
  const __half2 s2 = __half2half2(s);
  const __half2 c2 = (__half2)__float2half2_rn(1152.f);
  const unsigned x = (unsigned)iv.x, y = (unsigned)iv.y;
  const unsigned p0 = __byte_perm(x, 0x64646464u, 0x5140), p1 = __byte_perm(x, 0x64646464u, 0x7362);
  const unsigned p2 = __byte_perm(y, 0x64646464u, 0x5140), p3 = __byte_perm(y, 0x64646464u, 0x7362);
  const float2 f0 = __half22float2(__hmul2(__hsub2(*(const __half2*)&p0, c2), s2));
  const float2 f1 = __half22float2(__hmul2(__hsub2(*(const __half2*)&p1, c2), s2));
  const float2 f2 = __half22float2(__hmul2(__hsub2(*(const __half2*)&p2, c2), s2));
  const float2 f3 = __half22float2(__hmul2(__hsub2(*(const __half2*)&p3, c2), s2));
  a = make_float4(f0.x, f0.y, f1.x, f1.y);
  b = make_float4(f2.x, f2.y, f3.x, f3.y);
}

// int8x8 + scale -> 16B container of 8 halves (V tile)
#define DQH2(R, S2) __hmul2(__hsub2((R), (__half2)__float2half2_rn(1152.f)), (S2))
__device__ __forceinline__ float4 dq8h(const int2 iv, const __half s) {
  const __half2 s2 = __half2half2(s);
  const unsigned x = (unsigned)iv.x, y = (unsigned)iv.y;
  const unsigned p0 = __byte_perm(x, 0x64646464u, 0x5140), p1 = __byte_perm(x, 0x64646464u, 0x7362);
  const unsigned p2 = __byte_perm(y, 0x64646464u, 0x5140), p3 = __byte_perm(y, 0x64646464u, 0x7362);
  float4 o;
  *(__half2*)&o.x = DQH2(*(const __half2*)&p0, s2);
  *(__half2*)&o.y = DQH2(*(const __half2*)&p1, s2);
  *(__half2*)&o.z = DQH2(*(const __half2*)&p2, s2);
  *(__half2*)&o.w = DQH2(*(const __half2*)&p3, s2);
  return o;
}

// ---------------- STAGE: int8 KV tile -> FLOAT smem, coalesced ----------------
#define STAGE(BASEL, KO, VO) _Pragma("unroll") \
  for (int i = 0; i < TILE/NW; ++i) { \
    const int c = i*NTHR + tid; \
    const int row = c >> 5, c16 = c & 31; \
    const int l = (BASEL) + row; \
    float4 k0 = make_float4(0.f,0.f,0.f,0.f), k1 = make_float4(0.f,0.f,0.f,0.f); \
    float4 vh = make_float4(0.f,0.f,0.f,0.f); \
    if (l < l1) { \
      const int2 ki = *(const int2*)(Kc8 + (size_t)l*256 + c16*8); \
      const int2 vi = *(const int2*)(Vc8 + (size_t)l*256 + c16*8); \
      dq8f(ki, Ksc[(size_t)l*8 + (c16 >> 2)], k0, k1); \
      vh = dq8h(vi, Vsc[(size_t)l*8 + (c16 >> 2)]); \
    } \
    *(float4*)(SM + (KO) + KSLOT(row, c16*2)) = k0; \
    *(float4*)(SM + (KO) + KSLOT(row, c16*2+1)) = k1; \
    *(float4*)(SM + (VO) + VSLOT(row, c16)) = vh; \
  }

#define BUTTERFLY_MAX(V) _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) (V) = fmaxf((V), __shfl_xor_sync(FULL, (V), o));
#define BUTTERFLY_SUM(V) _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) (V) += __shfl_xor_sync(FULL, (V), o);

#define G4_QKROW(R, Mv, Sv, A8, PR) { \
  const int t_ = (R) % ROWS; \
  const bool vld = (lpl < l1) && (lpl <= pos + t_); \
  float sc = 0.f; \
  const float* qb = qw + (size_t)QROWIDX(R)*256; \
  _Pragma("unroll") \
  for (int c = 0; c < 32; ++c) { \
    const float4 kf0 = *(const float4*)(SM + kcur + KSLOT(lane, c*2)); \
    const float4 kf1 = *(const float4*)(SM + kcur + KSLOT(lane, c*2+1)); \
    const float4 qa = *(const float4*)(qb + c*8); \
    const float4 qcx = *(const float4*)(qb + c*8 + 4); \
    sc += qa.x*kf0.x; sc += qa.y*kf0.y; sc += qa.z*kf0.z; sc += qa.w*kf0.w; \
    sc += qcx.x*kf1.x; sc += qcx.y*kf1.y; sc += qcx.z*kf1.z; sc += qcx.w*kf1.w; \
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

#define G4_PVROW(A8, PR) _Pragma("unroll") \
  for (int i2 = 0; i2 < TILE; ++i2) { \
    const float p = __shfl_sync(FULL, (PR), i2); \
    const float4 vf4 = *(const float4*)(SM + vcur + VSLOT(i2, lane)); \
    const __half2* hv = (const __half2*)&vf4; \
    const float2 g0 = __half22float2(hv[0]), g1 = __half22float2(hv[1]), g2 = __half22float2(hv[2]), g3 = __half22float2(hv[3]); \
    const float vv[8] = {g0.x, g0.y, g1.x, g1.y, g2.x, g2.y, g3.x, g3.y}; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) (A8)[j] += p * vv[j]; \
  }

extern "C" __global__ void LBOUNDS KNAME(
    const unsigned char* __restrict__ kv, const __half* __restrict__ sc, const float* __restrict__ qw,
    const int* __restrict__ pos_slot, float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
{
  const int g = blockIdx.x / S;
  const int s = blockIdx.x % S;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int pos = pos_slot[0];                       // read ONCE
  const int l0 = s * CH;
  const int l1 = (l0 + CH) < (pos + ROWS) ? (l0 + CH) : (pos + ROWS);
  const unsigned char* Kc8 = kv + (size_t)g * (CTXK*256);
  const unsigned char* Vc8 = kv + (size_t)(4 + g) * (CTXK*256);
  const __half* Ksc = sc + (size_t)g * (CTXK*8);
  const __half* Vsc = sc + (size_t)(4 + g) * (CTXK*8);
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

  // ---- partial writes: IDENTICAL (empty split -> identity partials) ----
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
