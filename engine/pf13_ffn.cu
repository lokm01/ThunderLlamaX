// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P13 kernel B — the PERSISTENT-CTA fused FFN tile probe.
// 82 CTAs (grid literal 82 = one/SM, exactly one wave); NPARC = NBLK*NGRID
// work parcels (block, n-tile) distributed compile-time: parcel p is handled
// by CTA p%82 via a sequential stride-82 loop (LITERAL constant — the
// gridDim=0 law). Each CTA streams its weight parcels CONTINUOUSLY: the
// packed7 unit stream with a WRING-deep register ring (loads lead decodes by
// WRING-2 chunks — persistent 1-CTA/SM allows >128 regs, which the launch-
// bound P11-G2 could NOT afford), x re-staged per chunk (L2-hot), decode +
// mma order VERBATIM from pf_gemm3.cu REPACK=1 FFN m32 -> bit-identical
// outputs expected vs pfg3_ffn_r7_m32_nw8k128.
// LAWS: flat indexing, no gridDim/blockDim reads, hardcoded sizes, sequential
// loops, full-warp masks, single 16B-aligned smem (43520B = shipped m32
// profile), uG/uU/xp only compile-time indexed (unrolled), acc regs only.
// Build: -DKNAME -DWRING -DNBLK -DKDIM -DNDIM -DNTHR -DNTILE -DKCH -DMTILE
#include <cuda_fp16.h>

#define FULL 0xffffffffu
#define XS_LD (KCH + 8)
#define WS_LD (KCH + 8)
#define NWARP (NTHR / 32)
#define RPT (NTILE / NWARP)
#if RPT != 8
#error "pf13_ffn: NTILE/NWARP must be 8 (row-group layout)"
#endif
#define NCL (KCH / 32)
#define KSTEPS (KCH / 16)
#define NCH (KDIM / KCH)
#define NGRID (NDIM / NTILE)
#define NGRP (NTILE / 8)
#define MRG (MTILE / 16)
#define FGN (MTILE / 16)
#define XTPR ((MTILE * KCH + NTHR * 4 - 1) / (NTHR * 4))
#define NCTA 82
#define NPARC (NBLK * NGRID)
#define WBLK ((size_t)NGRID * NGRP * NCH * 32)   // uint4 units per plane per block
#if MTILE != 32
#error "pf13_ffn: MTILE 32 (one M-block; the chunk shape)"
#endif
#if NCH % WRING
#error "pf13_ffn: WRING must divide NCH"
#endif

__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

__device__ __forceinline__ void hmma16816(float &c0, float &c1, float &c2, float &c3,
                                          const unsigned a0, const unsigned a1,
                                          const unsigned a2, const unsigned a3,
                                          const unsigned b0, const unsigned b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// P4 dq_iq3_r VERBATIM (register-form IQ3 decode; packed7 sidx field cc&3)
__device__ __forceinline__ void dq_iq3_r(const float* gridf, const unsigned int qv,
    const unsigned int swv, const unsigned int dv, const int cc, __half* w8) {
  const float d = __half2float(__ushort_as_half((unsigned short)dv));
  const float db = d * (((float)(swv >> 28)) + 0.5f) * 0.5f;
  const unsigned int sidx = (swv >> (7u * (unsigned int)(cc & 3))) & 0x7Fu;
  const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
  const float4 g0 = *((const float4*)(gridf + ((qv & 0xFFu) << 2)));
  const float4 g1 = *((const float4*)(gridf + ((qv >> 8) << 2)));
  const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
  const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
  const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
  const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
  w8[0] = __float2half(db*g0.x*sg0); w8[1] = __float2half(db*g0.y*sg1);
  w8[2] = __float2half(db*g0.z*sg2); w8[3] = __float2half(db*g0.w*sg3);
  w8[4] = __float2half(db*g1.x*sg4); w8[5] = __float2half(db*g1.y*sg5);
  w8[6] = __float2half(db*g1.z*sg6); w8[7] = __float2half(db*g1.w*sg7);
}

__device__ __forceinline__ void dq_unit(const float* gridf, const uint4 U, const int cc, __half* w8) {
  const unsigned int qv = (cc == 0) ? (U.x & 0xFFFFu) : (cc == 1) ? (U.x >> 16)
                       : (cc == 2) ? (U.y & 0xFFFFu) : (U.y >> 16);
  dq_iq3_r(gridf, qv, U.z, U.w & 0xFFFFu, cc, w8);
}

// x regs -> smem (xs), per-parcel xb base
#define XCM2(X) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR) * 4; \
    if (lin < MTILE*KCH) { \
      const int m = lin / KCH, kk = lin % KCH; \
      *(uint2*)(xs + (size_t)m*XS_LD + kk) = X[t]; \
    } \
  } \
} while (0)

#define XPR2(X, KC) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR) * 4; \
    if (lin < MTILE*KCH) { \
      const int m = lin / KCH, kk = lin % KCH; \
      X[t] = *(const uint2*)(xb + (size_t)m*KDIM + (size_t)(KC) + kk); \
    } \
  } \
} while (0)

// decode the lane's unit reg -> this lane's ws rows
#define WCM2(U, WSB) do { \
  __half* wsP_ = (WSB); \
  _Pragma("unroll") \
  for (int cc = 0; cc < NCL; ++cc) { \
    __half w8[8]; \
    dq_unit(gridf, U, cc, w8); \
    const int kk = (qc_*NCL + cc)*8; \
    _Pragma("unroll") \
    for (int j = 0; j < 8; ++j) wsP_[(size_t)(warp*8 + r_)*WS_LD + kk + j] = w8[j]; \
  } \
} while (0)

#define MMAR() do { \
  const int g = lane >> 2, tp = (lane & 3) * 2; \
  _Pragma("unroll") \
  for (int s = 0; s < KSTEPS; ++s) { \
    const int kb = s*16; \
    _Pragma("unroll") \
    for (int rg = 0; rg < MRG; ++rg) { \
      const unsigned a0 = *(const unsigned*)(xs + (size_t)(rg*16 + g)*XS_LD + kb + tp); \
      const unsigned a1 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp); \
      const unsigned a2 = *(const unsigned*)(xs + (size_t)(rg*16 + g)*XS_LD + kb + tp + 8); \
      const unsigned a3 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp + 8); \
      const __half* wr = ws + (size_t)(warp*8 + g)*WS_LD + kb + tp; \
      const unsigned b0 = *(const unsigned*)(wr); \
      const unsigned b1 = *(const unsigned*)(wr + 8); \
      hmma16816(ACC(0, rg, 0), ACC(0, rg, 1), ACC(0, rg, 2), ACC(0, rg, 3), a0, a1, a2, a3, b0, b1); \
    } \
    _Pragma("unroll") \
    for (int rg = 0; rg < FGN; ++rg) { \
      const unsigned a0 = *(const unsigned*)(xs + (size_t)(rg*16 + g)*XS_LD + kb + tp); \
      const unsigned a1 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp); \
      const unsigned a2 = *(const unsigned*)(xs + (size_t)(rg*16 + g)*XS_LD + kb + tp + 8); \
      const unsigned a3 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp + 8); \
      const __half* wr = ws + NTILE*WS_LD + (size_t)(warp*8 + g)*WS_LD + kb + tp; \
      const unsigned b0 = *(const unsigned*)(wr); \
      const unsigned b1 = *(const unsigned*)(wr + 8); \
      hmma16816(ACC(1, rg, 0), ACC(1, rg, 1), ACC(1, rg, 2), ACC(1, rg, 3), a0, a1, a2, a3, b0, b1); \
    } \
  } \
} while (0)

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
    const unsigned char* __restrict__ w2,
    const float* __restrict__ gridf, const __half* __restrict__ x16,
    __half* __restrict__ out16)
{
  __shared__ __align__(16) __half sm[MTILE * XS_LD + 2 * NTILE * WS_LD];
  float acc[2][MRG][4];
#define ACC(pl, rg, j) acc[pl][rg][j]
  __half* xs = sm;
  __half* ws = sm + MTILE * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int r_ = lane >> 2, qc_ = lane & 3;

  // ===== persistent parcel loop: CTA c handles parcels c, c+82, c+164, ... =====
  for (int parc = blockIdx.x; parc < NPARC; parc += NCTA) {
    __syncthreads();   // smem WAR: prior parcel's final MMAR readers must drain
    const int b = parc / NGRID;
    const int nb = parc - b * NGRID;
    const uint4* w1u4 = (const uint4*)(w1 + (size_t)b * WBLK * 16);
    const uint4* w2u4 = (const uint4*)(w2 + (size_t)b * WBLK * 16);
    const __half* xb = x16 + (size_t)b * MTILE * KDIM;
    __half* ob = out16 + (size_t)b * MTILE * NDIM;
    const size_t goff = ((size_t)(nb * NGRP + warp) * NCH) * 32 + lane;

    uint4 uG[WRING], uU[WRING];
    uint2 xp[XTPR];

    _Pragma("unroll")
    for (int pl = 0; pl < 2; ++pl)
      _Pragma("unroll")
      for (int rg = 0; rg < MRG; ++rg)
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) ACC(pl, rg, j) = 0.f;

    // prologue: chunks 0..WRING-2 into slots; decode chunk0 -> ws; stage x0
    _Pragma("unroll")
    for (int j = 0; j < WRING - 1; ++j) {
      uG[j] = w1u4[goff + (size_t)j * 32];
      uU[j] = w2u4[goff + (size_t)j * 32];
    }
    XPR2(xp, 0);
    WCM2(uG[0], ws);
    WCM2(uU[0], ws + NTILE * WS_LD);
    XCM2(xp);
    __syncthreads();

    // main: at chunk c issue loads for chunk c+WRING-1 (lead = WRING-2 chunks),
    // mma the staged chunk c, decode chunk c+1 in the post-mma bubble
    _Pragma("unroll 1")
    for (int cb = 0; cb < NCH; cb += WRING) {
      _Pragma("unroll")
      for (int j = 0; j < WRING; ++j) {
        const int c = cb + j;
        if (c + WRING - 1 < NCH) {
          uG[(j + WRING - 1) % WRING] = w1u4[goff + (size_t)(c + WRING - 1) * 32];
          uU[(j + WRING - 1) % WRING] = w2u4[goff + (size_t)(c + WRING - 1) * 32];
        }
        if (c + 1 < NCH) XPR2(xp, (c + 1) * KCH);
        MMAR();
        __syncthreads();
        if (c + 1 < NCH) {
          WCM2(uG[(j + 1) % WRING], ws);
          WCM2(uU[(j + 1) % WRING], ws + NTILE * WS_LD);
          XCM2(xp);
        }
        __syncthreads();
      }
    }

    // epilogue (per row-group, the classic c-frag map) — verbatim FFN silu-mul
    {
      const int g = lane >> 2, tp = (lane & 3) * 2;
      const int n = nb * NTILE + warp*8 + tp;
      _Pragma("unroll")
      for (int rg = 0; rg < MRG; ++rg) {
        const int m0 = rg*16 + g;
        const __half hg0 = (__half)ACC(0, rg, 0), hg1 = (__half)ACC(0, rg, 1);
        const __half hg2 = (__half)ACC(0, rg, 2), hg3 = (__half)ACC(0, rg, 3);
        const __half hn0 = hsilu_h(hg0), hn1 = hsilu_h(hg1), hn2 = hsilu_h(hg2), hn3 = hsilu_h(hg3);
        *(__half2*)(ob + (size_t)m0*NDIM + n)     = __halves2half2(__hmul(hn0, (__half)ACC(1, rg, 0)), __hmul(hn1, (__half)ACC(1, rg, 1)));
        *(__half2*)(ob + (size_t)(m0+8)*NDIM + n) = __halves2half2(__hmul(hn2, (__half)ACC(1, rg, 2)), __hmul(hn3, (__half)ACC(1, rg, 3)));
      }
    }
  }
}
