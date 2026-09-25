// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P7-B pf_gemm3: the W-RESIDENT GEMM generation — repacked-layout (packed7)
// IQ3 weights + M joins the grid (M=MTILE rows/CTA) + DBUF register ring.
//
// REPACK=1 (IQ3_XXS packed classes only): stage W reads the pack_w7 layout
//   wr7[group][chunk][unit r*4+c] — each lane owns ONE 16B unit holding its
//   whole decode input set for the chunk (q u16 xNCL, sw u32, d u16 — the
//   words VERBATIM). A warp's chunk slice = 32 consecutive uint4 = the
//   P7-A 795 GB/s pattern (strided stage_w = 284). The NEXT chunk's units
//   sit in registers while mma runs (DBUF); decode (dq_iq3_r, VERBATIM P4)
//   runs in the post-mma bubble. Decode math and the per-row k-order are
//   IDENTICAL to the shipped M16/M32 kernels -> outputs BIT-IDENTICAL.
// REPACK=0: classic stage_w VERBATIM (pf_gemm.cu) — the M-grid control.
//
// M-grid: grid = (NDIM/NTILE) * (Mrows/MTILE); mb = blockIdx.x / NGRID,
//   nb = blockIdx.x % NGRID; x16/out16/res16 rows offset by mb*MTILE.
//   At MTILE=32 on a 32-row chunk the grid is 1 M-block (drop-in for P6).
//
// LAWS: flat indexing, no gridDim/blockDim reads, hardcoded sizes,
//   sequential/unrolled loops, full-warp masks, single 16B-aligned smem
//   array, per-kernel cubins + warp-token names, acc arrays only ever
//   compile-time indexed. smem = MTILE*XS_LD + (FFN?2:1)*NTILE*WS_LD halfs.
// Build: -DKNAME -DQCLASS -DKDIM -DNDIM -DNTHR -DNTILE -DKCH -DMTILE
//        -DHMMA=1 [-DFFN=1] [-DRES=1] [-DREPACK=1]
#include <cuda_fp16.h>
#if RES && FFN
#error "RES+FFN not supported"
#endif
#if !HMMA
#error "pf_gemm3: HMMA only"
#endif
#if REPACK && QCLASS != 1
#error "REPACK: QCLASS 1 (IQ3_XXS packed) only"
#endif
#if REPACK && KCH != 128
#error "REPACK: KCH 128 only (unit layout is 14B/16B)"
#endif
#define FULL 0xffffffffu

#if QCLASS == 1
#define ROWBYTES ((size_t)98 * (KDIM >> 8))
#elif QCLASS == 2
#define ROWBYTES ((size_t)176 * (KDIM >> 8))
#elif QCLASS == 3
#define ROWBYTES ((size_t)212 * (KDIM >> 8))
#elif QCLASS == 4
#define ROWBYTES ((size_t)144 * (KDIM >> 8))
#elif QCLASS == 5
#define ROWBYTES ((size_t)110 * (KDIM >> 8))
#elif QCLASS == 6
#define ROWBYTES ((size_t)34 * (KDIM >> 5))
#else
#define ROWBYTES ((size_t)(KDIM / 2 + KDIM / 16))
#endif

#define XS_LD (KCH + 8)
#define WS_LD (KCH + 8)
#define NWARP (NTHR / 32)
#define RPT (NTILE / NWARP)
#if RPT != 8
#error "pf_gemm3: NTILE/NWARP must be 8 (row-group layout)"
#endif
#define NCL (KCH / 32)
#define KSTEPS (KCH / 16)
#define NCH (KDIM / KCH)
#define NGRID (NDIM / NTILE)
#define NGRP (NTILE / 8)
#define MRG (MTILE / 16)          // m16 row-groups
#if FFN
#define FGN (MTILE / 16)
#else
#define FGN 0
#endif
#define XTPR ((MTILE * KCH + NTHR * 4 - 1) / (NTHR * 4))
#if MTILE % 16
#error "MTILE multiple of 16"
#endif
#if REPACK && RING4 && (FFN || PING)
#error "RING4: non-FFN base body only (FFN units x2 spill-blocked; PING is its own twin)"
#endif
#if REPACK && RING4 && (NCH % 4)
#error "RING4: NCH % 4 != 0 (unroll-by-4 group cycle)"
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

// ============================ REPACK=1 body =================================
#if REPACK

// P4 dq_iq3_r VERBATIM (register-form IQ3_XXS decode; sidx field = cc&3 —
// in the packed7 unit, q word cc is lane-chunk lc0 + c*4 + cc, lc&3 == cc).
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

// unit u32 view: q0=U.x&FFFF q1=U.x>>16 q2=U.y&FFFF q3=U.y>>16, sw=U.z, d=U.w&FFFF
__device__ __forceinline__ void dq_unit(const float* gridf, const uint4 U, const int cc, __half* w8) {
  const unsigned int qv = (cc == 0) ? (U.x & 0xFFFFu) : (cc == 1) ? (U.x >> 16)
                       : (cc == 2) ? (U.y & 0xFFFFu) : (U.y >> 16);
  dq_iq3_r(gridf, qv, U.z, U.w & 0xFFFFu, cc, w8);
}

#define XPR(R, KC) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR) * 4; \
    if (lin < MTILE*KCH) { \
      const int m = lin / KCH, kk = lin % KCH; \
      x##R[t] = *(const uint2*)(x16 + (size_t)(mb*MTILE + m)*KDIM + (size_t)(KC) + kk); \
    } \
  } \
} while (0)

#define XCM(R) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR) * 4; \
    if (lin < MTILE*KCH) { \
      const int m = lin / KCH, kk = lin % KCH; \
      *(uint2*)(xs + (size_t)m*XS_LD + kk) = x##R[t]; \
    } \
  } \
} while (0)

// decode the lane's unit regs -> this lane's ws rows (post-mma bubble)
#define WCM(P, R, WSB) do { \
  __half* wsP_ = (WSB); \
  _Pragma("unroll") \
  for (int cc = 0; cc < NCL; ++cc) { \
    __half w8[8]; \
    dq_unit(gridf, u##P##R, cc, w8); \
    const int kk = (qc_*NCL + cc)*8; \
  _Pragma("unroll") \
  for (int j = 0; j < 8; ++j) wsP_[(size_t)(warp*8 + r_)*WS_LD + kk + j] = w8[j]; \
  } \
} while (0)

// R2d ring-4: decode the named unit reg u##R (same body, u0..u3 rotation)
#define WCMR(R, WSB) do { \
  __half* wsP_ = (WSB); \
  _Pragma("unroll") \
  for (int cc = 0; cc < NCL; ++cc) { \
    __half w8[8]; \
    dq_unit(gridf, u##R, cc, w8); \
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

#if RING4
// R2d: DBUF ring-depth-4 — 4 named unit-register sets (chunk k in u[k&3]),
// manual unroll-by-4 group cycle. Each phase: [issue unit(s+4) -> u[s&3] (the
// slot's unit(s) was consumed staging chunk s at phase s-1 -> free) + x(s+2)]
// -> mma(s) -> sync -> [decode u[(s+1)&3] -> ws + x-commit(s+1)] -> sync.
// Unit loads get ~3 phases of DRAM-latency cover (depth-2 gave ~1). Decode
// (dq_unit/WCMR), mma fragment map, epilogue and per-row k-order VERBATIM
// from the base body -> outputs BIT-IDENTICAL. Loads/staging beyond NCH are
// predicated off (last group). Non-FFN only (see #error above).
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
    const float* __restrict__ gridf, const __half* __restrict__ x16,
#if RES
    const float* __restrict__ res16, float* __restrict__ out32
#else
    __half* __restrict__ out16
#endif
    )
{
  __shared__ __align__(16) __half sm[MTILE * XS_LD + NTILE * WS_LD];
  float acc[1][MRG][4];
#define ACC(pl, rg, j) acc[pl][rg][j]
  __half* xs = sm;
  __half* ws = sm + MTILE * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x / NGRID;
  const int nb = blockIdx.x - mb * NGRID;
  const int r_ = lane >> 2, qc_ = lane & 3;
  const uint4* w1u4 = (const uint4*)w1;
  _Pragma("unroll")
  for (int rg = 0; rg < MRG; ++rg)
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) ACC(0, rg, j) = 0.f;

  uint4 u0, u1, u2, u3;
  uint2 xE[XTPR], xO[XTPR];
  const size_t goff = ((size_t)(nb * NGRP + warp) * NCH + 0) * 32 + lane;

  // prologue: stage chunk 0 (u0/xE); issue units 1..3 + x(1)
  u0 = w1u4[goff];
  XPR(E, 0);
  WCMR(0, ws);
  XCM(E);
  __syncthreads();
  u1 = w1u4[goff + (size_t)1 * 32];
  u2 = w1u4[goff + (size_t)2 * 32];
  u3 = w1u4[goff + (size_t)3 * 32];
  XPR(O, KCH);
  _Pragma("unroll 1")
  for (int g4 = 0; g4 < NCH / 4; ++g4) {
    const int c0 = 4 * g4;
    // phase A (chunk c0): stage c0+1 from u1/xO
    if (c0 + 4 < NCH) u0 = w1u4[goff + (size_t)(c0 + 4) * 32];
    if (c0 + 2 < NCH) XPR(E, (size_t)(c0 + 2) * KCH);
    MMAR();
    __syncthreads();
    WCMR(1, ws); XCM(O);
    __syncthreads();
    // phase B (chunk c0+1): stage c0+2 from u2/xE
    if (c0 + 5 < NCH) u1 = w1u4[goff + (size_t)(c0 + 5) * 32];
    if (c0 + 3 < NCH) XPR(O, (size_t)(c0 + 3) * KCH);
    MMAR();
    __syncthreads();
    WCMR(2, ws); XCM(E);
    __syncthreads();
    // phase C (chunk c0+2): stage c0+3 from u3/xO
    if (c0 + 6 < NCH) u2 = w1u4[goff + (size_t)(c0 + 6) * 32];
    if (c0 + 4 < NCH) XPR(E, (size_t)(c0 + 4) * KCH);
    MMAR();
    __syncthreads();
    WCMR(3, ws); XCM(O);
    __syncthreads();
    // phase D (chunk c0+3): stage c0+4 from u0/xE (skipped at the last group)
    if (c0 + 7 < NCH) u3 = w1u4[goff + (size_t)(c0 + 7) * 32];
    if (c0 + 5 < NCH) XPR(O, (size_t)(c0 + 5) * KCH);
    MMAR();
    __syncthreads();
    if (c0 + 4 < NCH) { WCMR(0, ws); XCM(E); }
    __syncthreads();
  }

  // epilogue (per row-group, the classic c-frag map)
  {
    const int g = lane >> 2, tp = (lane & 3) * 2;
    const int n = nb * NTILE + warp*8 + tp;
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg) {
      const int m0 = mb * MTILE + rg*16 + g;
#if RES
      out32[(size_t)m0*NDIM + n]           = res16[(size_t)m0*NDIM + n] + ACC(0, rg, 0);
      out32[(size_t)m0*NDIM + n + 1]       = res16[(size_t)m0*NDIM + n + 1] + ACC(0, rg, 1);
      out32[(size_t)(m0+8)*NDIM + n]       = res16[(size_t)(m0+8)*NDIM + n] + ACC(0, rg, 2);
      out32[(size_t)(m0+8)*NDIM + n + 1]   = res16[(size_t)(m0+8)*NDIM + n + 1] + ACC(0, rg, 3);
#else
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     = __halves2half2((__half)ACC(0, rg, 0), (__half)ACC(0, rg, 1));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) = __halves2half2((__half)ACC(0, rg, 2), (__half)ACC(0, rg, 3));
#endif
    }
  }
#undef ACC
}
#elif !PING  // P16: the base r7 kernel is replaced by the PING twin when PING is set
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
#if FFN
    const unsigned char* __restrict__ w2,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
#if RES
    const float* __restrict__ res16, float* __restrict__ out32
#else
    __half* __restrict__ out16
#endif
    )
{
#if FFN
  __shared__ __align__(16) __half sm[MTILE * XS_LD + 2 * NTILE * WS_LD];
  float acc[2][MRG][4];
#else
  __shared__ __align__(16) __half sm[MTILE * XS_LD + NTILE * WS_LD];
  float acc[1][MRG][4];
#endif
#define ACC(pl, rg, j) acc[pl][rg][j]
  __half* xs = sm;
  __half* ws = sm + MTILE * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x / NGRID;
  const int nb = blockIdx.x - mb * NGRID;
  const int r_ = lane >> 2, qc_ = lane & 3;
  const uint4* w1u4 = (const uint4*)w1;
#if FFN
  const uint4* w2u4 = (const uint4*)w2;
#endif
  _Pragma("unroll")
  for (int pl = 0; pl < (FFN ? 2 : 1); ++pl)
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg)
      _Pragma("unroll")
      for (int j = 0; j < 4; ++j) ACC(pl, rg, j) = 0.f;

  uint4 uGE, uGO;
#if FFN
  uint4 uUE, uUO;
#endif
  uint2 xE[XTPR], xO[XTPR];
  const size_t goff = ((size_t)(nb * NGRP + warp) * NCH + 0) * 32 + lane;

  // prologue: unit(0) + x(0) -> regs -> decode -> ws/xs
  uGE = w1u4[goff];
#if FFN
  uUE = w2u4[goff];
#endif
  XPR(E, 0);
  WCM(G, E, ws);
#if FFN
  WCM(U, E, ws + NTILE*WS_LD);
#endif
  XCM(E);
  __syncthreads();
  _Pragma("unroll 1")
  for (int s = 0; s + 1 < NCH; s += 2) {
    uGO = w1u4[goff + (size_t)(s + 1) * 32];
#if FFN
    uUO = w2u4[goff + (size_t)(s + 1) * 32];
#endif
    XPR(O, (s + 1) * KCH);
    MMAR();
    __syncthreads();
    WCM(G, O, ws);
#if FFN
    WCM(U, O, ws + NTILE*WS_LD);
#endif
    XCM(O);
    __syncthreads();
    if (s + 2 >= NCH) break;
    uGE = w1u4[goff + (size_t)(s + 2) * 32];
#if FFN
    uUE = w2u4[goff + (size_t)(s + 2) * 32];
#endif
    XPR(E, (s + 2) * KCH);
    MMAR();
    __syncthreads();
    WCM(G, E, ws);
#if FFN
    WCM(U, E, ws + NTILE*WS_LD);
#endif
    XCM(E);
    __syncthreads();
  }
  MMAR();

  // epilogue (per row-group, the classic c-frag map)
  {
    const int g = lane >> 2, tp = (lane & 3) * 2;
    const int n = nb * NTILE + warp*8 + tp;
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg) {
      const int m0 = mb * MTILE + rg*16 + g;
#if RES
      out32[(size_t)m0*NDIM + n]           = res16[(size_t)m0*NDIM + n] + ACC(0, rg, 0);
      out32[(size_t)m0*NDIM + n + 1]       = res16[(size_t)m0*NDIM + n + 1] + ACC(0, rg, 1);
      out32[(size_t)(m0+8)*NDIM + n]       = res16[(size_t)(m0+8)*NDIM + n] + ACC(0, rg, 2);
      out32[(size_t)(m0+8)*NDIM + n + 1]   = res16[(size_t)(m0+8)*NDIM + n + 1] + ACC(0, rg, 3);
#elif FFN
      const __half hg0 = (__half)ACC(0, rg, 0), hg1 = (__half)ACC(0, rg, 1);
      const __half hg2 = (__half)ACC(0, rg, 2), hg3 = (__half)ACC(0, rg, 3);
      const __half hn0 = hsilu_h(hg0), hn1 = hsilu_h(hg1), hn2 = hsilu_h(hg2), hn3 = hsilu_h(hg3);
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     = __halves2half2(__hmul(hn0, (__half)ACC(1, rg, 0)), __hmul(hn1, (__half)ACC(1, rg, 1)));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) = __halves2half2(__hmul(hn2, (__half)ACC(1, rg, 2)), __hmul(hn3, (__half)ACC(1, rg, 3)));
#else
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     = __halves2half2((__half)ACC(0, rg, 0), (__half)ACC(0, rg, 1));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) = __halves2half2((__half)ACC(0, rg, 2), (__half)ACC(0, rg, 3));
#endif
    }
  }
#undef ACC
}
#endif  // !PING
#endif  // REPACK

// ============================ REPACK=0 (classic control) ====================
#if !REPACK
// decode one 8k lane-chunk lc of 256k-block b of W row r -> 8 halfs
// (VERBATIM from pf_gemm.cu w1c-proven references)
__device__ __forceinline__ void dq_chunk_iq3(const unsigned char* rowp, const float* gridf,
                                             int b, int lc, __half* w8) {
  const unsigned short* qsp = (const unsigned short*)(rowp);
  const unsigned int* scp = (const unsigned int*)(rowp + 64*(KDIM >> 8));
  const unsigned short* dpp = (const unsigned short*)(rowp + 96*(KDIM >> 8));
  const float d = __half2float(__ushort_as_half(dpp[b]));
  const unsigned int sw = scp[8*b + (lc>>2)];
  const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
  const unsigned int sidx = (sw >> (7u * (unsigned int)(lc & 3))) & 0x7Fu;
  const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
  const unsigned int q = qsp[32*b + lc];
  const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2)));
  const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2)));
  const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
  const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
  const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
  const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
  w8[0] = __float2half(db*g0.x*sg0); w8[1] = __float2half(db*g0.y*sg1);
  w8[2] = __float2half(db*g0.z*sg2); w8[3] = __float2half(db*g0.w*sg3);
  w8[4] = __float2half(db*g1.x*sg4); w8[5] = __float2half(db*g1.y*sg5);
  w8[6] = __float2half(db*g1.z*sg6); w8[7] = __float2half(db*g1.w*sg7);
}

// P8: IQ3_S classic decode (VERBATIM pf_gemm.cu dq_chunk_iq3s) — the attn
// o-proj M-grid fold build (QCLASS=5).
__device__ __forceinline__ void dq_chunk_iq3s(const unsigned char* rowp, const float* grid512,
                                              int b, int lc, __half* w8) {
  const unsigned char* blk = rowp + (size_t)b*110;
  const float d = __half2float(*((const __half*)blk));
  const int g0 = lc*2, g1 = lc*2 + 1;
  const int sraw = lc >> 2;
  const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
  const unsigned char* sgnb = blk + 74 + lc;
  const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lc);
  const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0>>3)] >> (g0&7)) & 1u) << 8));
  const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1>>3)] >> (g1&7)) & 1u) << 8));
  const float4 gv0 = *((const float4*)(grid512 + (qb0 << 2)));
  const float4 gv1 = *((const float4*)(grid512 + (qb1 << 2)));
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) {
    const float gv = (j < 4) ? ((&gv0.x)[j]) : ((&gv1.x)[j-4]);
    const float sgn = ((*sgnb >> j) & 1) ? -1.f : 1.f;
    w8[j] = __float2half(d * sc * gv * sgn);
  }
}

__device__ __forceinline__ void stage_w(__half* ws, const unsigned char* w,
                                        const float* gridf, int kc, int lane, int warp, int n0) {
  const int r = lane >> 2;
  const int c = lane & 3;
  const unsigned char* rowp0 = w + (size_t)(n0 + warp*RPT) * ROWBYTES;
  const int lc0 = (kc & 255) >> 3;
  const unsigned char* rowp = rowp0 + (size_t)r * ROWBYTES;
  _Pragma("unroll")
  for (int cc = 0; cc < NCL; ++cc) {
    const int lc = lc0 + c*NCL + cc;
    __half w8[8];
#if QCLASS == 5
    dq_chunk_iq3s(rowp, gridf, (kc >> 8), lc, w8);
#else
    dq_chunk_iq3(rowp, gridf, (kc >> 8), lc, w8);
#endif
    const int kk = (c*NCL + cc)*8;
    _Pragma("unroll")
    for (int j = 0; j < 8; ++j) ws[(size_t)(warp*RPT + r)*WS_LD + kk + j] = w8[j];
  }
}

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
#if FFN
    const unsigned char* __restrict__ w2,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
#if RES
    const float* __restrict__ res16, float* __restrict__ out32
#else
    __half* __restrict__ out16
#endif
    )
{
#if FFN
  __shared__ __align__(16) __half sm[MTILE * XS_LD + 2 * NTILE * WS_LD];
  float acc[2][MRG][4];
#else
  __shared__ __align__(16) __half sm[MTILE * XS_LD + NTILE * WS_LD];
  float acc[1][MRG][4];
#endif
#define ACC(pl, rg, j) acc[pl][rg][j]
  __half* xs = sm;
  __half* ws = sm + MTILE * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x / NGRID;
  const int nb = blockIdx.x - mb * NGRID;
  const int n0 = nb * NTILE;
  _Pragma("unroll")
  for (int pl = 0; pl < (FFN ? 2 : 1); ++pl)
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg)
      _Pragma("unroll")
      for (int j = 0; j < 4; ++j) ACC(pl, rg, j) = 0.f;

  _Pragma("unroll 2")
  for (int kc = 0; kc < KDIM; kc += KCH) {
    _Pragma("unroll")
    for (int i = 0; i < XTPR; ++i) {
      const int lin = (tid + i*NTHR) * 4;
      if (lin < MTILE*KCH) {
        const int m = lin / KCH, kk = lin % KCH;
        *(uint2*)(xs + (size_t)m*XS_LD + kk) = *(const uint2*)(x16 + (size_t)(mb*MTILE + m)*KDIM + kc + kk);
      }
    }
    stage_w(ws, w1, gridf, kc, lane, warp, n0);
#if FFN
    stage_w(ws + NTILE*WS_LD, w2, gridf, kc, lane, warp, n0);
#endif
    __syncthreads();
    const int g = lane >> 2, tp = (lane & 3) * 2;
    _Pragma("unroll")
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s*16;
      _Pragma("unroll")
      for (int rg = 0; rg < MRG; ++rg) {
        const unsigned a0 = *(const unsigned*)(xs + (size_t)(rg*16 + g)*XS_LD + kb + tp);
        const unsigned a1 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp);
        const unsigned a2 = *(const unsigned*)(xs + (size_t)(rg*16 + g)*XS_LD + kb + tp + 8);
        const unsigned a3 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp + 8);
        const __half* wr = ws + (size_t)(warp*8 + g)*WS_LD + kb + tp;
        const unsigned b0 = *(const unsigned*)(wr);
        const unsigned b1 = *(const unsigned*)(wr + 8);
        hmma16816(ACC(0, rg, 0), ACC(0, rg, 1), ACC(0, rg, 2), ACC(0, rg, 3), a0, a1, a2, a3, b0, b1);
#if FFN
        const __half* wu = ws + NTILE*WS_LD + (size_t)(warp*8 + g)*WS_LD + kb + tp;
        const unsigned c0 = *(const unsigned*)(wu);
        const unsigned c1 = *(const unsigned*)(wu + 8);
        hmma16816(ACC(1, rg, 0), ACC(1, rg, 1), ACC(1, rg, 2), ACC(1, rg, 3), a0, a1, a2, a3, c0, c1);
#endif
      }
    }
    __syncthreads();
  }
  {
    const int g = lane >> 2, tp = (lane & 3) * 2;
    const int n = n0 + warp*8 + tp;
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg) {
      const int m0 = mb * MTILE + rg*16 + g;
#if RES
      out32[(size_t)m0*NDIM + n]           = res16[(size_t)m0*NDIM + n] + ACC(0, rg, 0);
      out32[(size_t)m0*NDIM + n + 1]       = res16[(size_t)m0*NDIM + n + 1] + ACC(0, rg, 1);
      out32[(size_t)(m0+8)*NDIM + n]       = res16[(size_t)(m0+8)*NDIM + n] + ACC(0, rg, 2);
      out32[(size_t)(m0+8)*NDIM + n + 1]   = res16[(size_t)(m0+8)*NDIM + n + 1] + ACC(0, rg, 3);
#elif FFN
      const __half hg0 = (__half)ACC(0, rg, 0), hg1 = (__half)ACC(0, rg, 1);
      const __half hg2 = (__half)ACC(0, rg, 2), hg3 = (__half)ACC(0, rg, 3);
      const __half hn0 = hsilu_h(hg0), hn1 = hsilu_h(hg1), hn2 = hsilu_h(hg2), hn3 = hsilu_h(hg3);
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     = __halves2half2(__hmul(hn0, (__half)ACC(1, rg, 0)), __hmul(hn1, (__half)ACC(1, rg, 1)));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) = __halves2half2(__hmul(hn2, (__half)ACC(1, rg, 2)), __hmul(hn3, (__half)ACC(1, rg, 3)));
#else
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     = __halves2half2((__half)ACC(0, rg, 0), (__half)ACC(0, rg, 1));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) = __halves2half2((__half)ACC(0, rg, 2), (__half)ACC(0, rg, 3));
#endif
    }
  }
#undef ACC
}
#endif  // !REPACK

// ================= P16: PING — the smem ws ping-pong (serial-chain lever) ==
// The P14 wall: the per-chunk chain stage/decode -> __syncthreads -> mma ->
// __syncthreads gates ~3.1us/chunk (decode on the inter-barrier critical
// path). PING moves decode(k+1) into mma(k)'s shadow by double-buffering
// the staged W tile(s):
//   PING=1: ws x2 (FFN: both G/U planes x2), xs single-buffered -> decode
//     still off the mma-to-mma path; the xs re-stage keeps its own barrier
//     (2 barriers/chunk, but only the tiny XCM sits between them).
//     smem = MTILE*XS_LD + 2*(FFN?2:1)*NTILE*WS_LD halfs.
//   PING=2: ws AND xs double-buffered -> ONE swap barrier per chunk (the
//     full ping-pong; xs stores overlap other warps' mma reads).
//     smem = 2*MTILE*XS_LD + 2*(FFN?2:1)*NTILE*WS_LD halfs.
// Decode (dq_unit/WCM), mma fragment map (MMAR), epilogue and the per-row
// k-order are VERBATIM the r7 body -> outputs BIT-IDENTICAL (gated).
#if PING
#if !REPACK
#error "PING: REPACK=1 only (r7 unit ring)"
#endif
#if PING != 1 && PING != 2
#error "PING: 1 or 2"
#endif
// x-store with an explicit base (the r7 XCM writes the fixed xs)
#define XCM2(R, XB) do { \
  __half* xsB_ = (XB); \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR) * 4; \
    if (lin < MTILE*KCH) { \
      const int m = lin / KCH, kk = lin % KCH; \
      *(uint2*)(xsB_ + (size_t)m*XS_LD + kk) = x##R[t]; \
    } \
  } \
} while (0)
// mma over an explicit ws plane-pair base + xs base (fragment map VERBATIM MMAR)
#define MMARP(W, XB) do { \
  const __half* wsP_ = (W); \
  const __half* xsP_ = (XB); \
  const int g = lane >> 2, tp = (lane & 3) * 2; \
  _Pragma("unroll") \
  for (int s = 0; s < KSTEPS; ++s) { \
    const int kb = s*16; \
    _Pragma("unroll") \
    for (int rg = 0; rg < MRG; ++rg) { \
      const unsigned a0 = *(const unsigned*)(xsP_ + (size_t)(rg*16 + g)*XS_LD + kb + tp); \
      const unsigned a1 = *(const unsigned*)(xsP_ + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp); \
      const unsigned a2 = *(const unsigned*)(xsP_ + (size_t)(rg*16 + g)*XS_LD + kb + tp + 8); \
      const unsigned a3 = *(const unsigned*)(xsP_ + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp + 8); \
      const __half* wr = wsP_ + (size_t)(warp*8 + g)*WS_LD + kb + tp; \
      const unsigned b0 = *(const unsigned*)(wr); \
      const unsigned b1 = *(const unsigned*)(wr + 8); \
      hmma16816(ACC(0, rg, 0), ACC(0, rg, 1), ACC(0, rg, 2), ACC(0, rg, 3), a0, a1, a2, a3, b0, b1); \
    } \
    _Pragma("unroll") \
    for (int rg = 0; rg < FGN; ++rg) { \
      const unsigned a0 = *(const unsigned*)(xsP_ + (size_t)(rg*16 + g)*XS_LD + kb + tp); \
      const unsigned a1 = *(const unsigned*)(xsP_ + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp); \
      const unsigned a2 = *(const unsigned*)(xsP_ + (size_t)(rg*16 + g)*XS_LD + kb + tp + 8); \
      const unsigned a3 = *(const unsigned*)(xsP_ + (size_t)(rg*16 + g + 8)*XS_LD + kb + tp + 8); \
      const __half* wr = wsP_ + NTILE*WS_LD + (size_t)(warp*8 + g)*WS_LD + kb + tp; \
      const unsigned b0 = *(const unsigned*)(wr); \
      const unsigned b1 = *(const unsigned*)(wr + 8); \
      hmma16816(ACC(1, rg, 0), ACC(1, rg, 1), ACC(1, rg, 2), ACC(1, rg, 3), a0, a1, a2, a3, b0, b1); \
    } \
  } \
} while (0)
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
#if FFN
    const unsigned char* __restrict__ w2,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
#if RES
    const float* __restrict__ res16, float* __restrict__ out32
#else
    __half* __restrict__ out16
#endif
    )
{
#if PING == 1
  #if FFN
  __shared__ __align__(16) __half sm[MTILE * XS_LD + 4 * NTILE * WS_LD];
  float acc[2][MRG][4];
  #else
  __shared__ __align__(16) __half sm[MTILE * XS_LD + 2 * NTILE * WS_LD];
  float acc[1][MRG][4];
  #endif
  __half* xsA = sm;
  __half* xsB = sm;                       // same buffer (xs single)
  __half* ws0 = sm + MTILE * XS_LD;
#else
  #if FFN
  __shared__ __align__(16) __half sm[2 * MTILE * XS_LD + 4 * NTILE * WS_LD];
  float acc[2][MRG][4];
  #else
  __shared__ __align__(16) __half sm[2 * MTILE * XS_LD + 2 * NTILE * WS_LD];
  float acc[1][MRG][4];
  #endif
  __half* xsA = sm;
  __half* xsB = sm + MTILE * XS_LD;
  __half* ws0 = sm + 2 * MTILE * XS_LD;
#endif
  __half* ws1 = ws0 + (FFN ? 2 : 1) * NTILE * WS_LD;
#define ACC(pl, rg, j) acc[pl][rg][j]
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x / NGRID;
  const int nb = blockIdx.x - mb * NGRID;
  const int r_ = lane >> 2, qc_ = lane & 3;
  const uint4* w1u4 = (const uint4*)w1;
#if FFN
  const uint4* w2u4 = (const uint4*)w2;
#endif
  _Pragma("unroll")
  for (int pl = 0; pl < (FFN ? 2 : 1); ++pl)
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg)
      _Pragma("unroll")
      for (int j = 0; j < 4; ++j) ACC(pl, rg, j) = 0.f;

  uint4 uGE, uGO;
#if FFN
  uint4 uUE, uUO;
#endif
  uint2 xE[XTPR], xO[XTPR];
  const size_t goff = ((size_t)(nb * NGRP + warp) * NCH + 0) * 32 + lane;

  // prologue: chunk 0 units+x -> regs -> decode -> ws0/xsA; chunk 1 -> O regs
  uGE = w1u4[goff];
#if FFN
  uUE = w2u4[goff];
#endif
  XPR(E, 0);
  WCM(G, E, ws0);
#if FFN
  WCM(U, E, ws0 + NTILE*WS_LD);
#endif
  XCM2(E, xsA);
  __syncthreads();
  uGO = w1u4[goff + (size_t)32];
#if FFN
  uUO = w2u4[goff + (size_t)32];
#endif
  XPR(O, KCH);

  _Pragma("unroll 1")
  for (int s = 0; s + 1 < NCH; s += 2) {
    // ---- half A: mma(s) on ws0/xsA; decode(s+1, O regs) -> ws1; prefetch(s+2) -> E
    MMARP(ws0, xsA);
    WCM(G, O, ws1);
#if FFN
    WCM(U, O, ws1 + NTILE*WS_LD);
#endif
    if (s + 2 < NCH) {
      uGE = w1u4[goff + (size_t)(s + 2) * 32];
#if FFN
      uUE = w2u4[goff + (size_t)(s + 2) * 32];
#endif
      XPR(E, (s + 2) * KCH);
    }
#if PING == 2
    XCM2(O, xsB);
#endif
    __syncthreads();
#if PING == 1
    XCM2(O, xsA);
    __syncthreads();
#endif
    if (s + 2 >= NCH) break;
    // ---- half B: mma(s+1) on ws1/xsB; decode(s+2, E regs) -> ws0; prefetch(s+3) -> O
    MMARP(ws1, xsB);
    WCM(G, E, ws0);
#if FFN
    WCM(U, E, ws0 + NTILE*WS_LD);
#endif
    if (s + 3 < NCH) {
      uGO = w1u4[goff + (size_t)(s + 3) * 32];
#if FFN
      uUO = w2u4[goff + (size_t)(s + 3) * 32];
#endif
      XPR(O, (s + 3) * KCH);
    }
#if PING == 2
    XCM2(E, xsA);
#endif
    __syncthreads();
#if PING == 1
    XCM2(E, xsB);
    __syncthreads();
#endif
  }

  // epilogue (VERBATIM the r7 classic map)
  {
    const int g = lane >> 2, tp = (lane & 3) * 2;
    const int n = nb * NTILE + warp*8 + tp;
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg) {
      const int m0 = mb * MTILE + rg*16 + g;
#if RES
      out32[(size_t)m0*NDIM + n]           = res16[(size_t)m0*NDIM + n] + ACC(0, rg, 0);
      out32[(size_t)m0*NDIM + n + 1]       = res16[(size_t)m0*NDIM + n + 1] + ACC(0, rg, 1);
      out32[(size_t)(m0+8)*NDIM + n]       = res16[(size_t)(m0+8)*NDIM + n] + ACC(0, rg, 2);
      out32[(size_t)(m0+8)*NDIM + n + 1]   = res16[(size_t)(m0+8)*NDIM + n + 1] + ACC(0, rg, 3);
#elif FFN
      const __half hg0 = (__half)ACC(0, rg, 0), hg1 = (__half)ACC(0, rg, 1);
      const __half hg2 = (__half)ACC(0, rg, 2), hg3 = (__half)ACC(0, rg, 3);
      const __half hn0 = hsilu_h(hg0), hn1 = hsilu_h(hg1), hn2 = hsilu_h(hg2), hn3 = hsilu_h(hg3);
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     = __halves2half2(__hmul(hn0, (__half)ACC(1, rg, 0)), __hmul(hn1, (__half)ACC(1, rg, 1)));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) = __halves2half2(__hmul(hn2, (__half)ACC(1, rg, 2)), __hmul(hn3, (__half)ACC(1, rg, 3)));
#else
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     = __halves2half2((__half)ACC(0, rg, 0), (__half)ACC(0, rg, 1));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) = __halves2half2((__half)ACC(0, rg, 2), (__half)ACC(0, rg, 3));
#endif
    }
  }
#undef ACC
#undef XCM2
#undef MMARP
}
#endif  // PING
