// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R7b pf_gemm3w v2: WARP-SPEC GEMM — producers stage the X stream (uint4,
// ring-2 regs, loads issued BEFORE the stage meet -> DRAM latency hides under
// the consumers' mma); consumers run MMAR VERBATIM per K=64 stage in k order
// + their OWN W-unit register ring + post-mma decode (the base DBUF pattern).
//
// BIT-IDENTITY CONSTRUCTION:
//   xs[m][kl] == base XPR value x16[mb*MTILE+m][s*64+kl] (same bytes, uint4
//   legal: KDIM*2 % 16 == 0 for all in-plan classes).
//   ws: the base writes k-in-chunk (qc_*4+cc)*8 -> stage h covers k-in-chunk
//   [64h,64h+64) == lanes qc_ in {2h,2h+1} with ALL cc: lane decodes its unit
//   (dq_unit VERBATIM) into stage-local kk ((qc_&1)*4+cc)*8. Values AND
//   value->position map identical; consumers run the same HMMA fragment
//   sequence in the same k order (stage halves in sequence) -> BIT-IDENTICAL.
// PIPELINE (NS=2 smem bufs, ONE meet barrier per stage, bar id 1):
//   producer: [load X(s+1)->regs; nbar; store X(s+1)->buf(s+1)&1]
//   consumer: [nbar; MMAR(s); decode ws(s+1)->buf(s+1)&1]
//   buf(s-1)&1 reuse is safe: consumers arrive at meet(s) only after finishing
//   MMAR(s-1). Named barriers need QMD barrier_count>=2 on this dext
//   (NV_QMD_BARRIERS=16 NV_QMD_BARRIERS_NAMES=pfg3w; mbarrier is DEAD here).
// LAWS: flat indexing, hardcoded sizes, single smem array, full-warp masks,
//   compile-time-indexed acc, one kernel per cubin, warp-token names, 0 spill.
// Build: -DKNAME -DQCLASS=1 -DREPACK=1 -DKDIM -DNDIM -DNTHR -DNTILE -DKCHW=64
//        -DMTILE -DHMMA=1 -DNCONS -DNPROD [-DFFN=1] [-DRES=1]
#include <cuda_fp16.h>
#if !HMMA
#error "pf_gemm3w: HMMA only"
#endif
#if QCLASS != 1
#error "pf_gemm3w: REPACK IQ3_XXS packed only"
#endif
#if KCHW != 64
#error "pf_gemm3w: KCHW 64"
#endif
#if NTILE != NCONS * 8
#error "pf_gemm3w: NTILE == NCONS*8"
#endif
#define FULL 0xffffffffu

#define XS_LD (KCHW + 8)
#define WS_LD (KCHW + 8)
#define NWARP (NTHR / 32)
#if NCONS + NPROD != NWARP
#error "pf_gemm3w: NCONS + NPROD == NWARP"
#endif
#define NCLW (KCHW / 32)      // 2
#define KSTEPS (KCHW / 16)    // 4
#define NCH (KDIM / 128)      // base unit chunks
#define NST (KDIM / KCHW)     // stages
#define NGRID (NDIM / NTILE)
#define MRG (MTILE / 16)
#if FFN
#define FGN (MTILE / 16)
#else
#define FGN 0
#endif
#define XPT ((MTILE * KCHW + NPROD * 32 * 8 - 1) / (NPROD * 32 * 8))
#define STAGE_HALF (MTILE * XS_LD + (FFN ? 2 : 1) * NTILE * WS_LD)

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
__device__ __forceinline__ void nbar(int id, int cnt) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(cnt));
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
  __shared__ __align__(16) __half sm[2 * STAGE_HALF];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x / NGRID;
  const int nb = blockIdx.x - mb * NGRID;
  const int cons = warp < NCONS;

#if FFN
  float acc[2][MRG][4];
#else
  float acc[1][MRG][4];
#endif
#define ACC(pl, rg, j) acc[pl][rg][j]

  if (!cons) {
    // ---------------- producers: the X stream only ----------------
    // static unroll-by-2 ring (RING4 convention: a dynamically-indexed unit
    // array demotes to local memory on nvcc; named sets stay in registers)
    const int pt = tid - NCONS * 32;
    uint4 xR0[XPT], xR1[XPT];
#define XLD(A, kc) do { \
      _Pragma("unroll") \
      for (int t = 0; t < XPT; ++t) { \
        const int lin = (pt + t * (NPROD * 32)) * 8; \
        if (lin < MTILE * KCHW) { \
          const int m = lin / KCHW, kk = lin % KCHW; \
          A[t] = *(const uint4*)(x16 + (size_t)(mb * MTILE + m) * KDIM + (size_t)(kc) + kk); \
        } \
      } \
    } while (0)
#define XST(A, buf) do { \
      __half* xsB = sm + (buf) * STAGE_HALF; \
      _Pragma("unroll") \
      for (int t = 0; t < XPT; ++t) { \
        const int lin = (pt + t * (NPROD * 32)) * 8; \
        if (lin < MTILE * KCHW) { \
          const int m = lin / KCHW, kk = lin % KCHW; \
          *(uint4*)(xsB + (size_t)m * XS_LD + kk) = A[t]; \
        } \
      } \
    } while (0)
    XLD(xR0, 0);
    XST(xR0, 0);
    _Pragma("unroll 1")
    for (int s = 0; s < NST; s += 2) {
      if (s + 1 < NST) XLD(xR1, (size_t)(s + 1) * KCHW);
      nbar(1, NTHR);
      if (s + 1 < NST) XST(xR1, (s + 1) & 1);
      if (s + 2 < NST) XLD(xR0, (size_t)(s + 2) * KCHW);
      nbar(1, NTHR);
      if (s + 2 < NST) XST(xR0, (s + 2) & 1);
    }
    return;
#undef XLD
#undef XST
  }

  // ---------------- consumers: mma + own unit ring + post-mma decode ----------------
  {
    const int r_ = lane >> 2, qc_ = lane & 3;
    const uint4* w1u4 = (const uint4*)w1;
#if FFN
    const uint4* w2u4 = (const uint4*)w2;
#endif
    const size_t goff = ((size_t)(nb * NCONS + warp) * NCH) * 32 + lane;
    uint4 u1, u1N;
#if FFN
    uint4 u2, u2N;
#endif
    _Pragma("unroll")
    for (int pl = 0; pl < (FFN ? 2 : 1); ++pl)
      _Pragma("unroll")
      for (int rg = 0; rg < MRG; ++rg)
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) ACC(pl, rg, j) = 0.f;

    // v3 chunk-boundary decode: at odd stage s the lane decodes its unit of
    // chunk (s+2)>>1 ONCE (4 cc, all lanes, no divergence): lanes qc_ in
    // {0,1} write the h=0 slice into buf((s+2)&1); lanes qc_ in {2,3} write
    // the h=1 slice into buf((s+3)&1) (= buf(s+1)&1, free after MMAR(s+1);
    // each warp owns its rows; the overwrite happens post-MMAR(s+1)).
#define CDECP(Uv, buf, poff, qh) do { \
      if ((qc_ >> 1) == (qh)) { \
        __half* wsB = sm + (buf) * STAGE_HALF + MTILE * XS_LD + (poff); \
        _Pragma("unroll") \
        for (int cc = 0; cc < 4; ++cc) { \
          __half w8[8]; \
          dq_unit(gridf, Uv, cc, w8); \
          const int kk = ((qc_ & 1) * 4 + cc) * 8; \
          _Pragma("unroll") \
          for (int j = 0; j < 8; ++j) wsB[(size_t)(warp * 8 + r_) * WS_LD + kk + j] = w8[j]; \
        } \
      } \
    } while (0)
#if FFN
#define CDECN(U1v, U2v, bufA, bufB) do { CDECP(U1v, bufA, 0, 0); CDECP(U1v, bufB, 0, 1); CDECP(U2v, bufA, NTILE * WS_LD, 0); CDECP(U2v, bufB, NTILE * WS_LD, 1); } while (0)
#else
#define CDECN(U1v, U2v, bufA, bufB) do { CDECP(U1v, bufA, 0, 0); CDECP(U1v, bufB, 0, 1); } while (0)
#endif

#define MMARS(buf) do { \
      const __half* xs = sm + (buf) * STAGE_HALF; \
      const __half* ws = sm + (buf) * STAGE_HALF + MTILE * XS_LD; \
      const int g = lane >> 2, tp = (lane & 3) * 2; \
      _Pragma("unroll") \
      for (int st = 0; st < KSTEPS; ++st) { \
        const int kb = st * 16; \
        _Pragma("unroll") \
        for (int rg = 0; rg < MRG; ++rg) { \
          const unsigned a0 = *(const unsigned*)(xs + (size_t)(rg * 16 + g) * XS_LD + kb + tp); \
          const unsigned a1 = *(const unsigned*)(xs + (size_t)(rg * 16 + g + 8) * XS_LD + kb + tp); \
          const unsigned a2 = *(const unsigned*)(xs + (size_t)(rg * 16 + g) * XS_LD + kb + tp + 8); \
          const unsigned a3 = *(const unsigned*)(xs + (size_t)(rg * 16 + g + 8) * XS_LD + kb + tp + 8); \
          const __half* wr = ws + (size_t)(warp * 8 + g) * WS_LD + kb + tp; \
          const unsigned b0 = *(const unsigned*)(wr); \
          const unsigned b1 = *(const unsigned*)(wr + 8); \
          hmma16816(ACC(0, rg, 0), ACC(0, rg, 1), ACC(0, rg, 2), ACC(0, rg, 3), a0, a1, a2, a3, b0, b1); \
        } \
        _Pragma("unroll") \
        for (int rg = 0; rg < FGN; ++rg) { \
          const unsigned a0 = *(const unsigned*)(xs + (size_t)(rg * 16 + g) * XS_LD + kb + tp); \
          const unsigned a1 = *(const unsigned*)(xs + (size_t)(rg * 16 + g + 8) * XS_LD + kb + tp); \
          const unsigned a2 = *(const unsigned*)(xs + (size_t)(rg * 16 + g) * XS_LD + kb + tp + 8); \
          const unsigned a3 = *(const unsigned*)(xs + (size_t)(rg * 16 + g + 8) * XS_LD + kb + tp + 8); \
          const __half* wr = ws + NTILE * WS_LD + (size_t)(warp * 8 + g) * WS_LD + kb + tp; \
          const unsigned b0 = *(const unsigned*)(wr); \
          const unsigned b1 = *(const unsigned*)(wr + 8); \
          hmma16816(ACC(1, rg, 0), ACC(1, rg, 1), ACC(1, rg, 2), ACC(1, rg, 3), a0, a1, a2, a3, b0, b1); \
        } \
      } \
    } while (0)

    // prologue: chunk 0 -> stage 0 (h=0 -> buf0) and stage 1 (h=1 -> buf1)
    u1 = w1u4[goff];
#if FFN
    u2 = w2u4[goff];
#endif
    CDECP(u1, 0, 0, 0);
#if FFN
    CDECP(u2, 0, NTILE * WS_LD, 0);
#endif
    if (1 < NST) {
      CDECP(u1, 1, 0, 1);
#if FFN
      CDECP(u2, 1, NTILE * WS_LD, 1);
#endif
    }
    _Pragma("unroll 1")
    for (int s = 0; s < NST; ++s) {
      if (s + 2 < NST && (((s + 2) & 1) == 0)) {
        u1N = w1u4[goff + (size_t)((s + 2) >> 1) * 32];
#if FFN
        u2N = w2u4[goff + (size_t)((s + 2) >> 1) * 32];
#endif
      }
      nbar(1, NTHR);
      MMARS(s & 1);
      if (s + 1 < NST && ((s & 1) == 1)) {
        u1 = u1N;
#if FFN
        u2 = u2N;
#endif
        CDECN(u1, u2, (s + 1) & 1, (s + 2) & 1);
      }
    }

    // epilogue (per row-group, the classic c-frag map) — VERBATIM
    const int g = lane >> 2, tp = (lane & 3) * 2;
    const int n = nb * NTILE + warp * 8 + tp;
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg) {
      const int m0 = mb * MTILE + rg * 16 + g;
#if RES
      out32[(size_t)m0 * NDIM + n]           = res16[(size_t)m0 * NDIM + n] + ACC(0, rg, 0);
      out32[(size_t)m0 * NDIM + n + 1]       = res16[(size_t)m0 * NDIM + n + 1] + ACC(0, rg, 1);
      out32[(size_t)(m0 + 8) * NDIM + n]     = res16[(size_t)(m0 + 8) * NDIM + n] + ACC(0, rg, 2);
      out32[(size_t)(m0 + 8) * NDIM + n + 1] = res16[(size_t)(m0 + 8) * NDIM + n + 1] + ACC(0, rg, 3);
#elif FFN
      const __half hg0 = (__half)ACC(0, rg, 0), hg1 = (__half)ACC(0, rg, 1);
      const __half hg2 = (__half)ACC(0, rg, 2), hg3 = (__half)ACC(0, rg, 3);
      const __half hn0 = hsilu_h(hg0), hn1 = hsilu_h(hg1), hn2 = hsilu_h(hg2), hn3 = hsilu_h(hg3);
      *(__half2*)(out16 + (size_t)m0 * NDIM + n)       = __halves2half2(__hmul(hn0, (__half)ACC(1, rg, 0)), __hmul(hn1, (__half)ACC(1, rg, 1)));
      *(__half2*)(out16 + (size_t)(m0 + 8) * NDIM + n) = __halves2half2(__hmul(hn2, (__half)ACC(1, rg, 2)), __hmul(hn3, (__half)ACC(1, rg, 3)));
#else
      *(__half2*)(out16 + (size_t)m0 * NDIM + n)       = __halves2half2((__half)ACC(0, rg, 0), (__half)ACC(0, rg, 1));
      *(__half2*)(out16 + (size_t)(m0 + 8) * NDIM + n) = __halves2half2((__half)ACC(0, rg, 2), (__half)ACC(0, rg, 3));
#endif
    }
#undef ACC
#undef MMARS
#undef CDECN
#undef CDECP
  }
}
