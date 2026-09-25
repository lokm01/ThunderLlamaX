// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P4 Stage 1c: multi-segment single-launch pGEMM. One launch runs 2-3
// weight classes back-to-back (segment selected by blockIdx.x), removing the
// inter-kernel launch gap. Body = the P1 classic single-buffered kernel
// VERBATIM (stage x -> stage W -> sync -> HMMA -> sync -> plain fp16 epilogue)
// with the per-segment (weight ptr, out ptr, NDIM, quant class) selected at
// block scope (uniform branch). All segments share KDIM/KCH/NTHR/NTILE and
// the SAME x16 input rows (true for qkv+gate and q+k+v: every consumer of
// xh16 in a block). Decode math copied byte-for-byte from pf_gemm.cu.
// Build: -DKNAME -DSEGN(2|3) -DKDIM -DNTHR -DNTILE -DKCH
//   per-seg A/B/C: -DQCA/-DNDA/-DGRID_A (, B, C)
// LAWS: flat indexing; no gridDim reads; hardcoded sizes; unrolled loops;
// single 16B-aligned smem array; per-kernel cubin + warp-token name; full
// masks (no shuffles).
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define XS_LD (KCH + 8)
#define WS_LD (KCH + 8)
#define NWARP (NTHR / 32)
#define RPT (NTILE / NWARP)
#define NCL (KCH / 32)
#define KSTEPS (KCH / 16)

__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- per-class decode (VERBATIM from pf_gemm.cu, explicitly named) ----
__device__ __forceinline__ void dq_c1(const unsigned char* rowp, const float* gridf,
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

__device__ __forceinline__ void dq_c2(const unsigned char* rowp, int b, int lc, __half* w8) {
  const unsigned char* blk = rowp + b*176;
  const float d = __half2float(*((const __half*)blk));
  const float dm = __half2float(*((const __half*)(blk+2)));
  const int s = lc >> 2;
  float sc, mn;
  if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
  else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
         mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
  const unsigned char* lo = blk + 48 + ((lc>>3)<<5) + ((lc&3)<<3);
  const unsigned char* qh = blk + 16 + ((lc&3)<<3);
  const int nsh = ((lc>>2)&1) << 2;
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) {
    int qv = (lo[j] >> nsh) & 0xF;
    qv += (int)(((qh[j] >> s) & 1u) << 4);
    w8[j] = __float2half(d*sc*(float)qv - dm*mn);
  }
}

__device__ __forceinline__ void dq_c3(const unsigned char* rowp, int b, int lc, __half* w8) {
  const unsigned char* blk = rowp + b*212;
  const float d = __half2float(*((const __half*)(blk+2)));
  const bool nib_hi = ((lc&15) >= 8);
  const int c2 = (lc>>2)&3;
  const unsigned char* lo = blk + 20 + ((lc>>4)<<6) + ((lc&7)<<3);
  const unsigned char* qh = blk + 148 + ((lc>>4)<<5) + ((lc&3)<<3);
  const int sc8 = (signed char)blk[4 + (lc>>1)];
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) {
    const int xl = nib_hi ? (lo[j] >> 4) & 0xF : lo[j] & 0xF;
    const int xh2 = ((qh[j] >> (c2<<1)) & 3) << 4;
    w8[j] = __float2half(d * (float)sc8 * (float)((signed char)((xl | xh2) - 32)));
  }
}

__device__ __forceinline__ void dq_c4(const unsigned char* rowp, int b, int lc, __half* w8) {
  const unsigned char* blk = rowp + b*144;
  const float d = __half2float(*((const __half*)blk));
  const float dm = __half2float(*((const __half*)(blk+2)));
  const int s = lc >> 2;
  float sc, mn;
  if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
  else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
         mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
  const unsigned char* lo = blk + 16 + ((lc>>3)<<5) + ((lc&3)<<3);
  const int nsh = ((lc>>2)&1) << 2;
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) {
    const float qv = (float)((lo[j] >> nsh) & 0xF);
    w8[j] = __float2half(d*sc*qv - dm*mn);
  }
}

__device__ __forceinline__ void dq_seg(int cls, const unsigned char* rowp, const float* gridf,
                                       int b, int lc, __half* w8) {
  if (cls == 1)      dq_c1(rowp, gridf, b, lc, w8);
  else if (cls == 2) dq_c2(rowp, b, lc, w8);
  else if (cls == 3) dq_c3(rowp, b, lc, w8);
  else               dq_c4(rowp, b, lc, w8);
}

__device__ __forceinline__ int rowb_seg(int cls) {
  if (cls == 1) return 98 * (KDIM >> 8);
  if (cls == 2) return 176 * (KDIM >> 8);
  if (cls == 3) return 212 * (KDIM >> 8);
  return 144 * (KDIM >> 8);
}

// P6 M32: 32 x-rows per CTA (see pf_gemm.cu P6 section) — two m16 fragments
// share every staged W tile; segment select + decode VERBATIM the M=16 kernel.
#ifndef M32
#define M32 0
#endif

#if !M32
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ wA, const unsigned char* __restrict__ wB,
#if SEGN > 2
    const unsigned char* __restrict__ wC,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
    __half* __restrict__ outA, __half* __restrict__ outB
#if SEGN > 2
    , __half* __restrict__ outC
#endif
    )
{
  __shared__ __align__(16) __half sm[16 * XS_LD + NTILE * WS_LD];
  __half* xs = sm;
  __half* ws = sm + 16 * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int bid = blockIdx.x;
  const unsigned char* w1; __half* out16; int NDL, CLS;
  if (bid < GRID_A)             { w1 = wA; out16 = outA; NDL = NDA; CLS = QCA; }
  else if (bid < GRID_A+GRID_B) { w1 = wB; out16 = outB; NDL = NDB; CLS = QCB; }
#if SEGN > 2
  else                          { w1 = wC; out16 = outC; NDL = NDC; CLS = QCC; }
#endif
  const int n0 = (bid - (bid >= GRID_A ? (bid >= GRID_A+GRID_B ? GRID_A+GRID_B : GRID_A) : 0)) * NTILE;
  const int r = lane >> 2, c = lane & 3;
  const unsigned char* rowp = w1 + (size_t)(n0 + warp*RPT + r) * rowb_seg(CLS);
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  _Pragma("unroll 2")
  for (int kc = 0; kc < KDIM; kc += KCH) {
    // stage x
    _Pragma("unroll")
    for (int i = 0; i < (16*KCH + NTHR*4 - 1)/(NTHR*4); ++i) {
      const int lin = (tid + i*NTHR) * 4;
      if (lin < 16*KCH) {
        const int m = lin / KCH, kk = lin % KCH;
        *(uint2*)(xs + (size_t)m*XS_LD + kk) = *(const uint2*)(x16 + (size_t)m*KDIM + kc + kk);
      }
    }
    // stage W rows
    {
      const int lc0 = (kc & 255) >> 3;
      _Pragma("unroll")
      for (int cc = 0; cc < NCL; ++cc) {
        const int lc = lc0 + c*NCL + cc;
        __half w8[8];
        dq_seg(CLS, rowp, gridf, (kc >> 8), lc, w8);
        const int kk = (c*NCL + cc)*8;
        _Pragma("unroll")
        for (int j = 0; j < 8; ++j) ws[(size_t)(warp*RPT + r)*WS_LD + kk + j] = w8[j];
      }
    }
    __syncthreads();
    // HMMA (pf_gemm.cu fragment map, verbatim)
    _Pragma("unroll")
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s*16;
      const int g = lane >> 2, tp = (lane & 3) * 2;
      const unsigned a0 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp);
      const unsigned a1 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp);
      const unsigned a2 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp + 8);
      const unsigned a3 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp + 8);
      const __half* wr = ws + (size_t)(warp*8 + g)*WS_LD + kb + tp;
      const unsigned b0 = *(const unsigned*)(wr);
      const unsigned b1 = *(const unsigned*)(wr + 8);
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();
  }
  const int g = lane >> 2, tp = (lane & 3) * 2;
  const int n = n0 + warp*8 + tp;
  *(__half2*)(out16 + (size_t)g*NDL + n)     = __halves2half2((__half)c0, (__half)c1);
  *(__half2*)(out16 + (size_t)(g+8)*NDL + n) = __halves2half2((__half)c2, (__half)c3);
}
#endif  // !M32

// ================= P6 M32 twin (32 x-rows; segment structure unchanged) ======
#if M32
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ wA, const unsigned char* __restrict__ wB,
#if SEGN > 2
    const unsigned char* __restrict__ wC,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
    __half* __restrict__ outA, __half* __restrict__ outB
#if SEGN > 2
    , __half* __restrict__ outC
#endif
    )
{
  __shared__ __align__(16) __half sm[32 * XS_LD + NTILE * WS_LD];
  __half* xs = sm;
  __half* ws = sm + 32 * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int bid = blockIdx.x;
  const unsigned char* w1; __half* out16; int NDL, CLS;
  if (bid < GRID_A)             { w1 = wA; out16 = outA; NDL = NDA; CLS = QCA; }
  else if (bid < GRID_A+GRID_B) { w1 = wB; out16 = outB; NDL = NDB; CLS = QCB; }
#if SEGN > 2
  else                          { w1 = wC; out16 = outC; NDL = NDC; CLS = QCC; }
#endif
  const int n0 = (bid - (bid >= GRID_A ? (bid >= GRID_A+GRID_B ? GRID_A+GRID_B : GRID_A) : 0)) * NTILE;
  const int r = lane >> 2, c = lane & 3;
  const unsigned char* rowp = w1 + (size_t)(n0 + warp*RPT + r) * rowb_seg(CLS);
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  float c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;

  _Pragma("unroll 2")
  for (int kc = 0; kc < KDIM; kc += KCH) {
    // stage x: 32 rows
    _Pragma("unroll")
    for (int i = 0; i < (32*KCH + NTHR*4 - 1)/(NTHR*4); ++i) {
      const int lin = (tid + i*NTHR) * 4;
      if (lin < 32*KCH) {
        const int m = lin / KCH, kk = lin % KCH;
        *(uint2*)(xs + (size_t)m*XS_LD + kk) = *(const uint2*)(x16 + (size_t)m*KDIM + kc + kk);
      }
    }
    // stage W rows (VERBATIM M=16)
    {
      const int lc0 = (kc & 255) >> 3;
      _Pragma("unroll")
      for (int cc = 0; cc < NCL; ++cc) {
        const int lc = lc0 + c*NCL + cc;
        __half w8[8];
        dq_seg(CLS, rowp, gridf, (kc >> 8), lc, w8);
        const int kk = (c*NCL + cc)*8;
        _Pragma("unroll")
        for (int j = 0; j < 8; ++j) ws[(size_t)(warp*RPT + r)*WS_LD + kk + j] = w8[j];
      }
    }
    __syncthreads();
    // HMMA doubled (fragment map verbatim; b shared by both row-groups)
    _Pragma("unroll")
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s*16;
      const int g = lane >> 2, tp = (lane & 3) * 2;
      const unsigned a0 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp);
      const unsigned a1 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp);
      const unsigned a2 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp + 8);
      const unsigned a3 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp + 8);
      const unsigned a4 = *(const unsigned*)(xs + (size_t)(g+16)*XS_LD + kb + tp);
      const unsigned a5 = *(const unsigned*)(xs + (size_t)(g+24)*XS_LD + kb + tp);
      const unsigned a6 = *(const unsigned*)(xs + (size_t)(g+16)*XS_LD + kb + tp + 8);
      const unsigned a7 = *(const unsigned*)(xs + (size_t)(g+24)*XS_LD + kb + tp + 8);
      const __half* wr = ws + (size_t)(warp*8 + g)*WS_LD + kb + tp;
      const unsigned b0 = *(const unsigned*)(wr);
      const unsigned b1 = *(const unsigned*)(wr + 8);
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
        : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b0), "r"(b1));
    }
    __syncthreads();
  }
  const int g = lane >> 2, tp = (lane & 3) * 2;
  const int n = n0 + warp*8 + tp;
  *(__half2*)(out16 + (size_t)g*NDL + n)      = __halves2half2((__half)c0, (__half)c1);
  *(__half2*)(out16 + (size_t)(g+8)*NDL + n)  = __halves2half2((__half)c2, (__half)c3);
  *(__half2*)(out16 + (size_t)(g+16)*NDL + n) = __halves2half2((__half)c4, (__half)c5);
  *(__half2*)(out16 + (size_t)(g+24)*NDL + n) = __halves2half2((__half)c6, (__half)c7);
}
#endif  // M32
