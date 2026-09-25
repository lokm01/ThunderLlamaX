// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P8w4: the FUSED W4A8 prefill FFN GEMM — gate+up planes in ONE launch, the
// silu(g)*u epilogue of pfg3_ffn (fp16 out16 [M][NDIM], fd consumes it
// unchanged). IMMA mma.sync.m16n8k32.s8.s8.s32; scheme = the P8 discriminator:
// X s8 per-(row,128k-chunk) scale sx + rowsums; W nib s8 (0..15) per-(row,128k)
// fp16 scale swd; offset-8 correction per chunk: facc += sx*sw*(sum xq*wq
// - 8*rowsum), fp32. Per-128 scales (v2 — one scale per chunk, swd[row][CH]).
// M-grid: mb = blockIdx.x / NGRID (pf_gemm3 law) — M64 plan g=272, M128 g=544.
// W layout = packed4 units (pack_w4): [group][chunk][unit r*4+qc] uint4, nib
// w at bits 4w, k = chunk*128 + qc*32 + w. X through REGISTERS (XPR/XCM law).
// smem = ONE 16B-aligned array 27648B (in-graph legal). LAWS: flat indexing,
// hardcoded sizes, one kernel per cubin, uint4 fully consumed, 0 spill.
// Build: -DKNAME=p8w4ffn -DKDIM=5120 -DNDIM=17408 -DNTHR=256 -DNTILE=64 -DKCH=128 -DMTILE=64
#include <cuda_fp16.h>
#define XS_LD (KCH + 16)
#define WS_LD (KCH + 16)
#define NWARP (NTHR / 32)
#define RPT (NTILE / NWARP)
#if RPT != 8
#error "NTILE/NWARP must be 8"
#endif
#define KSTEPS (KCH / 32)
#define NCH (KDIM / KCH)
#define NGRID (NDIM / NTILE)
#define MRG (MTILE / 16)
#define XTPR ((MTILE * KCH + NTHR * 4 - 1) / (NTHR * 4))

__device__ __forceinline__ void imma16832(int &c0, int &c1, int &c2, int &c3,
    const unsigned a0, const unsigned a1, const unsigned a2, const unsigned a3,
    const unsigned b0, const unsigned b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

#define XPR8(R, CH) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR) * 4; \
    if (lin < MTILE*KCH) { \
      const int m = lin / KCH, kk = lin % KCH; \
      x##R[t] = *(const int*)(xq + (size_t)(mbm + m)*KDIM + (size_t)(CH)*KCH + kk); \
    } \
  } \
} while (0)

#define XCM8(R) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR) * 4; \
    if (lin < MTILE*KCH) { \
      const int m = lin / KCH, kk = lin % KCH; \
      *(int*)(xs + m*XS_LD + kk) = x##R[t]; \
    } \
  } \
} while (0)

#define DEC5(R, WSB) do { \
  const unsigned* uu = (const unsigned*)&u##R; \
  _Pragma("unroll") \
  for (int cc = 0; cc < 4; ++cc) { \
    _Pragma("unroll") \
    for (int j = 0; j < 8; ++j) { \
      const int w = cc*8 + j; \
      const int nib = (uu[w >> 3] >> (4*(w & 7))) & 0xF; \
      (WSB)[(size_t)(warp*8 + r_)*WS_LD + qc_*32 + w] = (signed char)nib; \
    } \
  } \
} while (0)

#define MMAR8(CH) do { \
  int ag[MRG][4], au[MRG][4]; \
  _Pragma("unroll") \
  for (int rg = 0; rg < MRG; ++rg) \
    _Pragma("unroll") \
    for (int j = 0; j < 4; ++j) { ag[rg][j] = 0; au[rg][j] = 0; } \
  const int g = lane >> 2, t4 = (lane & 3) * 4; \
  _Pragma("unroll") \
  for (int st = 0; st < KSTEPS; ++st) { \
    const int kb = st * 32; \
    _Pragma("unroll") \
    for (int rg = 0; rg < MRG; ++rg) { \
      const unsigned a0 = *(const unsigned*)(xs + (size_t)(rg*16 + g)*XS_LD + kb + t4); \
      const unsigned a1 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8)*XS_LD + kb + t4); \
      const unsigned a2 = *(const unsigned*)(xs + (size_t)(rg*16 + g)*XS_LD + kb + 16 + t4); \
      const unsigned a3 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8)*XS_LD + kb + 16 + t4); \
      const unsigned bg0 = *(const unsigned*)(ws + (size_t)(warp*8 + g)*WS_LD + kb + t4); \
      const unsigned bg1 = *(const unsigned*)(ws + (size_t)(warp*8 + g)*WS_LD + kb + 16 + t4); \
      const unsigned bu0 = *(const unsigned*)(wu + (size_t)(warp*8 + g)*WS_LD + kb + t4); \
      const unsigned bu1 = *(const unsigned*)(wu + (size_t)(warp*8 + g)*WS_LD + kb + 16 + t4); \
      imma16832(ag[rg][0], ag[rg][1], ag[rg][2], ag[rg][3], a0, a1, a2, a3, bg0, bg1); \
      imma16832(au[rg][0], au[rg][1], au[rg][2], au[rg][3], a0, a1, a2, a3, bu0, bu1); \
    } \
  } \
  const float swg0 = __half2float(swd1[(size_t)(n0 + warp*8 + (lane & 3)*2) * NCH + (CH)]); \
  const float swg1 = __half2float(swd1[(size_t)(n0 + warp*8 + (lane & 3)*2 + 1) * NCH + (CH)]); \
  const float swu0 = __half2float(swd2[(size_t)(n0 + warp*8 + (lane & 3)*2) * NCH + (CH)]); \
  const float swu1 = __half2float(swd2[(size_t)(n0 + warp*8 + (lane & 3)*2 + 1) * NCH + (CH)]); \
  _Pragma("unroll") \
  for (int rg = 0; rg < MRG; ++rg) { \
    const float sxg  = sx[(size_t)(mbm + rg*16 + g)*NCH + (CH)]; \
    const float sxh  = sx[(size_t)(mbm + rg*16 + 8 + g)*NCH + (CH)]; \
    const int rsg  = rowsum[(size_t)(mbm + rg*16 + g)*NCH + (CH)]; \
    const int rsg8 = rowsum[(size_t)(mbm + rg*16 + 8 + g)*NCH + (CH)]; \
    fg[rg][0] += (sxg*swg0) * (float)(ag[rg][0] - 8*rsg); \
    fg[rg][1] += (sxg*swg1) * (float)(ag[rg][1] - 8*rsg); \
    fg[rg][2] += (sxh*swg0) * (float)(ag[rg][2] - 8*rsg8); \
    fg[rg][3] += (sxh*swg1) * (float)(ag[rg][3] - 8*rsg8); \
    fu[rg][0] += (sxg*swu0) * (float)(au[rg][0] - 8*rsg); \
    fu[rg][1] += (sxg*swu1) * (float)(au[rg][1] - 8*rsg); \
    fu[rg][2] += (sxh*swu0) * (float)(au[rg][2] - 8*rsg8); \
    fu[rg][3] += (sxh*swu1) * (float)(au[rg][3] - 8*rsg8); \
  } \
} while (0)

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1u,     // fg nibble units [NG/8][NCH][32][16B]
    const unsigned char* __restrict__ w2u,     // fu nibble units
    const __half* __restrict__ swd1,           // fg scales [NDIM][NCH]
    const __half* __restrict__ swd2,           // fu scales [NDIM][NCH]
    const signed char* __restrict__ xq,        // [Mrows][KDIM] s8 acts
    const float* __restrict__ sx,              // [Mrows][NCH]
    const int* __restrict__ rowsum,            // [Mrows][NCH]
    __half* __restrict__ out16)                // [Mrows][NDIM] f16 silu(g)*u
{
  __shared__ __align__(16) signed char sm[MTILE * XS_LD + 2 * NTILE * WS_LD];
  signed char* xs = sm;
  signed char* ws = sm + MTILE * XS_LD;
  signed char* wu = ws + NTILE * WS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x / NGRID;
  const int nb = blockIdx.x - mb * NGRID;
  const int mbm = mb * MTILE;
  const int n0 = nb * NTILE;
  const int r_ = lane >> 2, qc_ = lane & 3;
  const uint4* w14 = (const uint4*)w1u;
  const uint4* w24 = (const uint4*)w2u;
  const size_t goff = ((size_t)(n0 / 8 + warp) * NCH) * 32 + lane;

  float fg[MRG][4], fu[MRG][4];
  _Pragma("unroll")
  for (int rg = 0; rg < MRG; ++rg)
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) { fg[rg][j] = 0.f; fu[rg][j] = 0.f; }

  int xE[XTPR], xO[XTPR];
  uint4 u1E, u1O, u2E, u2O;
  u1E = w14[goff];
  u2E = w24[goff];
  XPR8(E, 0); XCM8(E);
  DEC5(1E, ws);
  DEC5(2E, wu);
  __syncthreads();
  _Pragma("unroll 1")
  for (int s = 0; s + 1 < NCH; s += 2) {
    u1O = w14[goff + (size_t)(s + 1) * 32];
    u2O = w24[goff + (size_t)(s + 1) * 32];
    XPR8(O, (s + 1));
    MMAR8(s);
    __syncthreads();
    DEC5(1O, ws); DEC5(2O, wu);
    XCM8(O);
    __syncthreads();
    if (s + 2 >= NCH) break;
    u1E = w14[goff + (size_t)(s + 2) * 32];
    u2E = w24[goff + (size_t)(s + 2) * 32];
    XPR8(E, (s + 2));
    MMAR8(s + 1);
    __syncthreads();
    DEC5(1E, ws); DEC5(2E, wu);
    XCM8(E);
    __syncthreads();
  }
  MMAR8(NCH - 1);   // the final chunk (staged by the last iteration before break)

  // epilogue: the pfg3_ffn silu(g)*u fp16 write, c-frag map VERBATIM
  {
    const int g = lane >> 2, tp2 = (lane & 3) * 2;
    const int n = n0 + warp*8 + tp2;
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg) {
      const int m0 = mbm + rg*16 + g;
      const __half hg0 = (__half)fg[rg][0], hg1 = (__half)fg[rg][1];
      const __half hg2 = (__half)fg[rg][2], hg3 = (__half)fg[rg][3];
      const __half hn0 = hsilu_h(hg0), hn1 = hsilu_h(hg1), hn2 = hsilu_h(hg2), hn3 = hsilu_h(hg3);
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     =
          __halves2half2(__hmul(hn0, (__half)fu[rg][0]), __hmul(hn1, (__half)fu[rg][1]));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) =
          __halves2half2(__hmul(hn2, (__half)fu[rg][2]), __hmul(hn3, (__half)fu[rg][3]));
    }
  }
}
