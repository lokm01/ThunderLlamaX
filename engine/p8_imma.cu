// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P8 IMMA discriminator: W4A8 prefill FFN GEMM at the census-worst shape
// (M=64, K=5120, N=17408 fg+fu-class) — mma.sync.m16n8k32.s32.s8.s8.s32 vs
// the shipped fp16-HMMA packed7 stream. SPEED DISCRIMINATOR ONLY (Tier-2):
// weights are random int4+scales; numerics checked vs a numpy reference of
// the SAME random data. Scheme: X s8 per-(row,128k-chunk) scale sx; W nib s8
// (0..15) per-(row,256k-block) fp16 scale sw; offset-8 correction via
// per-(row,chunk) rowsums: out = sx*sw*(sum xq*wq - 8*rowsum) per chunk,
// fp32-accumulated; int32 mma partials WITHIN a chunk.
// W layout = the packed7 unit pattern (group/chunk/32 uint4; 16B loads).
// X staging goes through REGISTERS (the pf_gemm3 XPR/XCM law — direct
// smem writes before mma race). LAWS: flat indexing, hardcoded sizes, one
// kernel per cubin, uint4 fully consumed, 0 spill.
// Build: -DKNAME -DKDIM=5120 -DNDIM=17408 -DNTHR=256 -DNTILE=64 -DKCH=128 -DMTILE=64
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

#define XPR8(R, CH) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR) * 4; \
    if (lin < MTILE*KCH) { \
      const int m = lin / KCH, kk = lin % KCH; \
      x##R[t] = *(const int*)(xq + (size_t)m*KDIM + (size_t)(CH)*KCH + kk); \
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

#define DEC5(R) do { \
  const unsigned* uu = (const unsigned*)&u##R; \
  _Pragma("unroll") \
  for (int cc = 0; cc < 4; ++cc) { \
    _Pragma("unroll") \
    for (int j = 0; j < 8; ++j) { \
      const int w = cc*8 + j; \
      const int nib = (uu[w >> 3] >> (4*(w & 7))) & 0xF; \
      ws[(size_t)(warp*8 + r_)*WS_LD + qc_*32 + w] = (signed char)nib; \
    } \
  } \
} while (0)

#define MMAR8(CH) do { \
  int acc[MRG][4]; \
  _Pragma("unroll") \
  for (int rg = 0; rg < MRG; ++rg) \
    _Pragma("unroll") \
    for (int j = 0; j < 4; ++j) acc[rg][j] = 0; \
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
      const unsigned b0 = *(const unsigned*)(ws + (size_t)(warp*8 + g)*WS_LD + kb + t4); \
      const unsigned b1 = *(const unsigned*)(ws + (size_t)(warp*8 + g)*WS_LD + kb + 16 + t4); \
      imma16832(acc[rg][0], acc[rg][1], acc[rg][2], acc[rg][3], a0, a1, a2, a3, b0, b1); \
    } \
  } \
  const __half swh0 = swd[(size_t)(n0 + warp*8 + (lane & 3)*2) * (KDIM >> 8) + ((CH) >> 1)]; \
  const __half swh1 = swd[(size_t)(n0 + warp*8 + (lane & 3)*2 + 1) * (KDIM >> 8) + ((CH) >> 1)]; \
  _Pragma("unroll") \
  for (int rg = 0; rg < MRG; ++rg) { \
    const float sxg = sx[(size_t)(rg*16 + g)*NCH + (CH)] * __half2float(swh0); \
    const float sxg8 = sx[(size_t)(rg*16 + g)*NCH + (CH)] * __half2float(swh1); \
    const int rsg = rowsum[(size_t)(rg*16 + g)*NCH + (CH)]; \
    const int rsg8 = rowsum[(size_t)(rg*16 + 8 + g)*NCH + (CH)]; \
    const float sxh = sx[(size_t)(rg*16 + 8 + g)*NCH + (CH)] * __half2float(swh0); \
    const float sxh8 = sx[(size_t)(rg*16 + 8 + g)*NCH + (CH)] * __half2float(swh1); \
    facc[rg][0] += sxg  * (float)(acc[rg][0] - 8*rsg); \
    facc[rg][1] += sxg8 * (float)(acc[rg][1] - 8*rsg); \
    facc[rg][2] += sxh  * (float)(acc[rg][2] - 8*rsg8); \
    facc[rg][3] += sxh8 * (float)(acc[rg][3] - 8*rsg8); \
  } \
} while (0)

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w4,     // [NG/8][NCH][32][16] nibble units
    const __half* __restrict__ swd,           // [NDIM][KDIM/256] scales
    const signed char* __restrict__ xq,       // [MTILE][KDIM] s8 acts
    const float* __restrict__ sx,             // [MTILE][NCH]
    const int* __restrict__ rowsum,           // [MTILE][NCH]
    float* __restrict__ out32)                // [MTILE][NDIM] f32
{
  __shared__ __align__(16) signed char xs[MTILE * XS_LD];
  __shared__ __align__(16) signed char ws[NTILE * WS_LD];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int nb = blockIdx.x % NGRID;
  const int n0 = nb * NTILE;
  const int r_ = lane >> 2, qc_ = lane & 3;
  const uint4* w4u = (const uint4*)w4;
  const size_t goff = ((size_t)(n0 / 8 + warp) * NCH) * 32 + lane;

  float facc[MRG][4];
  _Pragma("unroll")
  for (int rg = 0; rg < MRG; ++rg)
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) facc[rg][j] = 0.f;

  int xE[XTPR], xO[XTPR];
  uint4 uE, uO;
  uE = w4u[goff];
  XPR8(E, 0); XCM8(E);
  DEC5(E);
  __syncthreads();
  _Pragma("unroll 1")
  for (int s = 0; s + 1 < NCH; s += 2) {
    uO = w4u[goff + (size_t)(s + 1) * 32];
    XPR8(O, (s + 1));
    MMAR8(s);
    __syncthreads();
    DEC5(O); XCM8(O);
    __syncthreads();
    if (s + 2 >= NCH) break;
    uE = w4u[goff + (size_t)(s + 2) * 32];
    XPR8(E, (s + 2));
    MMAR8(s + 1);
    __syncthreads();
    DEC5(E); XCM8(E);
    __syncthreads();
  }
  MMAR8(NCH - 1);   // the final chunk (staged by the last iteration before break)

  const int g = lane >> 2, tp2 = (lane & 3) * 2;
  _Pragma("unroll")
  for (int rg = 0; rg < MRG; ++rg) {
    out32[(size_t)(rg*16 + g)*NDIM + n0 + warp*8 + tp2]         = facc[rg][0];
    out32[(size_t)(rg*16 + g)*NDIM + n0 + warp*8 + tp2 + 1]     = facc[rg][1];
    out32[(size_t)(rg*16 + 8 + g)*NDIM + n0 + warp*8 + tp2]     = facc[rg][2];
    out32[(size_t)(rg*16 + 8 + g)*NDIM + n0 + warp*8 + tp2 + 1] = facc[rg][3];
  }
}
