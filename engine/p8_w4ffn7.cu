// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P8w4-v2: the fused W4A8 FFN GEMM reading the EXISTING packed7 planes (zero
// new VRAM). IMMA m16n8k32.s8.s8.s32 with the IQ3_XXS codebook LINEARIZED:
// the grid has 8 unique levels [4,12,20,28,36,44,52,62] ~= delta*(1..15),
// delta=4.0507 -> per-weight error 1.49% RMS (8.4x better than a raw int4
// requant; offline-verified on real tensors). W operand = s8 sign*level (NO
// offset correction); per-(row,256b) scale sdb = fp16(d)*((sw>>28)+0.5)*0.5
// *delta (dq_iq3_r VERBATIM ints), stashed fp16 wsc/wuc[NTILE][NCH/2] smem;
// X = per-(row,128k-chunk) int8 (pfk_q8, sx only); per-chunk rescale
// facc += sx*sdb*acc. Fused gate+up + the pf_gemm3 FFN silu*u fp16 epilogue;
// M-grid mb = blockIdx.x/NGRID (m64 g=272, m128 g=544).
// packed7 unit (pack_w7): [group][chunk][unit r*4+qc] 16B = q u16 x4
// (word cc covers lc = lc0+qc*4+cc, 8 weights: rows (qv&0xFF),(qv>>8)),
// sw u32 (bits 28..31 the db exponent; bits 7*cc.. the sign packs), d u16.
// lut8 = 1024-entry s8: lut8[(row)*4+j] = clip(round(grid[row*4+j]/delta),0,15).
// smem = 27,648B planes + 2x2,560B scales + 1,024B lut = 33,792B (in-graph OK).
// LAWS: flat indexing, hardcoded sizes, one kernel per cubin, uint4 fully
// consumed, no char4, zero spill.
// Build (build_p8w4.py, symbol p8w4ffn7): -DKDIM=5120 -DNDIM=17408 -DNTHR=256
//   -DNTILE=64 -DKCH=128 -DMTILE=64  (the shipped cubin: p8w4ffn7_nw8k128)
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
#define NCHB (NCH / 2)
// TLX W4.4 (V-59 audit): wsc/wuc stash [NTILE][NCH/2] — an odd NCH would
// truncate the last scale column silently. Preprocessor-only guard (zero
// codegen delta vs the shipped cubin; NCH = 5120/128 = 40 here).
#if (NCH % 2) != 0
#error "p8_w4ffn7: NCH must be even (NCHB = NCH/2 scale stashes)"
#endif
#define NGRID (NDIM / NTILE)
#define MRG (MTILE / 16)
#define XTPR ((MTILE * KCH + NTHR * 4 - 1) / (NTHR * 4))
#define LDELTA 4.0507f

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

// DEC7: decode a packed7 unit into the lane's 32 signed levels (WSB plane) +
// stash sdb (fp16) per (row, 256-block) into SCB.
#define DEC7_CORE(R, WSB, ISU) do { \
  const unsigned* uu = (const unsigned*)&u##R; \
  const unsigned qa = uu[0], qb = uu[1], swv = uu[2]; \
  const float d = __half2float(__ushort_as_half((unsigned short)(uu[3] & 0xFFFFu))); \
  const float dbd = d * (((float)(swv >> 28)) + 0.5f) * 0.5f * LDELTA; \
  _Pragma("unroll") \
  for (int cc = 0; cc < 4; ++cc) { \
    const unsigned qv = (cc == 0) ? (qa & 0xFFFFu) : (cc == 1) ? (qa >> 16) \
                    : (cc == 2) ? (qb & 0xFFFFu) : (qb >> 16); \
    const unsigned sidx = (swv >> (7u * (unsigned int)cc)) & 0x7Fu; \
    const int spar = (int)(((sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u)); \
    const signed char* l0 = lut + (((int)(qv & 0xFFu)) << 2); \
    const signed char* l1 = lut + (((int)(qv >> 8)) << 2); \
    /* two's-complement byte composition (explicit; the ternary-negation \
       int-shift form miscompiled/wrong on bytes 1-3) */ \
    unsigned pk0, pk1; \
    pk0 = (unsigned)(((sidx>>0)&1) ? (unsigned char)(256 - (int)l0[0]) : (unsigned char)l0[0]) \
        | (unsigned)((((sidx>>1)&1) ? (unsigned char)(256 - (int)l0[1]) : (unsigned char)l0[1]) << 8) \
        | (unsigned)((((sidx>>2)&1) ? (unsigned char)(256 - (int)l0[2]) : (unsigned char)l0[2]) << 16) \
        | (unsigned)((((sidx>>3)&1) ? (unsigned char)(256 - (int)l0[3]) : (unsigned char)l0[3]) << 24); \
    pk1 = (unsigned)(((sidx>>4)&1) ? (unsigned char)(256 - (int)l1[0]) : (unsigned char)l1[0]) \
        | (unsigned)((((sidx>>5)&1) ? (unsigned char)(256 - (int)l1[1]) : (unsigned char)l1[1]) << 8) \
        | (unsigned)((((sidx>>6)&1) ? (unsigned char)(256 - (int)l1[2]) : (unsigned char)l1[2]) << 16) \
        | (unsigned)((spar ? (unsigned char)(256 - (int)l1[3]) : (unsigned char)l1[3]) << 24); \
    *(unsigned*)((WSB) + (size_t)(warp*8 + r_)*WS_LD + qc_*32 + cc*8) = pk0; \
    *(unsigned*)((WSB) + (size_t)(warp*8 + r_)*WS_LD + qc_*32 + cc*8 + 4) = pk1; \
  } \
  if (ISU) dbdU = dbd; else dbdG = dbd; \
} while (0)
#define DEC7G(R, WSB) DEC7_CORE(R, WSB, 0)
#define DEC7U(R, WSB) DEC7_CORE(R, WSB, 1)

// per-k-step rescale: step st covers k [st*32, st*32+32) = EXACTLY one scale
// word (the db exponent (sw>>28) is PER 32-k-group); the scale for output col
// warp*8+R comes from lane (R*4 + st) via shuffles (its DEC quarter == st).
#define MMAR7(CH) do { \
  const int g = lane >> 2, t4 = (lane & 3) * 4; \
  const int rA = ((lane & 3) * 2) * 4, rB = ((lane & 3) * 2 + 1) * 4; \
  _Pragma("unroll") \
  for (int st = 0; st < KSTEPS; ++st) { \
    int ag[MRG][4], au[MRG][4]; \
    _Pragma("unroll") \
    for (int rg = 0; rg < MRG; ++rg) \
      _Pragma("unroll") \
      for (int j = 0; j < 4; ++j) { ag[rg][j] = 0; au[rg][j] = 0; } \
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
    const float swg0 = __shfl_sync(0xffffffffu, dbdG, rA + st); \
    const float swg1 = __shfl_sync(0xffffffffu, dbdG, rB + st); \
    const float swu0 = __shfl_sync(0xffffffffu, dbdU, rA + st); \
    const float swu1 = __shfl_sync(0xffffffffu, dbdU, rB + st); \
    _Pragma("unroll") \
    for (int rg = 0; rg < MRG; ++rg) { \
      const float sxg  = sx[(size_t)(mbm + rg*16 + g)*NCH + (CH)]; \
      const float sxh  = sx[(size_t)(mbm + rg*16 + 8 + g)*NCH + (CH)]; \
      fg[rg][0] += (sxg*swg0) * (float)ag[rg][0]; \
      fg[rg][1] += (sxg*swg1) * (float)ag[rg][1]; \
      fg[rg][2] += (sxh*swg0) * (float)ag[rg][2]; \
      fg[rg][3] += (sxh*swg1) * (float)ag[rg][3]; \
      fu[rg][0] += (sxg*swu0) * (float)au[rg][0]; \
      fu[rg][1] += (sxg*swu1) * (float)au[rg][1]; \
      fu[rg][2] += (sxh*swu0) * (float)au[rg][2]; \
      fu[rg][3] += (sxh*swu1) * (float)au[rg][3]; \
    } \
  } \
} while (0)

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1u,     // fg packed7 units [NG/8][NCH][32][16B]
    const unsigned char* __restrict__ w2u,     // fu packed7 units
    const signed char* __restrict__ lut8,      // [1024] linearized codebook
    const signed char* __restrict__ xq,        // [Mrows][KDIM] s8 acts
    const float* __restrict__ sx,              // [Mrows][NCH]
    __half* __restrict__ out16                 // [Mrows][NDIM] f16 silu(g)*u
#if DUMPRAW
    , float* __restrict__ gout, float* __restrict__ uout  // raw facc dumps (dbg)
#endif
    )
{
  __shared__ __align__(16) signed char sm[MTILE * XS_LD + 2 * NTILE * WS_LD];
  __shared__ __align__(16) signed char luts[1024];
  signed char* xs = sm;
  signed char* ws = sm + MTILE * XS_LD;
  signed char* wu = ws + NTILE * WS_LD;
  signed char* lut = luts;
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

  for (int t = tid; t < 1024; t += NTHR) luts[t] = lut8[t];
  __syncthreads();

  float fg[MRG][4], fu[MRG][4];
  _Pragma("unroll")
  for (int rg = 0; rg < MRG; ++rg)
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) { fg[rg][j] = 0.f; fu[rg][j] = 0.f; }

  int xE[XTPR], xO[XTPR];
  float dbdG = 0.f, dbdU = 0.f;
  uint4 u1E, u1O, u2E, u2O;
  u1E = w14[goff];
  u2E = w24[goff];
  XPR8(E, 0); XCM8(E);
  DEC7G(1E, ws); DEC7U(2E, wu);
  __syncthreads();
  _Pragma("unroll 1")
  for (int s = 0; s + 1 < NCH; s += 2) {
    u1O = w14[goff + (size_t)(s + 1) * 32];
    u2O = w24[goff + (size_t)(s + 1) * 32];
    XPR8(O, (s + 1));
    MMAR7(s);
    __syncthreads();
    DEC7G(1O, ws); DEC7U(2O, wu);
    XCM8(O);
    __syncthreads();
    if (s + 2 >= NCH) break;
    u1E = w14[goff + (size_t)(s + 2) * 32];
    u2E = w24[goff + (size_t)(s + 2) * 32];
    XPR8(E, (s + 2));
    MMAR7(s + 1);
    __syncthreads();
    DEC7G(1E, ws); DEC7U(2E, wu);
    XCM8(E);
    __syncthreads();
  }
  MMAR7(NCH - 1);

  // epilogue: the pfg3_ffn silu(g)*u fp16 write, c-frag map VERBATIM
  {
    const int g = lane >> 2, tp2 = (lane & 3) * 2;
    const int n = n0 + warp*8 + tp2;
    _Pragma("unroll")
    for (int rg = 0; rg < MRG; ++rg) {
      const int m0 = mbm + rg*16 + g;
#if DUMPRAW
      gout[(size_t)m0*NDIM + n]         = fg[rg][0];
      gout[(size_t)m0*NDIM + n + 1]     = fg[rg][1];
      gout[(size_t)(m0+8)*NDIM + n]     = fg[rg][2];
      gout[(size_t)(m0+8)*NDIM + n + 1] = fg[rg][3];
      uout[(size_t)m0*NDIM + n]         = fu[rg][0];
      uout[(size_t)m0*NDIM + n + 1]     = fu[rg][1];
      uout[(size_t)(m0+8)*NDIM + n]     = fu[rg][2];
      uout[(size_t)(m0+8)*NDIM + n + 1] = fu[rg][3];
#else
      const __half hg0 = (__half)fg[rg][0], hg1 = (__half)fg[rg][1];
      const __half hg2 = (__half)fg[rg][2], hg3 = (__half)fg[rg][3];
      const __half hn0 = hsilu_h(hg0), hn1 = hsilu_h(hg1), hn2 = hsilu_h(hg2), hn3 = hsilu_h(hg3);
      *(__half2*)(out16 + (size_t)m0*NDIM + n)     =
          __halves2half2(__hmul(hn0, (__half)fu[rg][0]), __hmul(hn1, (__half)fu[rg][1]));
      *(__half2*)(out16 + (size_t)(m0+8)*NDIM + n) =
          __halves2half2(__hmul(hn2, (__half)fu[rg][2]), __hmul(hn3, (__half)fu[rg][3]));
#endif
    }
  }
}
