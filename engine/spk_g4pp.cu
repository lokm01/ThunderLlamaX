// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2G L1: PIPELINED-STAGE variant of spk_g4qh2.cu (QH + PVH canonical math).
// STAGE is split into LOAD_RAW (global int4 -> registers, issued while the
// PREVIOUS tile's QK/PV compute runs) and STORE_SMEM (dequant + STS), so the
// 16KB/tile int8 KV DRAM traffic overlaps the dot phases instead of phase-
// locking with them (no cp.async - registers only, dext-safe). Each thread
// stages ONE 16B int4 of K (tid<512) or V (tid>=512): one LDG.128 + one
// scale LDG.16 per thread per tile (vs LDG.64 x2 + gathers) = max memory-
// level parallelism at 1024 threads. Dequant math per element is UNCHANGED
// and smem layout is UNCHANGED -> smem bytes and all outputs are BIT-
// IDENTICAL to spk_g4qh2 (verified standalone; no baseline regen needed).
// LAWS: single __shared__ array, 16B-aligned compile-time offsets, no
// blockDim/gridDim reads, full-mask shuffles, sequential tile loop, TILE=32.
// -DKNAME -DCTXK -DROWS (3|1) -DS -DCH -DTILE (32) -DNW (32) -DPVH
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define RMAX (6*ROWS)
#ifndef NW
  #define NW 8
#endif
#define NTHR (NW*32)
#define RPMAX ((RMAX + NW - 1) / NW)
#define LBOUNDS __launch_bounds__(NTHR)

#define SM_K0   0
#define SM_V0   (TILE*512)
#define SM_BYTES (TILE*1024)
#define KSZ(R, C)  (((R) * 512) + (((C) ^ ((R) & 7)) * 16))
#define VSZ(R, C)  (((R) * 512) + ((C) * 16))

#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )

union F4H8 { float4 f; __half2 h[4]; };
#define DQH2(R, S2) __hmul2(__hsub2((R), (__half2)__float2half2_rn(1152.f)), (S2))
__device__ __forceinline__ float4 dq8(const int2 iv, const __half s) {
  F4H8 o;
  const __half2 s2 = __half2half2(s);
  const unsigned x = (unsigned)iv.x, y = (unsigned)iv.y;
  const unsigned p0 = __byte_perm(x, 0x64646464u, 0x5140), p1 = __byte_perm(x, 0x64646464u, 0x7362);
  const unsigned p2 = __byte_perm(y, 0x64646464u, 0x5140), p3 = __byte_perm(y, 0x64646464u, 0x7362);
  o.h[0] = DQH2(*(const __half2*)&p0, s2);
  o.h[1] = DQH2(*(const __half2*)&p1, s2);
  o.h[2] = DQH2(*(const __half2*)&p2, s2);
  o.h[3] = DQH2(*(const __half2*)&p3, s2);
  return o.f;
}

// ---- pipelined stage halves. Thread mapping: tid<512 stages K 16B chunk
// ---- (row=tid>>4, chunk pair c2=tid&15), tid>=512 stages V likewise.
// ---- Registers raw+sc survive across the compute phase (the LDG for tile
// ---- t+1 is issued before QK/PV of tile t and consumed after).
#define LOAD_RAW(BASEL) { \
  const int h = tid & 511; \
  const int row = h >> 4, c2 = h & 15; \
  const int l = (BASEL) + row; \
  raw = make_int4(0, 0, 0, 0); sraw = __float2half(0.f); \
  if (l < l1) { \
    if (tid < 512) { \
      raw = *(const int4*)(Kc8 + (size_t)l*256 + c2*16); \
      sraw = Ksc[(size_t)l*8 + (c2 >> 1)]; \
    } else { \
      raw = *(const int4*)(Vc8 + (size_t)l*256 + c2*16); \
      sraw = Vsc[(size_t)l*8 + (c2 >> 1)]; \
    } \
  } \
}

#define STORE_SMEM(BASEL) { \
  const int h = tid & 511; \
  const int row = h >> 4, c2 = h & 15; \
  const int l = (BASEL) + row; \
  float4 k0 = make_float4(0.f, 0.f, 0.f, 0.f), k1 = make_float4(0.f, 0.f, 0.f, 0.f); \
  if (l < l1) { \
    k0 = dq8(make_int2(raw.x, raw.y), sraw); \
    k1 = dq8(make_int2(raw.z, raw.w), sraw); \
  } \
  if (tid < 512) { \
    *(float4*)(SM + SM_K0 + KSZ(row, 2*c2)) = k0; \
    *(float4*)(SM + SM_K0 + KSZ(row, 2*c2 + 1)) = k1; \
  } else { \
    *(float4*)(SM + SM_V0 + VSZ(row, 2*c2)) = k0; \
    *(float4*)(SM + SM_V0 + VSZ(row, 2*c2 + 1)) = k1; \
  } \
}

#define BUTTERFLY_MAX(V) _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) (V) = fmaxf((V), __shfl_xor_sync(FULL, (V), o));
#define BUTTERFLY_SUM(V) _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) (V) += __shfl_xor_sync(FULL, (V), o);

// ---- half2 QK row: q from qw16 (fp16), k fp16 in smem; 2x half2 accs,
// ---- unpack every 64 elements into the fp32 sc. Softmax epilogue unchanged.
#define G4_QKROW(R, Mv, Sv, A8, PR) { \
  const int t_ = (R) % ROWS; \
  const bool vld = (lpl < l1) && (lpl <= pos + t_); \
  float sc = 0.f; \
  __half2 ha0 = __float2half2_rn(0.f), ha1 = __float2half2_rn(0.f); \
  const __half2* qb = (const __half2*)(qw16 + (size_t)QROWIDX(R)*256); \
  _Pragma("unroll 8") \
  for (int c = 0; c < 32; ++c) { \
    const float4 kf4 = *(const float4*)(SM + kcur + KSZ(lane, c)); \
    const __half2* hk = (const __half2*)&kf4; \
    const float4 qa4 = *(const float4*)(qb + c*4); \
    const __half2* hq = (const __half2*)&qa4; \
    ha0 = __hfma2(hq[0], hk[0], ha0); \
    ha1 = __hfma2(hq[1], hk[1], ha1); \
    ha0 = __hfma2(hq[2], hk[2], ha0); \
    ha1 = __hfma2(hq[3], hk[3], ha1); \
    if ((c & 7) == 7) { \
      const float2 f0 = __half22float2(ha0), f1 = __half22float2(ha1); \
      sc += (f0.x + f0.y) + (f1.x + f1.y); \
      ha0 = __float2half2_rn(0.f); ha1 = __float2half2_rn(0.f); \
    } \
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

#ifdef PVH
// W2F L2: partial-half2 PV. p packed once per row per tile as half2, broadcast
// by 32-bit shfl; v half2 straight from smem; 4 HFMA2 per row into fp16x2
// accumulators, converted to fp32 and added to A8 at the 32-row tile boundary.
#define G4_PVROW(A8, PR) { \
  __half2 hA[4]; \
  _Pragma("unroll") for (int j2 = 0; j2 < 4; ++j2) hA[j2] = __float2half2_rn(0.f); \
  const __half2 pb = __half2half2(__float2half(PR)); \
  _Pragma("unroll") \
  for (int i2 = 0; i2 < TILE; ++i2) { \
    const __half2 p2 = __shfl_sync(FULL, pb, i2); \
    const float4 vf4 = *(const float4*)(SM + vcur + VSZ(i2, lane)); \
    const __half2* hv = (const __half2*)&vf4; \
    _Pragma("unroll") for (int j2 = 0; j2 < 4; ++j2) hA[j2] = __hfma2(hv[j2], p2, hA[j2]); \
  } \
  _Pragma("unroll") for (int j2 = 0; j2 < 4; ++j2) { \
    const float2 f = __half22float2(hA[j2]); \
    (A8)[2*j2] += f.x; (A8)[2*j2+1] += f.y; \
  } \
}
#else
#define G4_PVROW(A8, PR) _Pragma("unroll") \
  for (int i2 = 0; i2 < TILE; ++i2) { \
    const float p = __shfl_sync(FULL, (PR), i2); \
    const float4 vf4 = *(const float4*)(SM + vcur + VSZ(i2, lane)); \
    const __half2* hv = (const __half2*)&vf4; \
    const float2 g0 = __half22float2(hv[0]), g1 = __half22float2(hv[1]), g2 = __half22float2(hv[2]), g3 = __half22float2(hv[3]); \
    const float vv[8] = {g0.x, g0.y, g1.x, g1.y, g2.x, g2.y, g3.x, g3.y}; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) (A8)[j] += p * vv[j]; \
  }
#endif

extern "C" __global__ void LBOUNDS KNAME(
    const unsigned char* __restrict__ kv, const __half* __restrict__ sc, const __half* __restrict__ qw16,
    const int* __restrict__ pos_slot, float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
{
  const int g = blockIdx.x / S;
  const int s = blockIdx.x % S;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int pos = pos_slot[0];
  const int l0 = s * CH;
  const int l1 = (l0 + CH) < (pos + ROWS) ? (l0 + CH) : (pos + ROWS);
  const unsigned char* Kc8 = kv + (size_t)g * (CTXK*256);
  const unsigned char* Vc8 = kv + (size_t)(4 + g) * (CTXK*256);
  const __half* Ksc = sc + (size_t)g * (CTXK*8);
  const __half* Vsc = sc + (size_t)(4 + g) * (CTXK*8);
  __shared__ __align__(16) char SM[SM_BYTES];

  // pipeline stage registers: raw int4 + its dequant scale, live across the
  // compute phase of the previous tile.
  int4 raw; __half sraw;

  float msv[RPMAX], ssv[RPMAX], aav[RPMAX][8], prv[RPMAX];
  _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {
    msv[ri] = -1e30f; ssv[ri] = 0.f; prv[ri] = 0.f;
    _Pragma("unroll") for (int j = 0; j < 8; ++j) aav[ri][j] = 0.f;
  }

  if (l0 < l1) {
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    const int kcur = SM_K0, vcur = SM_V0;
    LOAD_RAW(l0)
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      const int lpl = tile + lane;
      STORE_SMEM(tile)
      __syncthreads();
      if (t + 1 < nt) LOAD_RAW(tile + TILE)
      _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {
        const int row = warp + ri*NW;
        if (row < RMAX) G4_QKROW(row, msv[ri], ssv[ri], aav[ri], prv[ri])
      }
      _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {
        const int row = warp + ri*NW;
        if (row < RMAX) G4_PVROW(aav[ri], prv[ri])
      }
      __syncthreads();
    }
  }

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
