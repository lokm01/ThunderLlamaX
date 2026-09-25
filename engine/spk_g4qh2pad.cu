// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2F QH: half2-QK variant of spk_g4q.cu (int8-KV fat-CTA split-KV K1).
// Everything identical (STAGE int8 dequant via 0x6400 PRMT, smem layout, PV
// fp32, partial writes) EXCEPT the QK dot: q comes from qw16 (fp16 copy of
// the fp32 qw rows, emitted by spk_preqh.cu) and the dot accumulates in TWO
// half2 accumulators (4 independent fp16 chains of 16 adds each per 64-elem
// chunk), unpacked to fp32 and summed into sc every 8 c-iterations (=64
// elements). Op count per c: 4 HFMA2 + 1 LDS.128 + 1 LDG.128 (16B) vs the
// fp32 version's 8 cvt + 8 FFMA + 2 LDG.128. NUMERICS: Tier-2 class vs the
// fp32-q dot (q carries one fp16 rounding ~5e-4 rel; chunk sums carry fp16
// accumulation); Tier-1 (spec == T=1 with the same kernels) unaffected.
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

#define SM_K0   0
#define SM_V0   (TILE*512)
#define SM_BYTES (TILE*1024 + 3072)
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

#define STAGE(BASEL, KO, VO) _Pragma("unroll") \
  for (int i = 0; i < TILE/NW; ++i) { \
    const int c = i*NTHR + tid; \
    const int row = c >> 5, c16 = c & 31; \
    const int l = (BASEL) + row; \
    float4 kk = make_float4(0.f, 0.f, 0.f, 0.f), vv = make_float4(0.f, 0.f, 0.f, 0.f); \
    if (l < l1) { \
      const int2 ki = *(const int2*)(Kc8 + (size_t)l*256 + c16*8); \
      const int2 vi = *(const int2*)(Vc8 + (size_t)l*256 + c16*8); \
      kk = dq8(ki, Ksc[(size_t)l*8 + (c16 >> 2)]); \
      vv = dq8(vi, Vsc[(size_t)l*8 + (c16 >> 2)]); \
    } \
    *(float4*)(SM + (KO) + KSZ(row, c16)) = kk; \
    *(float4*)(SM + (VO) + VSZ(row, c16)) = vv; \
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
// accumulators, converted to fp32 and added to A8 at the 32-row tile boundary
// (FA2-style: fp32 precision per tile, half2 mults). NUMERICS: Tier-2 class vs
// the fp32 PV (p carries one fp16 rounding; tile partial sums accumulate fp16).
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

  float msv[RPMAX], ssv[RPMAX], aav[RPMAX][8], prv[RPMAX];
  _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {
    msv[ri] = -1e30f; ssv[ri] = 0.f; prv[ri] = 0.f;
    _Pragma("unroll") for (int j = 0; j < 8; ++j) aav[ri][j] = 0.f;
  }

  if (l0 < l1) {
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    const int kcur = SM_K0, vcur = SM_V0;
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      const int lpl = tile + lane;
      __syncthreads();
      STAGE(tile, SM_K0, SM_V0)
      __syncthreads();
      _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {
        const int row = warp + ri*NW;
        if (row < RMAX) G4_QKROW(row, msv[ri], ssv[ri], aav[ri], prv[ri])
      }
      _Pragma("unroll") for (int ri = 0; ri < RPMAX; ++ri) {
        const int row = warp + ri*NW;
        if (row < RMAX) G4_PVROW(aav[ri], prv[ri])
      }
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
