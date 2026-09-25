// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2E KV8: int8-KV variant of spk_g4.cu (fat-CTA split-KV attention K1).
// Storage-only quantization: kv becomes signed char [2][4][CTXK][256] with
// per-(row, 32-channel) fp16 scales sc [2][4][CTXK][8]. STAGE loads int8
// (8B/thread/row, coalesced) + one fp16 scale per thread (4 threads share a
// group scale -> L1 broadcast), dequantizes in registers to the SAME fp16
// smem layout as spk_g4.cu -> downstream QK/PV math byte-for-byte identical
// code, values carry storage-quant error only. Numerics contract: Tier-2
// (greedy sequence may differ from fp16-KV; Tier-1 = spec == T=1 with the
// SAME int8 kernels in both paths). LAWS: single __shared__ array, 16B-aligned
// compile-time offsets, no blockDim/gridDim reads, full-mask shuffles,
// sequential loops, TILE=32, int2 loads are 8B-aligned (c16*8).
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

// ---------------- smem layout (bytes, compile-time) ----------------
#define SM_K0   0
#define SM_V0   (TILE*512)
#define SM_BYTES (TILE*1024)
#define KSZ(R, C)  (((R) * 512) + (((C) ^ ((R) & 7)) * 16))   // K swizzled 16B slot
#define VSZ(R, C)  (((R) * 512) + ((C) * 16))                 // V plain 16B slot

#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )

// BIASED-uint8 KV (u = q + 128, u in [1,255]). The 0x6400 trick:
// half(0x6400 | u) = 1024 + u EXACTLY (exp-10 step-1 mantissa), so one PRMT
// builds a half2 pair {1024+u0, 1024+u1} straight from the bytes; HSUB2 by
// 1152 gives {u0-128, u1-128} = q (exact, Sterbenz); HMUL2 by the group scale
// = the single rounding (bit-identical to float-mul-then-f2h). 3 ops / 2 vals.
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

// ---------------- STAGE: int8 KV tile -> fp16 smem, coalesced ----------------
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

#define G4_QKROW(R, Mv, Sv, A8, PR) { \
  const int t_ = (R) % ROWS; \
  const bool vld = (lpl < l1) && (lpl <= pos + t_); \
  float sc = 0.f; \
  const float* qb = qw + (size_t)QROWIDX(R)*256; \
  _Pragma("unroll") \
  for (int c = 0; c < 32; ++c) { \
    const float4 kf4 = *(const float4*)(SM + kcur + KSZ(lane, c)); \
    const __half2* hk = (const __half2*)&kf4; \
    const float2 k0 = __half22float2(hk[0]), k1 = __half22float2(hk[1]), k2 = __half22float2(hk[2]), k3 = __half22float2(hk[3]); \
    const float4 qa = *(const float4*)(qb + c*8); \
    const float4 qcx = *(const float4*)(qb + c*8 + 4); \
    const float qq[8] = {qa.x, qa.y, qa.z, qa.w, qcx.x, qcx.y, qcx.z, qcx.w}; \
    const float kk[8] = {k0.x, k0.y, k1.x, k1.y, k2.x, k2.y, k3.x, k3.y}; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) sc += qq[j] * kk[j]; \
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
    const float4 vf4 = *(const float4*)(SM + vcur + VSZ(i2, lane)); \
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

  // ---- partial writes: IDENTICAL to spk_g4 (empty split -> identity partials) ----
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
