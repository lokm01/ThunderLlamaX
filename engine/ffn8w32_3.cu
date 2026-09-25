// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2D-L1v2: half2-packed M=3 GEMV cores (old weight layout). Per-element
// fp contract IDENTICAL to m3.cu: same half values (float2half RN == halves2half2 RN
// per element; hmul2 == hmul elementwise; half22float2 == half2float), same per-acc
// add order (j ascending) -> BIT-IDENTICAL outputs. x loads stay half (no
// float round-trip: half->float->half is identity, values unchanged).
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define FFN_N 17408

#define RED3(A0,A1,A2) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); } }

// load row T k-chunk KO as 4 half2 (one uint4)
#define LDH2(NM, XB, TS, T, KO) const uint4 NM##_raw = *(const uint4*)((XB) + (T)*(TS) + (KO)); \
  const __half2* NM = (const __half2*)&NM##_raw;

// core: weight floats WV[8] -> half2 pairs, x rows X{0,1,2} as half2[4], acc A{0,1,2}
// order per acc: elements 2k, 2k+1 ascending == j ascending (m3.cu ACC3 order)
#define ACC3H2(X0, X1, X2, WV, A0, A1, A2) { \
  const __half2 w01 = __halves2half2(__float2half((WV)[0]), __float2half((WV)[1])); \
  const __half2 w23 = __halves2half2(__float2half((WV)[2]), __float2half((WV)[3])); \
  const __half2 w45 = __halves2half2(__float2half((WV)[4]), __float2half((WV)[5])); \
  const __half2 w67 = __halves2half2(__float2half((WV)[6]), __float2half((WV)[7])); \
  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); \
  { const float2 p = __half22float2(__hmul2(x0[0], w01)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[1], w23)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[2], w45)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[3], w67)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[0], w01)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[1], w23)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[2], w45)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[3], w67)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[0], w01)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[1], w23)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[2], w45)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[3], w67)); A2 += p.x; A2 += p.y; } }

__device__ __forceinline__ __half hsilu_h2(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- ffn8v_3: gate+up IQ3 GEMVs + silu-mul, 3 rows (NB=20, old 1960B layout) ----
extern "C" __global__ void __launch_bounds__(1024) ffn8w32_3(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const float* __restrict__ gridf, const __half* __restrict__ hhx3, __half* __restrict__ gact3)
{
  const int warp = (blockIdx.x << 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= FFN_N) return;
  const unsigned short* qg = (const unsigned short*)(wg + (size_t)warp * 1960u);
  const unsigned int* sg = (const unsigned int*)(wg + (size_t)warp * 1960u + 1280u);
  const unsigned short* dg = (const unsigned short*)(wg + (size_t)warp * 1960u + 1920u);
  const unsigned short* qu = (const unsigned short*)(wu + (size_t)warp * 1960u);
  const unsigned int* su = (const unsigned int*)(wu + (size_t)warp * 1960u + 1280u);
  const unsigned short* du = (const unsigned short*)(wu + (size_t)warp * 1960u + 1920u);
  float ag0=0.f,ag1=0.f,ag2=0.f, au0=0.f,au1=0.f,au2=0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    LDH2(xg0, hhx3, DIM, 0, koff) LDH2(xg1, hhx3, DIM, 1, koff) LDH2(xg2, hhx3, DIM, 2, koff)
    #define IQ3W(QP, SP, DP, A0, A1, A2) { \
      const float d = __half2float(__ushort_as_half((DP)[b])); \
      const unsigned int sw = (SP)[8*b + (lane>>2)]; \
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
      const unsigned int q = (QP)[32*b + lane]; \
      const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
      const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
      float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, \
                      db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
      ACC3H2(xg0, xg1, xg2, wv, A0, A1, A2) }
    IQ3W(qg, sg, dg, ag0, ag1, ag2)
    IQ3W(qu, su, du, au0, au1, au2)
    #undef IQ3W
  }
  RED3(ag0,ag1,ag2)
  RED3(au0,au1,au2)
  if (lane == 0) {
    gact3[0*FFN_N + warp] = __hmul(hsilu_h2((__half)ag0), (__half)au0);
    gact3[1*FFN_N + warp] = __hmul(hsilu_h2((__half)ag1), (__half)au1);
    gact3[2*FFN_N + warp] = __hmul(hsilu_h2((__half)ag2), (__half)au2);
  }
}

// ---- down8w32_3: fat-CTA down GEMV + residual ----
