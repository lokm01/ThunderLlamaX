// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2D-L2: T=4 PROBE trunk kernels (M=4; per-row fp op order IDENTICAL
// to the T=1/T=3 kernels -> rows bit-identical). v2 half2 cores + fat CTAs.
// Laws: flat indexing, sequential loops, no gridDim reads, per-kernel cubins.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define NVH 48
#define CONV_CH 10240
#define QDIM 2048
#define INNER 6144
#define FFN_N 17408
#define VOCAB 248320
#define EPS_N 1e-6f
#define EPS_Q 1e-6f
#define ISQ128 0.08838834764831845f

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x))); }
__device__ __forceinline__ __half hsilu_h4(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

#define LDH2(NM, XB, TS, T, KO) const uint4 NM##_raw = *(const uint4*)((XB) + (T)*(TS) + (KO)); \
  const __half2* NM = (const __half2*)&NM##_raw;

#define RED4(A0,A1,A2,A3) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); } }

// per-acc element order: 2k, 2k+1 ascending == j ascending (m3 ACC3 order)
#define ACC4H2(X0, X1, X2, X3, WV, A0, A1, A2, A3) { \
  const __half2 w01 = __halves2half2(__float2half((WV)[0]), __float2half((WV)[1])); \
  const __half2 w23 = __halves2half2(__float2half((WV)[2]), __float2half((WV)[3])); \
  const __half2 w45 = __halves2half2(__float2half((WV)[4]), __float2half((WV)[5])); \
  const __half2 w67 = __halves2half2(__float2half((WV)[6]), __float2half((WV)[7])); \
  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); \
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
  { const float2 p = __half22float2(__hmul2(x2[3], w67)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x3[0], w01)); A3 += p.x; A3 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x3[1], w23)); A3 += p.x; A3 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x3[2], w45)); A3 += p.x; A3 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x3[3], w67)); A3 += p.x; A3 += p.y; } }

// ---- h_embed4 ----
extern "C" __global__ void __launch_bounds__(256) aq3k8v4(
    const unsigned char* __restrict__ wq, const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv4,
    const float* __restrict__ gridf, const __half* __restrict__ xh4,
    __half* __restrict__ qrow4, __half* __restrict__ krow4, __half* __restrict__ vrow4)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 12288) {
    #define AQI3P4(ROWB, OB, OS, RIDX) { \
      const unsigned char* rowp = (ROWB); \
      const unsigned short* qsp = (const unsigned short*)(rowp); \
      const unsigned int* scp = (const unsigned int*)(rowp + 64*20); \
      const unsigned short* dpp = (const unsigned short*)(rowp + 96*20); \
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f; \
      _Pragma("unroll 5") for (int b = 0; b < 20; ++b) { \
        const float d = __half2float(__ushort_as_half(dpp[b])); \
        const unsigned int sw = scp[8*b + (lane>>2)]; \
        const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
        const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
        const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
        const unsigned int q = qsp[32*b + lane]; \
        const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
        const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
        const int koff = (b << 8) + (lane << 3); \
        LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) \
        const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
        const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
        const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
        const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
        float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
        ACC4H2(xv0, xv1, xv2, xv3, wv, a0, a1, a2, a3) } \
      RED4(a0,a1,a2,a3) \
      if (lane == 0) { (OB)[0*(OS)+(RIDX)] = (__half)a0; (OB)[1*(OS)+(RIDX)] = (__half)a1; (OB)[2*(OS)+(RIDX)] = (__half)a2; (OB)[3*(OS)+(RIDX)] = (__half)a3; } }
    AQI3P4(wq + (size_t)warp * 1960u, qrow4, 12288, warp)
    #undef AQI3P4
  } else if (warp < 13312) {
    #define AKI3P4 AQI3P4_UNUSED
    const int r = warp - 12288;
    #define AKI3(ROWB) { \
      const unsigned char* rowp = (ROWB); \
      const unsigned short* qsp = (const unsigned short*)(rowp); \
      const unsigned int* scp = (const unsigned int*)(rowp + 64*20); \
      const unsigned short* dpp = (const unsigned short*)(rowp + 96*20); \
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f; \
      _Pragma("unroll 5") for (int b = 0; b < 20; ++b) { \
        const float d = __half2float(__ushort_as_half(dpp[b])); \
        const unsigned int sw = scp[8*b + (lane>>2)]; \
        const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
        const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
        const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
        const unsigned int q = qsp[32*b + lane]; \
        const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
        const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
        const int koff = (b << 8) + (lane << 3); \
        LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) \
        const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
        const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
        const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
        const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
        float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
        ACC4H2(xv0, xv1, xv2, xv3, wv, a0, a1, a2, a3) } \
      RED4(a0,a1,a2,a3) \
      if (lane == 0) { krow4[0*1024+r] = (__half)a0; krow4[1*1024+r] = (__half)a1; krow4[2*1024+r] = (__half)a2; krow4[3*1024+r] = (__half)a3; } }
    AKI3(wk + (size_t)r * 1960u)
    #undef AKI3
  } else {
    const int r = warp - 13312;
    const unsigned char* rowp = wv4 + (size_t)r * 2880u;
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
    _Pragma("unroll 5") for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + b*144;
      const float d = __half2float(*((const __half*)blk));
      const float dm = __half2float(*((const __half*)(blk+2)));
      const int s = lane >> 2;
      float sc, mn;
      if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
      else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4)); mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
      const unsigned long long qs8 = *(const unsigned long long*)(blk + 16 + ((lane >> 3) << 5) + ((lane & 3) << 3));
      const int nsh = ((lane >> 2) & 1) << 2;
      const int koff = (b << 8) + (lane << 3);
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff)
      float wv[8];
      _Pragma("unroll") for (int j = 0; j < 8; ++j) {
        const float qv = (float)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu);
        wv[j] = d*sc*qv - dm*mn; }
      ACC4H2(xv0, xv1, xv2, xv3, wv, a0, a1, a2, a3)
    }
    RED4(a0,a1,a2,a3)
    if (lane == 0) { vrow4[0*1024+r] = (__half)a0; vrow4[1*1024+r] = (__half)a1; vrow4[2*1024+r] = (__half)a2; vrow4[3*1024+r] = (__half)a3; }
  }
}

// ---- head8v4 ----
