// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 R5-K6: T=7 PROBE trunk kernels (M=7; k2s7 slots 0..4 + rec6x t=5 + rec7x t=6; conv slots 0..3 + conv5x/6x/7x; [48][5] layout + live=slot-4 PRESERVED); k2s6 slots 0..4 + rec6x t=5; conv slots 0..3 + conv5x t=4 + conv6x t=5; [48][5] layout + live=slot-4 PRESERVED); k2s5 writes per-step slots 0..4 (slot 4 = live, sequential in-CTA); per-row fp op order IDENTICAL
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

#define RED7(A0,A1,A2,A3,A4,A5,A6) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); A5 += __shfl_down_sync(FULL, A5, o); A6 += __shfl_down_sync(FULL, A6, o); } }

// per-acc element order: 2k, 2k+1 ascending == j ascending (m3 ACC3 order)
#define ACC7H2(X0, X1, X2, X3, X4, X5, X6, WV, A0, A1, A2, A3, A4, A5, A6) { \
  const __half2 w01 = __halves2half2(__float2half((WV)[0]), __float2half((WV)[1])); \
  const __half2 w23 = __halves2half2(__float2half((WV)[2]), __float2half((WV)[3])); \
  const __half2 w45 = __halves2half2(__float2half((WV)[4]), __float2half((WV)[5])); \
  const __half2 w67 = __halves2half2(__float2half((WV)[6]), __float2half((WV)[7])); \
  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); const __half2* x4 = (X4); const __half2* x5 = (X5); const __half2* x6 = (X6); \
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
  { const float2 p = __half22float2(__hmul2(x3[3], w67)); A3 += p.x; A3 += p.y; } \
    { const float2 p = __half22float2(__hmul2(x4[0], w01)); A4 += p.x; A4 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x4[1], w23)); A4 += p.x; A4 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x4[2], w45)); A4 += p.x; A4 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x4[3], w67)); A4 += p.x; A4 += p.y; ; } \
    { const float2 p = __half22float2(__hmul2(x5[0], w01)); A5 += p.x; A5 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x5[1], w23)); A5 += p.x; A5 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x5[2], w45)); A5 += p.x; A5 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x5[3], w67)); A5 += p.x; A5 += p.y; ; } \
    { const float2 p = __half22float2(__hmul2(x6[0], w01)); A6 += p.x; A6 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x6[1], w23)); A6 += p.x; A6 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x6[2], w45)); A6 += p.x; A6 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x6[3], w67)); A6 += p.x; A6 += p.y; } }

// ---- h_embed4 ----
extern "C" __global__ void __launch_bounds__(256) q5g8v7(
    const unsigned char* __restrict__ wq5, const unsigned char* __restrict__ wq3g,
    const float* __restrict__ gridf, const __half* __restrict__ xh4,
    __half* __restrict__ qkv4, __half* __restrict__ gate4)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 10240) {
    const unsigned char* rowp = wq5 + (size_t)(warp) * 3520u;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f;
    _Pragma("unroll 5") for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + b*176;
      const float d = __half2float(*((const __half*)blk));
      const float dm = __half2float(*((const __half*)(blk+2)));
      const int s = lane >> 2;
      float sc, mn;
      if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
      else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4)); mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
      const unsigned long long qs8 = *(const unsigned long long*)(blk + 48 + ((lane >> 3) << 5) + ((lane & 3) << 3));
      const unsigned long long qh8 = *(const unsigned long long*)(blk + 16 + ((lane & 3) << 3));
      const int nsh = ((lane >> 2) & 1) << 2;
      const int koff = (b << 8) + (lane << 3);
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff)
      float wv[8];
      _Pragma("unroll") for (int j = 0; j < 8; ++j) {
        int qv = (int)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu);
        qv += (int)((((unsigned int)(qh8 >> (8*j)) >> s) & 1u) << 4);
        wv[j] = d*sc*(float)qv - dm*mn; }
      ACC7H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, wv, a0, a1, a2, a3, a4, a5, a6)
    }
    RED7(a0,a1,a2,a3,a4,a5,a6)
    if (lane == 0) { qkv4[0*10240+warp] = (__half)a0; qkv4[1*10240+warp] = (__half)a1; qkv4[2*10240+warp] = (__half)a2; qkv4[3*10240+warp] = (__half)a3; qkv4[4*10240+warp] = (__half)a4; qkv4[5*10240+warp] = (__half)a5; qkv4[6*10240+warp] = (__half)a6; }
  } else {
    const int r = warp - 10240;
    const unsigned char* rowp = wq3g + (size_t)r * 1960u;
    const unsigned short* qsp = (const unsigned short*)(rowp);
    const unsigned int* scp = (const unsigned int*)(rowp + 64*20);
    const unsigned short* dpp = (const unsigned short*)(rowp + 96*20);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f;
    _Pragma("unroll 5") for (int b = 0; b < 20; ++b) {
      const float d = __half2float(__ushort_as_half(dpp[b]));
      const unsigned int sw = scp[8*b + (lane>>2)];
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu;
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
      const unsigned int q = qsp[32*b + lane];
      const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2)));
      const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2)));
      const int koff = (b << 8) + (lane << 3);
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff)
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
      float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
      ACC7H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, wv, a0, a1, a2, a3, a4, a5, a6)
    }
    RED7(a0,a1,a2,a3,a4,a5,a6)
    if (lane == 0) { gate4[0*6144+r] = (__half)a0; gate4[1*6144+r] = (__half)a1; gate4[2*6144+r] = (__half)a2; gate4[3*6144+r] = (__half)a3; gate4[4*6144+r] = (__half)a4; gate4[5*6144+r] = (__half)a5; gate4[6*6144+r] = (__half)a6; }
  }
}

// ---- k2s4: conv+silu+L2+delta-scan 4 steps, per-step slots 0..3 + live 4 ----
