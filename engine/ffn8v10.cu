// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 R8-K9: T=10 PROBE trunk kernels (M=10; k2s10 slots 0..4 + rec6x t=5..rec10x t=9; conv slots 0..3 + conv5x..conv10x; [48][5] layout + live=slot-4 PRESERVED); k2s6 slots 0..4 + rec6x t=5; conv slots 0..3 + conv5x t=4 + conv6x t=5; [48][5] layout + live=slot-4 PRESERVED); k2s5 writes per-step slots 0..4 (slot 4 = live, sequential in-CTA); per-row fp op order IDENTICAL
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

#define RED10(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); A5 += __shfl_down_sync(FULL, A5, o); A6 += __shfl_down_sync(FULL, A6, o); A7 += __shfl_down_sync(FULL, A7, o); A8 += __shfl_down_sync(FULL, A8, o); A9 += __shfl_down_sync(FULL, A9, o); } }

// per-acc element order: 2k, 2k+1 ascending == j ascending (m3 ACC3 order)
#define ACC10H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) { \
  const __half2 w01 = __halves2half2(__float2half((WV)[0]), __float2half((WV)[1])); \
  const __half2 w23 = __halves2half2(__float2half((WV)[2]), __float2half((WV)[3])); \
  const __half2 w45 = __halves2half2(__float2half((WV)[4]), __float2half((WV)[5])); \
  const __half2 w67 = __halves2half2(__float2half((WV)[6]), __float2half((WV)[7])); \
  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); const __half2* x4 = (X4); const __half2* x5 = (X5); const __half2* x6 = (X6); const __half2* x7 = (X7); const __half2* x8 = (X8); const __half2* x9 = (X9); \
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
  { const float2 p = __half22float2(__hmul2(x6[3], w67)); A6 += p.x; A6 += p.y; ; } \
    { const float2 p = __half22float2(__hmul2(x7[0], w01)); A7 += p.x; A7 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x7[1], w23)); A7 += p.x; A7 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x7[2], w45)); A7 += p.x; A7 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x7[3], w67)); A7 += p.x; A7 += p.y; ; } \
    { const float2 p = __half22float2(__hmul2(x8[0], w01)); A8 += p.x; A8 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x8[1], w23)); A8 += p.x; A8 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x8[2], w45)); A8 += p.x; A8 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x8[3], w67)); A8 += p.x; A8 += p.y; ; } \
    { const float2 p = __half22float2(__hmul2(x9[0], w01)); A9 += p.x; A9 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x9[1], w23)); A9 += p.x; A9 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x9[2], w45)); A9 += p.x; A9 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x9[3], w67)); A9 += p.x; A9 += p.y; } }

// ---- h_embed4 ----
// R7a norms rung: one ROW per CTA (t = blockIdx.x, grid 8). Was a serial 8-row
// t-loop on ONE CTA (grid 1) — the D3 norms_emb pool (130 launches, 11.6ms
// deep-cycle). Per-row math VERBATIM (same tid loops) -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) ffn8v10(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const float* __restrict__ gridf, const __half* __restrict__ hhx4, __half* __restrict__ gact4)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= FFN_N) return;
  const unsigned short* qg = (const unsigned short*)(wg + (size_t)warp * 1960u);
  const unsigned int* sg = (const unsigned int*)(wg + (size_t)warp * 1960u + 1280u);
  const unsigned short* dg = (const unsigned short*)(wg + (size_t)warp * 1960u + 1920u);
  const unsigned short* qu = (const unsigned short*)(wu + (size_t)warp * 1960u);
  const unsigned int* su = (const unsigned int*)(wu + (size_t)warp * 1960u + 1280u);
  const unsigned short* du = (const unsigned short*)(wu + (size_t)warp * 1960u + 1920u);
  float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f,ag9=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f,au9=0.f;
  #pragma unroll 2  // R8: zero-spill at M=10
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    LDH2(xg0, hhx4, DIM, 0, koff) LDH2(xg1, hhx4, DIM, 1, koff) LDH2(xg2, hhx4, DIM, 2, koff) LDH2(xg3, hhx4, DIM, 3, koff) LDH2(xg4, hhx4, DIM, 4, koff) LDH2(xg5, hhx4, DIM, 5, koff) LDH2(xg6, hhx4, DIM, 6, koff) LDH2(xg7, hhx4, DIM, 7, koff) LDH2(xg8, hhx4, DIM, 8, koff) LDH2(xg9, hhx4, DIM, 9, koff)
    #define IQ3V10(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) { \
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
      float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
      ACC10H2(xg0, xg1, xg2, xg3, xg4, xg5, xg6, xg7, xg8, xg9, wv, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) }
    IQ3V10(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9)
    IQ3V10(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7, au8, au9)
    #undef IQ3V4
  }
  RED10(ag0,ag1,ag2,ag3,ag4,ag5,ag6,ag7,ag8,ag9)
  RED10(au0,au1,au2,au3,au4,au5,au6,au7,au8,au9)
  if (lane == 0) {
    gact4[0*FFN_N + warp] = __hmul(hsilu_h4((__half)ag0), (__half)au0);
    gact4[1*FFN_N + warp] = __hmul(hsilu_h4((__half)ag1), (__half)au1);
    gact4[2*FFN_N + warp] = __hmul(hsilu_h4((__half)ag2), (__half)au2);
    gact4[3*FFN_N + warp] = __hmul(hsilu_h4((__half)ag3), (__half)au3); gact4[4*FFN_N + warp] = __hmul(hsilu_h4((__half)ag4), (__half)au4); gact4[5*FFN_N + warp] = __hmul(hsilu_h4((__half)ag5), (__half)au5); gact4[6*FFN_N + warp] = __hmul(hsilu_h4((__half)ag6), (__half)au6); gact4[7*FFN_N + warp] = __hmul(hsilu_h4((__half)ag7), (__half)au7); gact4[8*FFN_N + warp] = __hmul(hsilu_h4((__half)ag8), (__half)au8); gact4[9*FFN_N + warp] = __hmul(hsilu_h4((__half)ag9), (__half)au9);
  }
}

// ---- down8nw32_4 ----
