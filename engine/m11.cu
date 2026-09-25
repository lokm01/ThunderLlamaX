// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 R8-K10: T=11 PROBE trunk kernels (M=11; k2s11 slots 0..4 + rec6x t=5..rec11x t=10; conv slots 0..3 + conv5x..conv11x; [48][5] layout + live=slot-4 PRESERVED); k2s6 slots 0..4 + rec6x t=5; conv slots 0..3 + conv5x t=4 + conv6x t=5; [48][5] layout + live=slot-4 PRESERVED); k2s5 writes per-step slots 0..4 (slot 4 = live, sequential in-CTA); per-row fp op order IDENTICAL
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

#define RED11(A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); A5 += __shfl_down_sync(FULL, A5, o); A6 += __shfl_down_sync(FULL, A6, o); A7 += __shfl_down_sync(FULL, A7, o); A8 += __shfl_down_sync(FULL, A8, o); A9 += __shfl_down_sync(FULL, A9, o); A10 += __shfl_down_sync(FULL, A10, o); } }

// per-acc element order: 2k, 2k+1 ascending == j ascending (m3 ACC3 order)
#define ACC11H2(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, WV, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) { \
  const __half2 w01 = __halves2half2(__float2half((WV)[0]), __float2half((WV)[1])); \
  const __half2 w23 = __halves2half2(__float2half((WV)[2]), __float2half((WV)[3])); \
  const __half2 w45 = __halves2half2(__float2half((WV)[4]), __float2half((WV)[5])); \
  const __half2 w67 = __halves2half2(__float2half((WV)[6]), __float2half((WV)[7])); \
  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); const __half2* x4 = (X4); const __half2* x5 = (X5); const __half2* x6 = (X6); const __half2* x7 = (X7); const __half2* x8 = (X8); const __half2* x9 = (X9); const __half2* x10 = (X10); \
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
  { const float2 p = __half22float2(__hmul2(x9[3], w67)); A9 += p.x; A9 += p.y; ; } \
    { const float2 p = __half22float2(__hmul2(x10[0], w01)); A10 += p.x; A10 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x10[1], w23)); A10 += p.x; A10 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x10[2], w45)); A10 += p.x; A10 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x10[3], w67)); A10 += p.x; A10 += p.y; } }

// ---- h_embed4 ----
// R7a norms rung: one ROW per CTA (t = blockIdx.x, grid 8). Was a serial 8-row
// t-loop on ONE CTA (grid 1) — the D3 norms_emb pool (130 launches, 11.6ms
// deep-cycle). Per-row math VERBATIM (same tid loops) -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) h_embed11(
    const unsigned char* __restrict__ emb, const float* __restrict__ grid512,
    const int* __restrict__ s0, const int* __restrict__ s1, const int* __restrict__ s2, const int* __restrict__ s3, const int* __restrict__ s4, const int* __restrict__ s5, const int* __restrict__ s6, const int* __restrict__ s7, const int* __restrict__ s8, const int* __restrict__ s9, const int* __restrict__ s10,
    float* __restrict__ x4)
{
  const int tid = threadIdx.x;
  const int t = blockIdx.x;
  const int toks[11] = { s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], s6[0], s7[0], s8[0], s9[0], s10[0] };
  {
    const unsigned char* row = emb + (size_t)toks[t] * 2200u;
    float* xo = x4 + t*DIM;
    #pragma unroll
    for (int i = 0; i < 20; ++i) {
      const unsigned char* blk = row + i*110;
      const int e = i*256 + tid;
      const float d = __half2float(*((const __half*)blk));
      const int g = tid >> 2, j4 = tid & 3;
      const unsigned int q = (unsigned int)blk[2 + g] + ((((unsigned int)blk[66 + (g>>3)] >> (g&7)) & 1u) << 8);
      const int s8 = tid >> 5;
      const float sc = 1.0f + 2.0f*(float)((blk[106 + (s8>>1)] >> ((s8&1)<<2)) & 0xF);
      const float sgn = ((blk[74 + (tid>>3)] >> (tid & 7)) & 1) ? -1.f : 1.f;
      xo[e] = d * sc * grid512[(q << 2) + j4] * sgn;
    }
  }
}

// ---- k0n4 ----
// R7a norms rung: one ROW per CTA (t = blockIdx.x, grid 8; was grid 1 with a
// serial 8-row t-loop). Per-row math VERBATIM -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) k0n11(
    const float* __restrict__ x, const float* __restrict__ nw, __half* __restrict__ xh4)
{
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x;
  {
    const float* xr = x + t*DIM;
    float ss = 0.f;
    for (int i = lane; i < DIM; i += 32) { const float v = xr[i]; ss += v*v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    const float r = rsqrtf(ss/DIM + EPS_N);
    __half* xo = xh4 + t*DIM;
    for (int i = threadIdx.x; i < DIM; i += 256) xo[i] = __float2half(xr[i]*r*nw[i]);
  }
}

// ---- k0ab4 ----
// R7a norms rung: (row t, group g) per CTA — grid 104 = 13*8 (was 13 CTAs each
// serially looping all 8 rows: 8 sumsq's per CTA + CTA0 writing 8 xh4 rows +
// CTAs 1-12 doing 96 dots x 8 rows serial). t = blockIdx.x & 7, g = blockIdx.x
// >> 3; w = (g-1)*8 + warpInCTA covers [0,96) exactly once. Every per-row /
// per-dot lane loop, shfl tree and store VERBATIM -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) k0ab11(
    const float* __restrict__ x, const float* __restrict__ nw,
    const float* __restrict__ walpha, const float* __restrict__ wbeta,
    __half* __restrict__ xh4, float* __restrict__ a4, float* __restrict__ b4)
{
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x % 11;
  const int g = blockIdx.x / 11;
  const float* xr = x + t*DIM;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = xr[i]; ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  if (g == 0) {
    __half* xo = xh4 + t*DIM;
    for (int i = threadIdx.x; i < DIM; i += 256) xo[i] = __float2half(xr[i]*r*nw[i]);
  } else {
    const int w = (g - 1) * 8 + (threadIdx.x >> 5);
    const float* wr = (w < 48 ? walpha + (size_t)w*DIM : wbeta + (size_t)(w-48)*DIM);
    float acc = 0.f;
    for (int i = lane; i < DIM; i += 32)
      acc += wr[i] * __half2float(__float2half(xr[i]*r*nw[i]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) { if (w < 48) a4[t*48+w] = acc; else b4[t*48+(w-48)] = acc; }
  }
}

// ---- q5g8v4 ----
extern "C" __global__ void __launch_bounds__(256) q5g8v11(
    const unsigned char* __restrict__ wq5, const unsigned char* __restrict__ wq3g,
    const float* __restrict__ gridf, const __half* __restrict__ xh4,
    __half* __restrict__ qkv4, __half* __restrict__ gate4)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 10240) {
    const unsigned char* rowp = wq5 + (size_t)(warp) * 3520u;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f, a8 = 0.f, a9 = 0.f, a10 = 0.f;
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
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff)
      float wv[8];
      _Pragma("unroll") for (int j = 0; j < 8; ++j) {
        int qv = (int)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu);
        qv += (int)((((unsigned int)(qh8 >> (8*j)) >> s) & 1u) << 4);
        wv[j] = d*sc*(float)qv - dm*mn; }
      ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    }
    RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
    if (lane == 0) { qkv4[0*10240+warp] = (__half)a0; qkv4[1*10240+warp] = (__half)a1; qkv4[2*10240+warp] = (__half)a2; qkv4[3*10240+warp] = (__half)a3; qkv4[4*10240+warp] = (__half)a4; qkv4[5*10240+warp] = (__half)a5; qkv4[6*10240+warp] = (__half)a6; qkv4[7*10240+warp] = (__half)a7; qkv4[8*10240+warp] = (__half)a8; qkv4[9*10240+warp] = (__half)a9; qkv4[10*10240+warp] = (__half)a10; }
  } else {
    const int r = warp - 10240;
    const unsigned char* rowp = wq3g + (size_t)r * 1960u;
    const unsigned short* qsp = (const unsigned short*)(rowp);
    const unsigned int* scp = (const unsigned int*)(rowp + 64*20);
    const unsigned short* dpp = (const unsigned short*)(rowp + 96*20);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f, a8 = 0.f, a9 = 0.f, a10 = 0.f;
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
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff)
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
      float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
      ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    }
    RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
    if (lane == 0) { gate4[0*6144+r] = (__half)a0; gate4[1*6144+r] = (__half)a1; gate4[2*6144+r] = (__half)a2; gate4[3*6144+r] = (__half)a3; gate4[4*6144+r] = (__half)a4; gate4[5*6144+r] = (__half)a5; gate4[6*6144+r] = (__half)a6; gate4[7*6144+r] = (__half)a7; gate4[8*6144+r] = (__half)a8; gate4[9*6144+r] = (__half)a9; gate4[10*6144+r] = (__half)a10; }
  }
}

// ---- k2s4: conv+silu+L2+delta-scan 4 steps, per-step slots 0..3 + live 4 ----
extern "C" __global__ void __launch_bounds__(256) k2s11(
    float* __restrict__ conv_b, float* __restrict__ rec_b, float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ conv7x, float* __restrict__ conv8x, float* __restrict__ conv9x, float* __restrict__ conv10x, float* __restrict__ conv11x, float* __restrict__ rec6x, float* __restrict__ rec7x, float* __restrict__ rec8x, float* __restrict__ rec9x, float* __restrict__ rec10x, float* __restrict__ rec11x,
    const __half* __restrict__ qkv4, const __half* __restrict__ gate4,
    const float* __restrict__ convw, const float* __restrict__ dtb, const float* __restrict__ ssm_a,
    const float* __restrict__ a4, const float* __restrict__ b4,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    float* __restrict__ core, const float* __restrict__ snw, __half* __restrict__ z4)
{
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kh = h % 16;
  const int qc0 = kh*128, kc0 = QDIM + kh*128, vc0 = 2*QDIM + h*128;
  const float* live = conv_b + 4*(3*CONV_CH);
  for (int t = 0; t < 11; ++t) {
    const float al = expf(softplus_f(a4[t*48+h] + dtb[h]) * ssm_a[h]);
    const float be = sig_f(b4[t*48+h]);
    const __half* qrow_t = qkv4 + t*10240;
    float qr[4], kr[4], qss = 0.f, kss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int qq = qc0 + lane*4 + j, kk = kc0 + lane*4 + j, vv = vc0 + lane*4 + j;
      const float q2 = __half2float(qrow_t[qq]), k2 = __half2float(qrow_t[kk]), v2 = __half2float(qrow_t[vv]);
      float sq, sk, sv;
      if (t == 0) {
        sq = live[qq]*convw[qq*4+0] + live[CONV_CH+qq]*convw[qq*4+1] + live[2*CONV_CH+qq]*convw[qq*4+2] + q2*convw[qq*4+3];
        sk = live[kk]*convw[kk*4+0] + live[CONV_CH+kk]*convw[kk*4+1] + live[2*CONV_CH+kk]*convw[kk*4+2] + k2*convw[kk*4+3];
        sv = live[vv]*convw[vv*4+0] + live[CONV_CH+vv]*convw[vv*4+1] + live[2*CONV_CH+vv]*convw[vv*4+2] + v2*convw[vv*4+3];
      } else {
        const float r1q = (t == 1) ? live[2*CONV_CH+qq] : __half2float(qkv4[(t-2)*10240+qq]);
        const float r1k = (t == 1) ? live[2*CONV_CH+kk] : __half2float(qkv4[(t-2)*10240+kk]);
        const float r1v = (t == 1) ? live[2*CONV_CH+vv] : __half2float(qkv4[(t-2)*10240+vv]);
        const float r2q = __half2float(qkv4[(t-1)*10240+qq]);
        const float r2k = __half2float(qkv4[(t-1)*10240+kk]);
        const float r2v = __half2float(qkv4[(t-1)*10240+vv]);
        const float r0q = (t < 3) ? live[t*CONV_CH+qq] : __half2float(qkv4[(t-3)*10240+qq]);
        const float r0k = (t < 3) ? live[t*CONV_CH+kk] : __half2float(qkv4[(t-3)*10240+kk]);
        const float r0v = (t < 3) ? live[t*CONV_CH+vv] : __half2float(qkv4[(t-3)*10240+vv]);
        sq = r0q*convw[qq*4+0] + r1q*convw[qq*4+1] + r2q*convw[qq*4+2] + q2*convw[qq*4+3];
        sk = r0k*convw[kk*4+0] + r1k*convw[kk*4+1] + r2k*convw[kk*4+2] + k2*convw[kk*4+3];
        sv = r0v*convw[vv*4+0] + r1v*convw[vv*4+1] + r2v*convw[vv*4+2] + v2*convw[vv*4+3];
      }
      sq *= sig_f(sq); sk *= sig_f(sk); sv *= sig_f(sv);
      qr[j] = sq; kr[j] = sk; v[h*128 + lane*4 + j] = sv;
      qss += sq*sq; kss += sk*sk;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { qss += __shfl_xor_sync(FULL, qss, o); kss += __shfl_xor_sync(FULL, kss, o); }
    const float qn = (1.0f / fmaxf(sqrtf(qss), EPS_Q)) * ISQ128;
    const float kn = 1.0f / fmaxf(sqrtf(kss), EPS_Q);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[h*128 + lane*4 + j] = qr[j]*qn;
      k[h*128 + lane*4 + j] = kr[j]*kn;
    }
    const float* vbuf = v + h*128;
    // R5 K=6 FIX: t=6's rec input is t=5's state, which lives in rec6x (the
    // [48][5] layout has no slot 5 — the naive (t-1) index read the NEXT block's
    // slot 0 = garbage -> NaN row 6). t=1..5 read slots 0..4 as before.
    const float* rec_in = (t == 0) ? (rec_b + 4*(NVH*128*128) + (size_t)h*128*128)
                                   : (t == 6) ? (rec6x + (size_t)h*128*128)
                                   : (t == 7) ? (rec7x + (size_t)h*128*128)
                                   : (t == 8) ? (rec8x + (size_t)h*128*128)
                                   : (t == 9) ? (rec9x + (size_t)h*128*128)
                                   : (t == 10) ? (rec10x + (size_t)h*128*128)
                                   : (rec_b + (size_t)(t-1)*(NVH*128*128) + (size_t)h*128*128);
    float* rec_out = (t >= 5) ? (((t == 5) ? rec6x : (t == 6) ? rec7x : (t == 7) ? rec8x : (t == 8) ? rec9x : (t == 9) ? rec10x : rec11x) + (size_t)h*128*128)
                                : (rec_b + (size_t)t*(NVH*128*128) + (size_t)h*128*128);
    #pragma unroll
    for (int vv2 = 0; vv2 < 16; ++vv2) {
      const int v_idx = warp*16 + vv2;
      const float* srow_in = rec_in + (size_t)v_idx*128;
      float* srow = rec_out + (size_t)v_idx*128;
      float s0 = srow_in[lane*4]*al, s1 = srow_in[lane*4+1]*al, s2 = srow_in[lane*4+2]*al, s3 = srow_in[lane*4+3]*al;
      const float k0 = kr[0]*kn, k1 = kr[1]*kn, k2 = kr[2]*kn, k3 = kr[3]*kn;
      float kd = s0*k0 + s1*k1 + s2*k2 + s3*k3;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) kd += __shfl_xor_sync(FULL, kd, o);
      const float dl = (vbuf[v_idx] - kd) * be;
      s0 += dl*k0; s1 += dl*k1; s2 += dl*k2; s3 += dl*k3;
      srow[lane*4] = s0; srow[lane*4+1] = s1; srow[lane*4+2] = s2; srow[lane*4+3] = s3;
      const float q0 = qr[0]*qn, q1 = qr[1]*qn, q2 = qr[2]*qn, q3 = qr[3]*qn;
      float qd = s0*q0 + s1*q1 + s2*q2 + s3*q3;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) qd += __shfl_xor_sync(FULL, qd, o);
      if (lane == 0) core[h*128 + v_idx] = qd;
    }
    {
      // R4 RACE FIX: t==4 would write conv slot 4 = the LIVE window other CTAs
      // still read at their t=0..2 (conv writes are cross-head strided — no CTA
      // owns its channels). Route t==4's window to conv5x (unread in-kernel);
      // acceptsel5k copies conv5x -> slot 4 only when m==4. rec slot writes are
      // head-local slices (race-free) and stay in-kernel for all t.
      float* dst = (t == 4) ? conv5x : (t == 5) ? conv6x : (t == 6) ? conv7x : (t == 7) ? conv8x : (t == 8) ? conv9x : (t == 9) ? conv10x : (t == 10) ? conv11x : (conv_b + (size_t)t * (3*CONV_CH));
      const int tid = (h << 8) + threadIdx.x;
      for (int i = tid; i < 3*CONV_CH; i += (NVH << 8)) {
        const int row = i / CONV_CH, c = i - row*CONV_CH;
        float val;
        if (t == 0) val = (row < 2) ? live[(row+1)*CONV_CH + c] : __half2float(qkv4[0*10240 + c]);
        else if (t == 1) val = (row == 0) ? live[2*CONV_CH + c] : __half2float(qkv4[(row-1)*10240 + c]);
        else val = __half2float(qkv4[(row + t - 2)*10240 + c]);
        dst[i] = val;
      }
    }
    __syncthreads();
    {
      float zz = 0.f;
      for (int i = lane; i < 128; i += 32) { const float c = core[h*128+i]; zz += c*c; }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) zz += __shfl_xor_sync(FULL, zz, o);
      const float rz = rsqrtf(zz/128 + EPS_N);
      if (threadIdx.x < 128) {
        const int j = threadIdx.x;
        const __half g = gate4[t*6144 + h*128 + j];
        z4[t*6144 + h*128 + j] = __float2half((core[h*128+j]*rz*snw[j]) * __half2float(
          __hmul(g, hrcp((__half)1.0f + hexp2(__hmul(g, __float2half(-1.4423828125f)))))));
      }
    }
    __syncthreads();
  }
}

// ---- op38nw32_4 ----
extern "C" __global__ void __launch_bounds__(1024) op38nw32_11(
    const unsigned char* __restrict__ wo, const float* __restrict__ gridf,
    const __half* __restrict__ z4, __half* __restrict__ attn_out4)
{
  const int warp = (blockIdx.x << 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned short* qsp = (const unsigned short*)(wo + (size_t)warp * 2352u);
  const unsigned int* scp = (const unsigned int*)(wo + (size_t)warp * 2352u + 64*24);
  const unsigned short* dpp = (const unsigned short*)(wo + (size_t)warp * 2352u + 96*24);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
  #pragma unroll 2  // R8: 64-reg budget at M=11 (fp order per row unchanged) (fp order per row unchanged)
  for (int b = 0; b < 24; ++b) {
    const float d = __half2float(__ushort_as_half(dpp[b]));
    const unsigned int sw = scp[8*b + (lane>>2)];
    const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
    const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu;
    const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
    const unsigned int q = qsp[32*b + lane];
    const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2)));
    const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2)));
    const int koff = (b << 8) + (lane << 3);
    LDH2(xv0, z4, 6144, 0, koff) LDH2(xv1, z4, 6144, 1, koff) LDH2(xv2, z4, 6144, 2, koff) LDH2(xv3, z4, 6144, 3, koff) LDH2(xv4, z4, 6144, 4, koff) LDH2(xv5, z4, 6144, 5, koff) LDH2(xv6, z4, 6144, 6, koff) LDH2(xv7, z4, 6144, 7, koff) LDH2(xv8, z4, 6144, 8, koff) LDH2(xv9, z4, 6144, 9, koff) LDH2(xv10, z4, 6144, 10, koff)
    const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
    const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
    const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
    const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
    float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
    ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
  }
  RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
  if (lane == 0) { attn_out4[0*DIM+warp] = (__half)a0; attn_out4[1*DIM+warp] = (__half)a1; attn_out4[2*DIM+warp] = (__half)a2; attn_out4[3*DIM+warp] = (__half)a3; attn_out4[4*DIM+warp] = (__half)a4; attn_out4[5*DIM+warp] = (__half)a5; attn_out4[6*DIM+warp] = (__half)a6; attn_out4[7*DIM+warp] = (__half)a7; attn_out4[8*DIM+warp] = (__half)a8; attn_out4[9*DIM+warp] = (__half)a9; attn_out4[10*DIM+warp] = (__half)a10; }
}

// ---- k3aonw32_4 (Q8_0) ----
extern "C" __global__ void __launch_bounds__(1024) k3aonw32_11(
    const unsigned char* __restrict__ wq8, const __half* __restrict__ z4, __half* __restrict__ attn_out4)
{
  const int warp = (blockIdx.x << 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wq8 + (size_t)warp * 6528u;
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
  #pragma unroll 4
  for (int b = 0; b < 192; ++b) {
    const unsigned char* blk = rowp + b*34;
    const float d = __half2float(*((const __half*)blk));
    const float w = d * (float)((signed char)blk[2+lane]);
    const __half wh = __float2half(w);
    a0 += __half2float(__hmul(z4[0*6144 + (b<<5)+lane], wh));
    a1 += __half2float(__hmul(z4[1*6144 + (b<<5)+lane], wh));
    a2 += __half2float(__hmul(z4[2*6144 + (b<<5)+lane], wh));
    a3 += __half2float(__hmul(z4[3*6144 + (b<<5)+lane], wh));
    a4 += __half2float(__hmul(z4[4*6144 + (b<<5)+lane], wh));
    a5 += __half2float(__hmul(z4[5*6144 + (b<<5)+lane], wh));
    a6 += __half2float(__hmul(z4[6*6144 + (b<<5)+lane], wh));
    a7 += __half2float(__hmul(z4[7*6144 + (b<<5)+lane], wh));
    a8 += __half2float(__hmul(z4[8*6144 + (b<<5)+lane], wh));
    a9 += __half2float(__hmul(z4[9*6144 + (b<<5)+lane], wh));
    a10 += __half2float(__hmul(z4[10*6144 + (b<<5)+lane], wh));
  }
  RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
  if (lane == 0) { attn_out4[0*DIM+warp] = (__half)a0; attn_out4[1*DIM+warp] = (__half)a1; attn_out4[2*DIM+warp] = (__half)a2; attn_out4[3*DIM+warp] = (__half)a3; attn_out4[4*DIM+warp] = (__half)a4; attn_out4[5*DIM+warp] = (__half)a5; attn_out4[6*DIM+warp] = (__half)a6; attn_out4[7*DIM+warp] = (__half)a7; attn_out4[8*DIM+warp] = (__half)a8; attn_out4[9*DIM+warp] = (__half)a9; attn_out4[10*DIM+warp] = (__half)a10; }
}

// ---- ao8nw32_4 (IQ3_S raw) ----
extern "C" __global__ void __launch_bounds__(1024) ao8nw32_11(
    const unsigned char* __restrict__ wo, const float* __restrict__ grid512,
    const __half* __restrict__ ao_in4, __half* __restrict__ attn_out4)
{
  const int warp = (blockIdx.x << 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wo + (size_t)warp * 2640u;
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
  #pragma unroll 2  // R8: 64-reg budget at M=11 (fp order per row unchanged)
  for (int b = 0; b < 24; ++b) {
    const unsigned char* blk = rowp + (size_t)b * 110u;
    const float d = __half2float(*((const __half*)blk));
    const int g0i = lane*2, g1i = lane*2 + 1;
    const int sraw = lane >> 2;
    const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
    const unsigned char* sgnb = blk + 74 + lane;
    const int koff = (b << 8) + (lane << 3);
    LDH2(xv0, ao_in4, 6144, 0, koff) LDH2(xv1, ao_in4, 6144, 1, koff) LDH2(xv2, ao_in4, 6144, 2, koff) LDH2(xv3, ao_in4, 6144, 3, koff) LDH2(xv4, ao_in4, 6144, 4, koff) LDH2(xv5, ao_in4, 6144, 5, koff) LDH2(xv6, ao_in4, 6144, 6, koff) LDH2(xv7, ao_in4, 6144, 7, koff) LDH2(xv8, ao_in4, 6144, 8, koff) LDH2(xv9, ao_in4, 6144, 9, koff) LDH2(xv10, ao_in4, 6144, 10, koff)
    const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lane);
    const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0i>>3)] >> (g0i&7)) & 1u) << 8));
    const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1i>>3)] >> (g1i&7)) & 1u) << 8));
    float wv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int g = (j < 4) ? qb0 : qb1;
      const float gv = grid512[(g << 2) + (j & 3)];
      const float sgn = ((*sgnb >> j) & 1) ? -1.f : 1.f;
      wv[j] = d * sc * gv * sgn;
    }
    ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
  }
  RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
  if (lane == 0) { attn_out4[0*DIM+warp] = (__half)a0; attn_out4[1*DIM+warp] = (__half)a1; attn_out4[2*DIM+warp] = (__half)a2; attn_out4[3*DIM+warp] = (__half)a3; attn_out4[4*DIM+warp] = (__half)a4; attn_out4[5*DIM+warp] = (__half)a5; attn_out4[6*DIM+warp] = (__half)a6; attn_out4[7*DIM+warp] = (__half)a7; attn_out4[8*DIM+warp] = (__half)a8; attn_out4[9*DIM+warp] = (__half)a9; attn_out4[10*DIM+warp] = (__half)a10; }
}

// ---- hh4 ----
// R7a norms rung: one ROW per CTA (t = blockIdx.x, grid 8; was grid 1 with a
// serial 8-row t-loop). Per-row math VERBATIM -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) hh11(
    const float* __restrict__ x, const __half* __restrict__ attn_out4, const float* __restrict__ nw2,
    float* __restrict__ hh4b, __half* __restrict__ hhx4)
{
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x;
  const float* xr = x + t*DIM;
  const __half* ar = attn_out4 + t*DIM;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = xr[i] + __half2float(ar[i]); ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  float* ho = hh4b + t*DIM; __half* hx = hhx4 + t*DIM;
  for (int i = threadIdx.x; i < DIM; i += 256) {
    const float v = xr[i] + __half2float(ar[i]);
    ho[i] = v;
    hx[i] = __float2half(v*r*nw2[i]);
  }
}

// ---- ffn8v4 ----
extern "C" __global__ void __launch_bounds__(256) ffn8v11(
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
  float ag0=0.f,ag1=0.f,ag2=0.f,ag3=0.f,ag4=0.f,ag5=0.f,ag6=0.f,ag7=0.f,ag8=0.f,ag9=0.f,ag10=0.f, au0=0.f,au1=0.f,au2=0.f,au3=0.f,au4=0.f,au5=0.f,au6=0.f,au7=0.f,au8=0.f,au9=0.f,au10=0.f;
  #pragma unroll 2  // R8: zero-spill at M=11
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    LDH2(xg0, hhx4, DIM, 0, koff) LDH2(xg1, hhx4, DIM, 1, koff) LDH2(xg2, hhx4, DIM, 2, koff) LDH2(xg3, hhx4, DIM, 3, koff) LDH2(xg4, hhx4, DIM, 4, koff) LDH2(xg5, hhx4, DIM, 5, koff) LDH2(xg6, hhx4, DIM, 6, koff) LDH2(xg7, hhx4, DIM, 7, koff) LDH2(xg8, hhx4, DIM, 8, koff) LDH2(xg9, hhx4, DIM, 9, koff) LDH2(xg10, hhx4, DIM, 10, koff)
    #define IQ3V11(QP, SP, DP, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) { \
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
      ACC11H2(xg0, xg1, xg2, xg3, xg4, xg5, xg6, xg7, xg8, xg9, xg10, wv, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) }
    IQ3V11(qg, sg, dg, ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9, ag10)
    IQ3V11(qu, su, du, au0, au1, au2, au3, au4, au5, au6, au7, au8, au9, au10)
    #undef IQ3V4
  }
  RED11(ag0,ag1,ag2,ag3,ag4,ag5,ag6,ag7,ag8,ag9,ag10)
  RED11(au0,au1,au2,au3,au4,au5,au6,au7,au8,au9,au10)
  if (lane == 0) {
    gact4[0*FFN_N + warp] = __hmul(hsilu_h4((__half)ag0), (__half)au0);
    gact4[1*FFN_N + warp] = __hmul(hsilu_h4((__half)ag1), (__half)au1);
    gact4[2*FFN_N + warp] = __hmul(hsilu_h4((__half)ag2), (__half)au2);
    gact4[3*FFN_N + warp] = __hmul(hsilu_h4((__half)ag3), (__half)au3); gact4[4*FFN_N + warp] = __hmul(hsilu_h4((__half)ag4), (__half)au4); gact4[5*FFN_N + warp] = __hmul(hsilu_h4((__half)ag5), (__half)au5); gact4[6*FFN_N + warp] = __hmul(hsilu_h4((__half)ag6), (__half)au6); gact4[7*FFN_N + warp] = __hmul(hsilu_h4((__half)ag7), (__half)au7); gact4[8*FFN_N + warp] = __hmul(hsilu_h4((__half)ag8), (__half)au8); gact4[9*FFN_N + warp] = __hmul(hsilu_h4((__half)ag9), (__half)au9); gact4[10*FFN_N + warp] = __hmul(hsilu_h4((__half)ag10), (__half)au10);
  }
}

// ---- down8nw32_4 ----
extern "C" __global__ void __launch_bounds__(1024) down8nw32_11(
    const unsigned char* __restrict__ wd, const float* __restrict__ gridf,
    const __half* __restrict__ gact4, const float* __restrict__ hh4b, float* __restrict__ y4)
{
  const int warp = (blockIdx.x << 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wd + (size_t)warp * 6664u;
  const unsigned short* qsp = (const unsigned short*)(rowp);
  const unsigned int* scp = (const unsigned int*)(rowp + 64*68);
  const unsigned short* dpp = (const unsigned short*)(rowp + 96*68);
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
  #pragma unroll 2  // R8: 64-reg budget at M=11 (fp order per row unchanged)
  for (int b = 0; b < 68; ++b) {
    const float d = __half2float(__ushort_as_half(dpp[b]));
    const unsigned int sw = scp[8*b + (lane>>2)];
    const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
    const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu;
    const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
    const unsigned int q = qsp[32*b + lane];
    const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2)));
    const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2)));
    const int koff = (b << 8) + (lane << 3);
    LDH2(xv0, gact4, FFN_N, 0, koff) LDH2(xv1, gact4, FFN_N, 1, koff) LDH2(xv2, gact4, FFN_N, 2, koff) LDH2(xv3, gact4, FFN_N, 3, koff) LDH2(xv4, gact4, FFN_N, 4, koff) LDH2(xv5, gact4, FFN_N, 5, koff) LDH2(xv6, gact4, FFN_N, 6, koff) LDH2(xv7, gact4, FFN_N, 7, koff) LDH2(xv8, gact4, FFN_N, 8, koff) LDH2(xv9, gact4, FFN_N, 9, koff) LDH2(xv10, gact4, FFN_N, 10, koff)
    const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
    const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
    const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
    const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
    float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
    ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
  }
  RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
  if (lane == 0) {
    y4[0*DIM+warp] = hh4b[0*DIM+warp] + (float)((__half)a0);
    y4[1*DIM+warp] = hh4b[1*DIM+warp] + (float)((__half)a1);
    y4[2*DIM+warp] = hh4b[2*DIM+warp] + (float)((__half)a2);
    y4[3*DIM+warp] = hh4b[3*DIM+warp] + (float)((__half)a3); y4[4*DIM+warp] = hh4b[4*DIM+warp] + (float)((__half)a4); y4[5*DIM+warp] = hh4b[5*DIM+warp] + (float)((__half)a5); y4[6*DIM+warp] = hh4b[6*DIM+warp] + (float)((__half)a6); y4[7*DIM+warp] = hh4b[7*DIM+warp] + (float)((__half)a7); y4[8*DIM+warp] = hh4b[8*DIM+warp] + (float)((__half)a8); y4[9*DIM+warp] = hh4b[9*DIM+warp] + (float)((__half)a9); y4[10*DIM+warp] = hh4b[10*DIM+warp] + (float)((__half)a10);
  }
}

// ---- aq3k8v4 (q IQ3 packed + k IQ3 packed + v Q4_K raw) ----
extern "C" __global__ void __launch_bounds__(256) aq3k8v11(
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
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f; \
      _Pragma("unroll 2") for (int b = 0; b < 20; ++b) { \
        const float d = __half2float(__ushort_as_half(dpp[b])); \
        const unsigned int sw = scp[8*b + (lane>>2)]; \
        const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
        const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
        const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
        const unsigned int q = qsp[32*b + lane]; \
        const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
        const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
        const int koff = (b << 8) + (lane << 3); \
        LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff) \
        const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
        const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
        const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
        const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
        float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
        ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) } \
      RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10) \
      if (lane == 0) { (OB)[0*(OS)+(RIDX)] = (__half)a0; (OB)[1*(OS)+(RIDX)] = (__half)a1; (OB)[2*(OS)+(RIDX)] = (__half)a2; (OB)[3*(OS)+(RIDX)] = (__half)a3; (OB)[4*(OS)+(RIDX)] = (__half)a4; (OB)[5*(OS)+(RIDX)] = (__half)a5; (OB)[6*(OS)+(RIDX)] = (__half)a6; (OB)[7*(OS)+(RIDX)] = (__half)a7; (OB)[8*(OS)+(RIDX)] = (__half)a8; (OB)[9*(OS)+(RIDX)] = (__half)a9; (OB)[10*(OS)+(RIDX)] = (__half)a10; } }  // R5 fix: the row-4 store was MISSING (gen_m5 slip) -> qrow3 row 4 stale -> wrong amds[4]
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
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f; \
      _Pragma("unroll 2") for (int b = 0; b < 20; ++b) { \
        const float d = __half2float(__ushort_as_half(dpp[b])); \
        const unsigned int sw = scp[8*b + (lane>>2)]; \
        const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
        const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
        const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
        const unsigned int q = qsp[32*b + lane]; \
        const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
        const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
        const int koff = (b << 8) + (lane << 3); \
        LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff) \
        const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
        const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
        const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
        const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
        float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
        ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) } \
      RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10) \
      if (lane == 0) { krow4[0*1024+r] = (__half)a0; krow4[1*1024+r] = (__half)a1; krow4[2*1024+r] = (__half)a2; krow4[3*1024+r] = (__half)a3; krow4[4*1024+r] = (__half)a4; krow4[5*1024+r] = (__half)a5; krow4[6*1024+r] = (__half)a6; krow4[7*1024+r] = (__half)a7; krow4[8*1024+r] = (__half)a8; krow4[9*1024+r] = (__half)a9; krow4[10*1024+r] = (__half)a10; } }
    AKI3(wk + (size_t)r * 1960u)
    #undef AKI3
  } else {
    const int r = warp - 13312;
    const unsigned char* rowp = wv4 + (size_t)r * 2880u;
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
    _Pragma("unroll 2") for (int b = 0; b < 20; ++b) {
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
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff)
      float wv[8];
      _Pragma("unroll") for (int j = 0; j < 8; ++j) {
        const float qv = (float)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu);
        wv[j] = d*sc*qv - dm*mn; }
      ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    }
    RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
    if (lane == 0) { vrow4[0*1024+r] = (__half)a0; vrow4[1*1024+r] = (__half)a1; vrow4[2*1024+r] = (__half)a2; vrow4[3*1024+r] = (__half)a3; vrow4[4*1024+r] = (__half)a4; vrow4[5*1024+r] = (__half)a5; vrow4[6*1024+r] = (__half)a6; vrow4[7*1024+r] = (__half)a7; vrow4[8*1024+r] = (__half)a8; vrow4[9*1024+r] = (__half)a9; vrow4[10*1024+r] = (__half)a10; }
  }
}

// ---- head8v4 ----
extern "C" __global__ void __launch_bounds__(256) head8v11(
    const unsigned char* __restrict__ wq5, const __half* __restrict__ xh4, __half* __restrict__ logits4)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= VOCAB) return;
  const unsigned char* rowp = wq5 + (size_t)warp * 3520u;
  float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
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
    LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff)
    float wv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int qv = (int)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu);
      qv += (int)((((unsigned int)(qh8 >> (8*j)) >> s) & 1u) << 4);
      wv[j] = d*sc*(float)qv - dm*mn;
    }
    ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
  }
  RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
  if (lane == 0) { logits4[0*VOCAB+warp] = (__half)a0; logits4[1*VOCAB+warp] = (__half)a1; logits4[2*VOCAB+warp] = (__half)a2; logits4[3*VOCAB+warp] = (__half)a3; logits4[4*VOCAB+warp] = (__half)a4; logits4[5*VOCAB+warp] = (__half)a5; logits4[6*VOCAB+warp] = (__half)a6; logits4[7*VOCAB+warp] = (__half)a7; logits4[8*VOCAB+warp] = (__half)a8; logits4[9*VOCAB+warp] = (__half)a9; logits4[10*VOCAB+warp] = (__half)a10; }
}

// ---- aq6k8v4 (q Q6_K packed + k IQ3 packed + v Q4_K raw) ----
extern "C" __global__ void __launch_bounds__(256) aq6k8v11(
    const unsigned char* __restrict__ wq, const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv4,
    const float* __restrict__ gridf, const __half* __restrict__ xh4,
    __half* __restrict__ qrow4, __half* __restrict__ krow4, __half* __restrict__ vrow4)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 12288) {
    const unsigned char* rowp = wq + (size_t)warp * 4240u;
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
    _Pragma("unroll 5") for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + b*212;
      const float d = __half2float(*((const __half*)(blk+2)));
      const bool nib_hi = ((lane&15) >= 8);
      const int c2 = (lane>>2)&3;
      const unsigned int loA = *(const unsigned int*)(blk + 20 + 64*(lane>>4) + (lane&7)*8);
      const unsigned int loB = *(const unsigned int*)(blk + 20 + 64*(lane>>4) + (lane&7)*8 + 4);
      const unsigned int qhA = *(const unsigned int*)(blk + 148 + (lane>>4)*32 + (lane&3)*8);
      const unsigned int qhB = *(const unsigned int*)(blk + 148 + (lane>>4)*32 + (lane&3)*8 + 4);
      const int sc8 = (signed char)blk[4 + (lane>>1)];
      const int koff = (b << 8) + (lane << 3);
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff)
      float wv[8];
      _Pragma("unroll") for (int j = 0; j < 8; ++j) {
        const unsigned int lob = (j < 4) ? (loA >> (8*j)) : (loB >> (8*(j-4)));
        const int xl = nib_hi ? (lob >> 4) & 0xF : lob & 0xF;
        const unsigned int qhb = (j < 4) ? (qhA >> (8*j)) : (qhB >> (8*(j-4)));
        const int xh2 = ((qhb >> (c2<<1)) & 3) << 4;
        wv[j] = d * (float)sc8 * (float)((signed char)((xl | xh2) - 32)); }
      ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    }
    RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
    if (lane == 0) { qrow4[0*12288+warp] = (__half)a0; qrow4[1*12288+warp] = (__half)a1; qrow4[2*12288+warp] = (__half)a2; qrow4[3*12288+warp] = (__half)a3; qrow4[4*12288+warp] = (__half)a4; qrow4[5*12288+warp] = (__half)a5; qrow4[6*12288+warp] = (__half)a6; qrow4[7*12288+warp] = (__half)a7; qrow4[8*12288+warp] = (__half)a8; qrow4[9*12288+warp] = (__half)a9; qrow4[10*12288+warp] = (__half)a10; }
  } else if (warp < 13312) {
    const int r = warp - 12288;
    const unsigned char* rowp = wk + (size_t)r * 1960u;
    const unsigned short* qsp = (const unsigned short*)(rowp);
    const unsigned int* scp = (const unsigned int*)(rowp + 64*20);
    const unsigned short* dpp = (const unsigned short*)(rowp + 96*20);
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
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
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff)
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
      float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
      ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    }
    RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
    if (lane == 0) { krow4[0*1024+r] = (__half)a0; krow4[1*1024+r] = (__half)a1; krow4[2*1024+r] = (__half)a2; krow4[3*1024+r] = (__half)a3; krow4[4*1024+r] = (__half)a4; krow4[5*1024+r] = (__half)a5; krow4[6*1024+r] = (__half)a6; krow4[7*1024+r] = (__half)a7; krow4[8*1024+r] = (__half)a8; krow4[9*1024+r] = (__half)a9; krow4[10*1024+r] = (__half)a10; }
  } else {
    const int r = warp - 13312;
    const unsigned char* rowp = wv4 + (size_t)r * 2880u;
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f,a8=0.f,a9=0.f,a10=0.f;
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
      LDH2(xv0, xh4, DIM, 0, koff) LDH2(xv1, xh4, DIM, 1, koff) LDH2(xv2, xh4, DIM, 2, koff) LDH2(xv3, xh4, DIM, 3, koff) LDH2(xv4, xh4, DIM, 4, koff) LDH2(xv5, xh4, DIM, 5, koff) LDH2(xv6, xh4, DIM, 6, koff) LDH2(xv7, xh4, DIM, 7, koff) LDH2(xv8, xh4, DIM, 8, koff) LDH2(xv9, xh4, DIM, 9, koff) LDH2(xv10, xh4, DIM, 10, koff)
      float wv[8];
      _Pragma("unroll") for (int j = 0; j < 8; ++j) {
        const float qv = (float)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu);
        wv[j] = d*sc*qv - dm*mn; }
      ACC11H2(xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, wv, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
    }
    RED11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)
    if (lane == 0) { vrow4[0*1024+r] = (__half)a0; vrow4[1*1024+r] = (__half)a1; vrow4[2*1024+r] = (__half)a2; vrow4[3*1024+r] = (__half)a3; vrow4[4*1024+r] = (__half)a4; vrow4[5*1024+r] = (__half)a5; vrow4[6*1024+r] = (__half)a6; vrow4[7*1024+r] = (__half)a7; vrow4[8*1024+r] = (__half)a8; vrow4[9*1024+r] = (__half)a9; vrow4[10*1024+r] = (__half)a10; }
  }
}
