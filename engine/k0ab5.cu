// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2D-L2: T=5 PROBE trunk kernels (M=5; k2s5 writes per-step slots 0..4 (slot 4 = live, sequential in-CTA); per-row fp op order IDENTICAL
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

#define RED5(A0,A1,A2,A3,A4) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); } }

// per-acc element order: 2k, 2k+1 ascending == j ascending (m3 ACC3 order)
#define ACC5H2(X0, X1, X2, X3, X4, WV, A0, A1, A2, A3, A4) { \
  const __half2 w01 = __halves2half2(__float2half((WV)[0]), __float2half((WV)[1])); \
  const __half2 w23 = __halves2half2(__float2half((WV)[2]), __float2half((WV)[3])); \
  const __half2 w45 = __halves2half2(__float2half((WV)[4]), __float2half((WV)[5])); \
  const __half2 w67 = __halves2half2(__float2half((WV)[6]), __float2half((WV)[7])); \
  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); const __half2* x4 = (X4); \
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
  { const float2 p = __half22float2(__hmul2(x4[3], w67)); A4 += p.x; A4 += p.y; } }

// ---- h_embed4 ----
extern "C" __global__ void __launch_bounds__(256) k0ab5(
    const float* __restrict__ x, const float* __restrict__ nw,
    const float* __restrict__ walpha, const float* __restrict__ wbeta,
    __half* __restrict__ xh4, float* __restrict__ a4, float* __restrict__ b4)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  float rr[5];
  for (int t = 0; t < 5; ++t) {
    const float* xr = x + t*DIM;
    float ss = 0.f;
    for (int i = lane; i < DIM; i += 32) { const float v = xr[i]; ss += v*v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    rr[t] = rsqrtf(ss/DIM + EPS_N);
  }
  if (blockIdx.x == 0) {
    for (int t = 0; t < 5; ++t) {
      const float* xr = x + t*DIM; __half* xo = xh4 + t*DIM; const float r = rr[t];
      for (int i = threadIdx.x; i < DIM; i += 256) xo[i] = __float2half(xr[i]*r*nw[i]);
    }
  } else {
    const int w = warp - 8;
    const float* wr = (w < 48 ? walpha + (size_t)w*DIM : wbeta + (size_t)(w-48)*DIM);
    for (int t = 0; t < 5; ++t) {
      const float* xr = x + t*DIM; const float r = rr[t];
      float acc = 0.f;
      for (int i = lane; i < DIM; i += 32)
        acc += wr[i] * __half2float(__float2half(xr[i]*r*nw[i]));
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
      if (lane == 0) { if (w < 48) a4[t*48+w] = acc; else b4[t*48+(w-48)] = acc; }
    }
  }
}

// ---- q5g8v4 ----
