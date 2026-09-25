// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-a: fully-fused GDN block (T=1), Qwen3.8-27B real dims.
// HARD RULES: flat indexing only, NO gridDim/blockDim reads (hardcoded 256 thr),
// single-array smem only (none used), nwarps/nthreads passed as needed.
// Numerics replicate stock tinygrad dtype flow: half(xh) * half(w) fp32-acc GEMVs
// with half outputs; fp32 alpha/beta GEMVs; fp32 conv/scan; half silu on gates.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define NVH 48
#define VDIM 128
#define KDIM 128
#define CONV_CH 10240
#define QDIM 2048
#define INNER 6144
#define FFN_N 17408
#define EPS_N 1e-6f
#define EPS_Q 1e-6f
#define ISQ128 0.08838834764831845f

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x))); }
__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- K0: RMS(x) -> xh = half(x*r*nw) ; 1 CTA, per-warp redundant reduce ----
extern "C" __global__ void __launch_bounds__(256) k3b_ffn(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const float* __restrict__ gridf, const __half* __restrict__ hhx, __half* __restrict__ gact)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= FFN_N) return;
  float ag = 0.f, au = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const __half* xb = hhx + (b << 8) + (lane << 3);
    const float4 xa = *(const float4*)(xb);
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #define IQ3ACC(ROWP, ACC) { \
      const unsigned char* blk = (ROWP) + (size_t)b * 98u; \
      const float d = __half2float(*((const __half*)blk)); \
      const unsigned short* scw = (const unsigned short*)(blk + 66); \
      const unsigned int sw = ((unsigned int)scw[2*(lane>>2)]) | (((unsigned int)scw[2*(lane>>2)+1]) << 16); \
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
      const unsigned int q = ((unsigned int)blk[2 + 2*lane]) | (((unsigned int)blk[3 + 2*lane]) << 8); \
      const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
      const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
      const float wv0 = db*g0.x*sg0, wv1 = db*g0.y*sg1, wv2 = db*g0.z*sg2, wv3 = db*g0.w*sg3; \
      const float wv4 = db*g1.x*sg4, wv5 = db*g1.y*sg5, wv6 = db*g1.z*sg6, wv7 = db*g1.w*sg7; \
      (ACC) += __half2float(__hmul(__float2half(xv[0]), __float2half(wv0))) \
             + __half2float(__hmul(__float2half(xv[1]), __float2half(wv1))); \
      (ACC) += __half2float(__hmul(__float2half(xv[2]), __float2half(wv2))) \
             + __half2float(__hmul(__float2half(xv[3]), __float2half(wv3))); \
      (ACC) += __half2float(__hmul(__float2half(xv[4]), __float2half(wv4))) \
             + __half2float(__hmul(__float2half(xv[5]), __float2half(wv5))); \
      (ACC) += __half2float(__hmul(__float2half(xv[6]), __float2half(wv6))) \
             + __half2float(__hmul(__float2half(xv[7]), __float2half(wv7))); }
    IQ3ACC(wg + (size_t)warp * 1960u, ag)
    IQ3ACC(wu + (size_t)warp * 1960u, au)
    #undef IQ3ACC
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { ag += __shfl_down_sync(FULL, ag, o); au += __shfl_down_sync(FULL, au, o); }
  if (lane == 0) gact[warp] = __hmul(hsilu_h((__half)ag), (__half)au);
}

// ---- K3c: down GEMV IQ3 [5120, 17408] + residual ----
