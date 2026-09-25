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
extern "C" __global__ void __launch_bounds__(256) k1_q5(
    const unsigned char* __restrict__ wq5, const __half* __restrict__ xh, __half* __restrict__ qkv_row)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const unsigned char* rowp = wq5 + (size_t)warp * 3520u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + b*176;
    const float d = __half2float(*((const __half*)blk));
    const float dm = __half2float(*((const __half*)(blk+2)));
    const int s = lane >> 2;
    float sc, mn;
    if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
    else { sc = (float)((blk[s+8] & 0xF) | ((blk[s] >> 6) << 4));
           mn = (float)((blk[s+8] >> 4) | ((blk[s+4] >> 6) << 4)); }
    // TRUTH-FITTED (stock tinygrad dequant): byte = (k>>6)*32 + (k&31), nibble = (k>>5)&1,
    // qh byte = k&31 bit = k>>5, scale sub = k>>5 (k = lane*8+j within the 256-block)
    const unsigned char* qsb = blk + 48 + ((lane >> 3) << 5) + ((lane & 3) << 3);
    const int nsh = ((lane >> 2) & 1) << 2;
    const unsigned char* qhp = blk + 16 + ((lane & 3) << 3);
    const int koff = (b << 8) + (lane << 3);
    const float4 xf0 = *(const float4*)(xh + koff);
    const __half2* h0 = (const __half2*)&xf0;
    float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
           f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int qv = (int)((qsb[j] >> nsh) & 0xF);
      qv += ((qhp[j] >> s) & 1) << 4;
      const float w = d*sc*(float)qv - dm*mn;
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) qkv_row[warp] = (__half)acc;
}
