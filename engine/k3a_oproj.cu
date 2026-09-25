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
extern "C" __global__ void __launch_bounds__(256) k3a_oproj(
    const unsigned char* __restrict__ wq8, const __half* __restrict__ z, __half* __restrict__ attn_out)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wq8 + (size_t)warp * 6528u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 192; ++b) {          // sequential blocks (lane-strided start miscompiles on this dext)
    const unsigned char* blk = rowp + b*34;
    const float d = __half2float(*((const __half*)blk));
    const float w = d * (float)((signed char)blk[2+lane]);
    acc += __half2float(__hmul(z[(b<<5)+lane], __float2half(w)));
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) attn_out[warp] = (__half)acc;
}

// ---- K3m: hh = x + attn_out ; hhx = half(ffn_norm(hh)) ----
