// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2G L3: half2-core port of mtpd.cu dkv (LDH2+ACC-pairs). Add order and fp16
// products identical to dkv -> BIT-IDENTICAL outputs. Same signature/grid.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120

extern "C" __global__ void __launch_bounds__(256) dkvh(
    const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv,
    const __half* __restrict__ xh, __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const bool isk = (warp < 1024);
  const int r = isk ? warp : warp - 1024;
  const unsigned char* rowp = (isk ? wk : wv) + (size_t)r * 2880u;
  float acc = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    const float4 xf = *(const float4*)(xh + koff);
    const __half2* hx = (const __half2*)&xf;
    const int subi = (b<<3) + (lane>>2);
    const unsigned long long qs8 = *(const unsigned long long*)(rowp + subi*16 + ((lane&1)<<3));
    const float d = __half2float(*((const __half*)(rowp + 2560 + subi*2)));
    float wv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float qv = (float)(((qs8 >> (8*j)) >> (((lane>>1)&1)<<2)) & 0xFu);
      wv[j] = d * (qv - 8.f);
    }
    const __half2 w01 = __halves2half2(__float2half(wv[0]), __float2half(wv[1]));
    const __half2 w23 = __halves2half2(__float2half(wv[2]), __float2half(wv[3]));
    const __half2 w45 = __halves2half2(__float2half(wv[4]), __float2half(wv[5]));
    const __half2 w67 = __halves2half2(__float2half(wv[6]), __float2half(wv[7]));
    { const float2 p = __half22float2(__hmul2(hx[0], w01)); acc += p.x; acc += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[1], w23)); acc += p.x; acc += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[2], w45)); acc += p.x; acc += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[3], w67)); acc += p.x; acc += p.y; }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) { if (isk) krow[r] = (__half)acc; else vrow[r] = (__half)acc; }
}
