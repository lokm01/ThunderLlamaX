// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2G L3: half2-core port of mtpd.cu dfgu (draft FFN gate+up Q4 GEMVs +
// silu-mul). Add order and fp16 products identical to dfgu -> BIT-IDENTICAL
// outputs. Same signature/grid.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define FFN_N 17408

__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

extern "C" __global__ void __launch_bounds__(256) dfguh(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const __half* __restrict__ hhx, __half* __restrict__ gact)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= FFN_N) return;
  const unsigned char* rg = wg + (size_t)warp * 2880u;
  const unsigned char* ru = wu + (size_t)warp * 2880u;
  float ag = 0.f, au = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    const float4 xf = *(const float4*)(hhx + koff);
    const __half2* hx = (const __half2*)&xf;
    const int subi = (b<<3) + (lane>>2);
    const unsigned long long qsg8 = *(const unsigned long long*)(rg + subi*16 + ((lane&1)<<3));
    const float dg = __half2float(*((const __half*)(rg + 2560 + subi*2)));
    const unsigned long long qsu8 = *(const unsigned long long*)(ru + subi*16 + ((lane&1)<<3));
    const float du = __half2float(*((const __half*)(ru + 2560 + subi*2)));
    float wg8[8], wu8[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float qg = (float)(((qsg8 >> (8*j)) >> (((lane>>1)&1)<<2)) & 0xFu);
      const float qu = (float)(((qsu8 >> (8*j)) >> (((lane>>1)&1)<<2)) & 0xFu);
      wg8[j] = dg * (qg - 8.f);
      wu8[j] = du * (qu - 8.f);
    }
    const __half2 g01 = __halves2half2(__float2half(wg8[0]), __float2half(wg8[1]));
    const __half2 g23 = __halves2half2(__float2half(wg8[2]), __float2half(wg8[3]));
    const __half2 g45 = __halves2half2(__float2half(wg8[4]), __float2half(wg8[5]));
    const __half2 g67 = __halves2half2(__float2half(wg8[6]), __float2half(wg8[7]));
    const __half2 u01 = __halves2half2(__float2half(wu8[0]), __float2half(wu8[1]));
    const __half2 u23 = __halves2half2(__float2half(wu8[2]), __float2half(wu8[3]));
    const __half2 u45 = __halves2half2(__float2half(wu8[4]), __float2half(wu8[5]));
    const __half2 u67 = __halves2half2(__float2half(wu8[6]), __float2half(wu8[7]));
    { const float2 p = __half22float2(__hmul2(hx[0], g01)); ag += p.x; ag += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[1], g23)); ag += p.x; ag += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[2], g45)); ag += p.x; ag += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[3], g67)); ag += p.x; ag += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[0], u01)); au += p.x; au += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[1], u23)); au += p.x; au += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[2], u45)); au += p.x; au += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[3], u67)); au += p.x; au += p.y; }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { ag += __shfl_down_sync(FULL, ag, o); au += __shfl_down_sync(FULL, au, o); }
  if (lane == 0) gact[warp] = __hmul(hsilu_h((__half)ag), (__half)au);
}
