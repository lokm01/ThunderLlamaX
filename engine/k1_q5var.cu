// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
#define FULL 0xffffffffu
// v1: NO uint32 weight load (byte loads); float4 xh KEPT
extern "C" __global__ void __launch_bounds__(256) k1_q5_v1(
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
    const unsigned char* qs = blk + 48 + (lane << 2);
    const unsigned char* qhp = blk + 16 + ((lane & 3) << 3);
    const int koff = (b << 8) + (lane << 3);
    const float4 xf0 = *(const float4*)(xh + koff);
    const __half2* h0 = (const __half2*)&xf0;
    float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
           f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned int bte = qs[j>>1];
      int qv = (j&1) ? (int)(bte>>4) : (int)(bte & 0xF);
      qv += ((qhp[j] >> s) & 1) << 4;
      const float w = d*sc*(float)qv - dm*mn;
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) qkv_row[warp] = (__half)acc;
}
// v2: uint32 weight load KEPT; NO float4 (8 half loads)
extern "C" __global__ void __launch_bounds__(256) k1_q5_v2(
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
    const unsigned int q4 = *(const unsigned int*)(blk + 48 + (lane << 2));
    const unsigned char* qhp = blk + 16 + ((lane & 3) << 3);
    const int koff = (b << 8) + (lane << 3);
    const __half* xs = xh + koff;
    float xvv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) xvv[j] = __half2float(xs[j]);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned int bte = (q4 >> ((j>>1)*8)) & 0xFF;
      int qv = (j&1) ? (int)(bte>>4) : (int)(bte & 0xF);
      qv += ((qhp[j] >> s) & 1) << 4;
      const float w = d*sc*(float)qv - dm*mn;
      acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) qkv_row[warp] = (__half)acc;
}
// v3: neither vector load
extern "C" __global__ void __launch_bounds__(256) k1_q5_v3(
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
    const unsigned char* qs = blk + 48 + (lane << 2);
    const unsigned char* qhp = blk + 16 + ((lane & 3) << 3);
    const int koff = (b << 8) + (lane << 3);
    const __half* xs = xh + koff;
    float xvv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) xvv[j] = __half2float(xs[j]);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned int bte = qs[j>>1];
      int qv = (j&1) ? (int)(bte>>4) : (int)(bte & 0xF);
      qv += ((qhp[j] >> s) & 1) << 4;
      const float w = d*sc*(float)qv - dm*mn;
      acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) qkv_row[warp] = (__half)acc;
}
