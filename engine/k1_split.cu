// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120

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
    const unsigned int q4 = *(const unsigned int*)(blk + 48 + (lane << 2));
    const unsigned char* qhp = blk + 16 + ((lane & 3) << 3);
    const int koff = (b << 8) + (lane << 3);
    const float4 xf0 = *(const float4*)(xh + koff);
    const __half2* h0 = (const __half2*)&xf0;
    float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
           f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned int bte = (q4 >> ((j>>1)*8)) & 0xFF;
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

extern "C" __global__ void __launch_bounds__(256) k1_iq3(
    const unsigned char* __restrict__ wq3g, const float* __restrict__ gridf,
    const __half* __restrict__ xh, __half* __restrict__ gate_row)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int row = warp;
  const unsigned char* rowp = wq3g + (size_t)row * 1960u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + (size_t)b * 98u;
    const float d = __half2float(*((const __half*)blk));
    const unsigned short* scw = (const unsigned short*)(blk + 66);
    const unsigned int sw = ((unsigned int)scw[2*(lane>>2)]) | (((unsigned int)scw[2*(lane>>2)+1]) << 16);
    const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
    const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu;
    const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
    const unsigned int q = ((unsigned int)blk[2 + 2*lane]) | (((unsigned int)blk[3 + 2*lane]) << 8);
    const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2)));
    const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2)));
    const int koff = (b << 8) + (lane << 3);
    const float4 xa = *(const float4*)(xh + koff);
    const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
    const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
    const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
    const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xvv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3,
                          db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(wv[j])));
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) gate_row[row] = (__half)acc;
}

extern "C" __global__ void __launch_bounds__(256) k1_ab(
    const float* __restrict__ walpha, const float* __restrict__ wbeta,
    const __half* __restrict__ xh, float* __restrict__ alpharaw, float* __restrict__ betaraw)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 48) {
    const float* wr = walpha + (size_t)warp * DIM;
    float acc = 0.f;
    for (int i = lane; i < DIM; i += 32) acc += wr[i] * __half2float(xh[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) alpharaw[warp] = acc;
  } else {
    const float* wr = wbeta + (size_t)(warp-48) * DIM;
    float acc = 0.f;
    for (int i = lane; i < DIM; i += 32) acc += wr[i] * __half2float(xh[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) betaraw[warp-48] = acc;
  }
}
