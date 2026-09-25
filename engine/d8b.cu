// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-c: WIDE-LOAD GEMV variants (u64 = 8B per lane on qs/qh runs, u32 on
// IQ3 scale words, u16 on IQ3 qs). NO REPACK: the fork layouts already store each
// lane's bytes contiguously; the originals just loaded them a byte at a time.
// MATH IS BIT-IDENTICAL to the W1-b kernels: same decoded values, same fp
// expression order, same accumulation order (j-loop then shfl tree).
// Probe evidence (probe_w1c.py): u16/u32/u64/uint4 loads ALL pass on this dext
// (the W1A 'uint32 faults on Q5' was an OOB artifact). OOB reads = hard fault.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define VOCAB 248320
#define FFN_N 17408
#define KVOUT 1024
#define EPS_N 1e-6f

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x))); }
__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- q5g8: GDN qkv (Q5, warps [0,10240)) + gate (IQ3 wide, [10240,16384)) ----
// ---------------- Q5_K wide row body (row 3520B = 20 x 176B blocks) ----------------
// lane's qs run: 8 bytes at blk+48+((lane>>3)<<5)+((lane&3)<<3) -> ONE u64.
// lane's qh run: 8 bytes at blk+16+((lane&3)<<3)                          -> ONE u64.
#define Q5ROW(OUT) { \
  const unsigned char* rowp = wq5 + (size_t)warp * 3520u; \
  _Pragma("unroll 1") \
  for (int b = 0; b < 20; ++b) { \
    const unsigned char* blk = rowp + b*176; \
    const float d = __half2float(*((const __half*)blk)); \
    const float dm = __half2float(*((const __half*)(blk+2))); \
    const int s = lane >> 2; \
    float sc, mn; \
    if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); } \
    else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4)); \
           mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); } \
    const unsigned long long qs8 = *(const unsigned long long*)(blk + 48 + ((lane >> 3) << 5) + ((lane & 3) << 3)); \
    const unsigned long long qh8 = *(const unsigned long long*)(blk + 16 + ((lane & 3) << 3)); \
    const int nsh = ((lane >> 2) & 1) << 2; \
    const int koff = (b << 8) + (lane << 3); \
    const float4 xf0 = *(const float4*)(xh + koff); \
    const __half2* h0 = (const __half2*)&xf0; \
    float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]), \
           f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]); \
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y }; \
    _Pragma("unroll") \
    for (int j = 0; j < 8; ++j) { \
      int qv = (int)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu); \
      qv += (int)((((unsigned int)(qh8 >> (8*j)) >> s) & 1u) << 4); \
      const float w = d*sc*(float)qv - dm*mn; \
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w))); \
    } \
  } \
  _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o); \
  if (lane == 0) (OUT)[warp] = (__half)acc; }

// ---------------- IQ3_XXS wide row macro (NB blocks x 98B) ----------------
#define IQ3ROW(ROWP, NB, XSRC, OUTE) { \
  _Pragma("unroll 1") \
  for (int b = 0; b < (NB); ++b) { \
    const unsigned char* blk = (ROWP) + (size_t)b * 98u; \
    const float d = __half2float(*((const __half*)blk)); \
    const unsigned int sw = *(const unsigned int*)(blk + 66 + 4*(lane>>2)); \
    const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
    const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
    const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
    const unsigned int q = *(const unsigned short*)(blk + 2 + 2*lane); \
    const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
    const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
    const int koff = (b << 8) + (lane << 3); \
    const float4 xa = *(const float4*)((XSRC) + koff); \
    const __half2* hx = (const __half2*)&xa; \
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]); \
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y }; \
    const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
    const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
    const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
    const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
    const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, \
                          db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
    _Pragma("unroll") \
    for (int j = 0; j < 8; ++j) \
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(wv[j]))); \
  } \
  _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o); \
  if (lane == 0) (OUTE); }

// ---- attention fused qkv kernels: q GEMV + k (IQ3 wide) + v (Q4 wide) in one ----
// warps [0,12288): q rows; [12288,13312): k; [13312,14336): v
#define KVBODY8(W) { \
  float acc = 0.f; \
  if ((W) < 13312) { \
    const unsigned char* rowp = wk + (size_t)((W)-12288) * 1960u; \
    _Pragma("unroll 1") \
    for (int b = 0; b < 20; ++b) { \
      const unsigned char* blk = rowp + (size_t)b * 98u; \
      const float d = __half2float(*((const __half*)blk)); \
      const unsigned short* scw = (const unsigned short*)(blk + 66); \
      const unsigned int sw = ((unsigned int)scw[2*(lane>>2)]) | (((unsigned int)scw[2*(lane>>2)+1]) << 16); \
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
      const unsigned int q = *(const unsigned short*)(blk + 2 + 2*lane); \
      const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
      const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
      const int koff = (b << 8) + (lane << 3); \
      const float4 xa = *(const float4*)(xh + koff); \
      const __half2* hx = (const __half2*)&xa; \
      float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]); \
      const float xvv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y }; \
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
      const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
      _Pragma("unroll") \
      for (int j = 0; j < 8; ++j) acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(wv[j]))); \
    } \
    _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o); \
    if (lane == 0) krow[(W)-12288] = (__half)acc; \
  } else { \
    const int r = (W) - 13312; \
    const unsigned char* rowp = wv4 + (size_t)r * 2880u; \
    _Pragma("unroll 1") \
    for (int b = 0; b < 20; ++b) { \
      const unsigned char* blk = rowp + b*144; \
      const float d = __half2float(*((const __half*)blk)); \
      const float dm = __half2float(*((const __half*)(blk+2))); \
      const int s = lane >> 2; \
      float sc, mn; \
      if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); } \
      else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4)); \
             mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); } \
      const unsigned long long qs8 = *(const unsigned long long*)(blk + 16 + ((lane >> 3) << 5) + ((lane & 3) << 3)); \
      const int nsh = ((lane >> 2) & 1) << 2; \
      const int koff = (b << 8) + (lane << 3); \
      const float4 xf0 = *(const float4*)(xh + koff); \
      const __half2* h0 = (const __half2*)&xf0; \
      float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]), f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]); \
      const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y }; \
      _Pragma("unroll") \
      for (int j = 0; j < 8; ++j) { \
        const float qv = (float)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu); \
        const float w = d*sc*qv - dm*mn; \
        acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w))); \
      } \
    } \
    _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o); \
    if (lane == 0) vrow[r] = (__half)acc; \
  } }


extern "C" __global__ void __launch_bounds__(256) d8b(
    const unsigned char* __restrict__ wd, const float* __restrict__ gridf,
    const __half* __restrict__ gact, const float* __restrict__ hh, float* __restrict__ y)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wd + (size_t)warp * 6664u;
  float acc = 0.f;
  IQ3ROW(rowp, 68, gact, (y[warp] = hh[warp] + (float)((__half)acc)))
}

// ---- op38: GDN o_proj IQ3 ssm_out (24 blocks x 98B, replaces k3a_iq3) ----
