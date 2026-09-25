// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-b merge pass: cut launches/token 612 -> 500 (host-bound ~85us/launch).
// k0ab = k0_norm (CTA 0) + k1_ab alpha/beta GEMV (CTAs 1..12, redundant per-warp norm)
// k2s  = k2_scan + k2b_z (same grid, z-phase after syncthreads; core is CTA-local)
// a_qkv_q6 / a_qkv_iq3 = q GEMV + a_kv (k IQ3 + v Q4_K) in one launch (branch by warp)
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

// ---- k0ab: CTA 0 = RMS(x)->xh; warps [8,104) = alpha/beta f32 GEMVs with redundant norm ----
// ---- a_qkv_q6 / a_qkv_iq3: q GEMV + k (IQ3) + v (Q4_K) in ONE launch ----
// warps [0,12288): q rows; [12288,13312): k rows (IQ3_XXS); [13312,14336): v rows (Q4_K)
#define KV_BODY(W) { \
  float acc = 0.f; \
  if ((W) < 13312) { \
    const unsigned char* rowp = wk + (size_t)((W)-12288) * 1960u; \
    _Pragma("unroll 4") \
    for (int b = 0; b < 20; ++b) { \
      const unsigned char* blk = rowp + (size_t)b * 98u; \
      const float d = __half2float(*((const __half*)blk)); \
      const unsigned short* scw = (const unsigned short*)(blk + 66); \
      const unsigned int sw = ((unsigned int)scw[2*(lane>>2)]) | (((unsigned int)scw[2*(lane>>2)+1]) << 16); \
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
      const unsigned int q = ((unsigned int)blk[2 + 2*lane]) | (((unsigned int)blk[3 + 2*lane]) << 8); \
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
    _Pragma("unroll 4") \
    for (int b = 0; b < 20; ++b) { \
      const unsigned char* blk = rowp + b*144; \
      const float d = __half2float(*((const __half*)blk)); \
      const float dm = __half2float(*((const __half*)(blk+2))); \
      const int s = lane >> 2; \
      float sc, mn; \
      if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); } \
      else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4)); mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); } \
      const unsigned char* qsb = blk + 16 + ((lane >> 3) << 5) + ((lane & 3) << 3); \
      const int nsh = ((lane >> 2) & 1) << 2; \
      const int koff = (b << 8) + (lane << 3); \
      const float4 xf0 = *(const float4*)(xh + koff); \
      const __half2* h0 = (const __half2*)&xf0; \
      float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]), f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]); \
      const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y }; \
      _Pragma("unroll") \
      for (int j = 0; j < 8; ++j) { \
        const float qv = (float)((qsb[j] >> nsh) & 0xF); \
        acc += __half2float(__hmul(__float2half(xv[j]), __float2half(d*sc*qv - dm*mn))); \
      } \
    } \
    _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o); \
    if (lane == 0) vrow[r] = (__half)acc; \
  } }

extern "C" __global__ void __launch_bounds__(256) k1_q5g(
    const unsigned char* __restrict__ wq5, const unsigned char* __restrict__ wq3g,
    const float* __restrict__ gridf, const __half* __restrict__ xh,
    __half* __restrict__ qkv_row, __half* __restrict__ gate_row)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  float acc = 0.f;
  if (warp < 10240) {
    const unsigned char* rowp = wq5 + (size_t)warp * 3520u;
    #pragma unroll 4
    for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + b*176;
      const float d = __half2float(*((const __half*)blk));
      const float dm = __half2float(*((const __half*)(blk+2)));
      const int s = lane >> 2;
      float sc, mn;
      if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
      else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
             mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
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
  } else {
    const int r = warp - 10240;   // [0,6144)
    const unsigned char* rowp = wq3g + (size_t)r * 1960u;
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
      const __half2* hx = (const __half2*)&xa;
      float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
      const float xvv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
      const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3,
                            db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
      #pragma unroll
      for (int j = 0; j < 8; ++j) acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(wv[j])));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) gate_row[r] = (__half)acc;
  }
}
