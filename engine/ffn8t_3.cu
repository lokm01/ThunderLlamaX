// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2D-L1: T-layout (lane-contiguous 16B) IQ3 GEMVs, M=3 probe rows.
// Weights from pack_t.py (pure transpose of the W1C-aligned IQ3 pack): decoded
// values identical, per-row k-order identical -> outputs BIT-IDENTICAL to m3.cu.
// Laws: flat indexing, sequential loops, no gridDim reads, per-kernel cubins,
// no runtime-indexed local arrays (j-loop fully unrolled -> constant indices),
// 16B-aligned loads for all parities (NB8 multiple of 8 keeps them aligned).
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define FFN_N 17408
#define EPS_N 1e-6f

// load 8 halves from row T (base XB, row stride TS) into 8 floats named NM
#define LD8(NM, XB, TS, T, KO) float NM[8]; { \
  const float4 xa = *(const float4*)((XB) + (T)*(TS) + (KO)); \
  const __half2* hx = (const __half2*)&xa; \
  float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]); \
  NM[0]=f0.x; NM[1]=f0.y; NM[2]=f1.x; NM[3]=f1.y; NM[4]=f2.x; NM[5]=f2.y; NM[6]=f3.x; NM[7]=f3.y; }

#define RED3(A0,A1,A2) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); } }

// ---- decode one block from REGISTER values (q=u16, sw=u32, d=u16) for rows X0..X2 ----
// ACC order identical to m3.cu: j ascending 0..7, per-j half product then fp add.
#define IQ3T_DEC(qv, swv, dv, X0, X1, X2, A0, A1, A2) { \
  const float d = __half2float(__ushort_as_half(dv)); \
  const float db = d * (((float)(swv >> 28)) + 0.5f) * 0.5f; \
  const unsigned int sidx = (swv >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
  const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
  const float4 g0 = *((const float4*)(gridf + ((qv & 0xFFu) << 2))); \
  const float4 g1 = *((const float4*)(gridf + ((qv >> 8) << 2))); \
  const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
  const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
  const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
  const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
  float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, \
                  db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
  _Pragma("unroll") for (int j = 0; j < 8; ++j) { \
    const __half wh = __float2half(wv[j]); \
    A0 += __half2float(__hmul(__float2half((X0)[j]), wh)); \
    A1 += __half2float(__hmul(__float2half((X1)[j]), wh)); \
    A2 += __half2float(__hmul(__float2half((X2)[j]), wh)); } }

__device__ __forceinline__ __half hsilu_h_t(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- ffn8t_3: gate+up T-packed GEMVs + silu-mul, 3 rows (NB=20, NB8=24) ----
extern "C" __global__ void __launch_bounds__(256) ffn8t_3(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const float* __restrict__ gridf, const __half* __restrict__ hhx3, __half* __restrict__ gact3)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= FFN_N) return;
#define NB 20
#define NB8 24
#define RS (98*NB8)
  const unsigned char* rg = wg + (size_t)warp * RS;
  const unsigned char* ru = wu + (size_t)warp * RS;
  const uint4* qg4 = (const uint4*)(rg + lane*2*NB8);
  const uint4* qu4 = (const uint4*)(ru + lane*2*NB8);
  const uint4* sg4 = (const uint4*)(rg + 64*NB8 + (lane>>2)*4*NB8);
  const uint4* su4 = (const uint4*)(ru + 64*NB8 + (lane>>2)*4*NB8);
  const uint4* dg4 = (const uint4*)(rg + 96*NB8);
  const uint4* du4 = (const uint4*)(ru + 96*NB8);
  float ag0=0.f,ag1=0.f,ag2=0.f, au0=0.f,au1=0.f,au2=0.f;
  _Pragma("unroll") for (int g = 0; g < NB8/8; ++g) {
    const uint4 qgv = qg4[g], quv = qu4[g];
    const uint4 sg0v = sg4[2*g], sg1v = sg4[2*g+1], su0v = su4[2*g], su1v = su4[2*g+1];
    const uint4 dgv = dg4[g], duv = du4[g];
    const unsigned short* qgh = (const unsigned short*)&qgv;
    const unsigned short* quh = (const unsigned short*)&quv;
    const unsigned int* sgh = (const unsigned int*)&sg0v;   // b=8g+0..3
    const unsigned int* sgh2 = (const unsigned int*)&sg1v;  // b=8g+4..7
    const unsigned int* suh = (const unsigned int*)&su0v;
    const unsigned int* suh2 = (const unsigned int*)&su1v;
    const unsigned short* dgh = (const unsigned short*)&dgv;
    const unsigned short* duh = (const unsigned short*)&duv;
    _Pragma("unroll") for (int j = 0; j < 8; ++j) {
      const int b = 8*g + j;
      if (b < NB) {
        const unsigned int swg = (j < 4) ? sgh[j] : sgh2[j-4];
        const unsigned int swu = (j < 4) ? suh[j] : suh2[j-4];
        const int koff = (b << 8) + (lane << 3);
        LD8(xg0, hhx3, DIM, 0, koff) LD8(xg1, hhx3, DIM, 1, koff) LD8(xg2, hhx3, DIM, 2, koff)
        IQ3T_DEC(qgh[j], swg, dgh[j], xg0, xg1, xg2, ag0, ag1, ag2)
        IQ3T_DEC(quh[j], swu, duh[j], xg0, xg1, xg2, au0, au1, au2)
      }
    }
  }
  RED3(ag0,ag1,ag2)
  RED3(au0,au1,au2)
  if (lane == 0) {
    gact3[0*FFN_N + warp] = __hmul(hsilu_h_t((__half)ag0), (__half)au0);
    gact3[1*FFN_N + warp] = __hmul(hsilu_h_t((__half)ag1), (__half)au1);
    gact3[2*FFN_N + warp] = __hmul(hsilu_h_t((__half)ag2), (__half)au2);
  }
#undef NB
#undef NB8
#undef RS
}

// ---- down8t_3: down T-packed GEMV + residual, 3 rows (NB=68, NB8=72) ----
