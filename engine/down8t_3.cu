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
extern "C" __global__ void __launch_bounds__(256) down8t_3(
    const unsigned char* __restrict__ wd, const float* __restrict__ gridf,
    const __half* __restrict__ gact3, const float* __restrict__ hh3, float* __restrict__ y3)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
#define NB 68
#define NB8 72
#define RS (98*NB8)
  const unsigned char* rp = wd + (size_t)warp * RS;
  const uint4* q4 = (const uint4*)(rp + lane*2*NB8);
  const uint4* s4 = (const uint4*)(rp + 64*NB8 + (lane>>2)*4*NB8);
  const uint4* d4 = (const uint4*)(rp + 96*NB8);
  float a0=0.f, a1=0.f, a2=0.f;
  _Pragma("unroll 3") for (int g = 0; g < NB8/8; ++g) {
    const uint4 qv = q4[g];
    const uint4 s0v = s4[2*g], s1v = s4[2*g+1];
    const uint4 dv = d4[g];
    const unsigned short* qh = (const unsigned short*)&qv;
    const unsigned int* sh = (const unsigned int*)&s0v;
    const unsigned int* sh2 = (const unsigned int*)&s1v;
    const unsigned short* dh = (const unsigned short*)&dv;
    _Pragma("unroll") for (int j = 0; j < 8; ++j) {
      const int b = 8*g + j;
      if (b < NB) {
        const unsigned int sw = (j < 4) ? sh[j] : sh2[j-4];
        const int koff = (b << 8) + (lane << 3);
        LD8(xv0, gact3, FFN_N, 0, koff) LD8(xv1, gact3, FFN_N, 1, koff) LD8(xv2, gact3, FFN_N, 2, koff)
        IQ3T_DEC(qh[j], sw, dh[j], xv0, xv1, xv2, a0, a1, a2)
      }
    }
  }
  RED3(a0,a1,a2)
  if (lane == 0) {
    y3[0*DIM+warp] = hh3[0*DIM+warp] + (float)((__half)a0);
    y3[1*DIM+warp] = hh3[1*DIM+warp] + (float)((__half)a1);
    y3[2*DIM+warp] = hh3[2*DIM+warp] + (float)((__half)a2);
  }
#undef NB
#undef NB8
#undef RS
}

// ---- op38t_3: GDN o proj T-packed IQ3, 3 rows (NB=24, NB8=24) ----
