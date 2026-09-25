// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2-MTP: T=3 PROBE trunk kernels (M=3 batched rows; per-row fp op order
// is IDENTICAL to the T=1 kernels -> bit-exact rows 0..2 vs the T=1 engine path).
// Laws: flat indexing, sequential loops, no gridDim reads, per-kernel cubins,
// sub-mask 0xff shuffles only, single-array smem, hardcoded sizes.
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
#define VOCAB 248320

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x))); }
__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// load 8 halves from row T (base XB, row stride TS) into 8 floats named NM
#define LD8(NM, XB, TS, T, KO) float NM[8]; { \
  const float4 xa = *(const float4*)((XB) + (T)*(TS) + (KO)); \
  const __half2* hx = (const __half2*)&xa; \
  float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]); \
  NM[0]=f0.x; NM[1]=f0.y; NM[2]=f1.x; NM[3]=f1.y; NM[4]=f2.x; NM[5]=f2.y; NM[6]=f3.x; NM[7]=f3.y; }

#define RED3(A0,A1,A2) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); } }

// 3-acc GEMV row loop over j (weight values wv[8] shared, x from x0/x1/x2)
#define ACC3(X0, X1, X2, W8) { \
  _Pragma("unroll") for (int j = 0; j < 8; ++j) { \
    const __half wh = __float2half((W8)[j]); \
    a0 += __half2float(__hmul(__float2half((X0)[j]), wh)); \
    a1 += __half2float(__hmul(__float2half((X1)[j]), wh)); \
    a2 += __half2float(__hmul(__float2half((X2)[j]), wh)); } }

// ---------------- Q5_K wide row body, 3 rows (raw layout; row 3520B) ----------------
#define Q5ROW3(XB, TS, OB, OS, RIDX) { \
  const unsigned char* rowp = wq5 + (size_t)(RIDX) * 3520u; \
  float a0 = 0.f, a1 = 0.f, a2 = 0.f; \
  _Pragma("unroll 5") \
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
    LD8(xv0, XB, TS, 0, koff) LD8(xv1, XB, TS, 1, koff) LD8(xv2, XB, TS, 2, koff) \
    float wv[8]; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) { \
      int qv = (int)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu); \
      qv += (int)((((unsigned int)(qh8 >> (8*j)) >> s) & 1u) << 4); \
      wv[j] = d*sc*(float)qv - dm*mn; } \
    ACC3(xv0, xv1, xv2, wv) \
  } \
  RED3(a0,a1,a2) \
  if (lane == 0) { (OB)[0*(OS)+(RIDX)] = (__half)a0; (OB)[1*(OS)+(RIDX)] = (__half)a1; (OB)[2*(OS)+(RIDX)] = (__half)a2; } }

// ---------------- IQ3_XXS packed row body, 3 rows ----------------
#define IQ3ROWP3(ROWB, NB, XB, TS, OB, OS, RIDX) { \
  const unsigned char* rowp = (ROWB); \
  const unsigned short* qsp = (const unsigned short*)(rowp); \
  const unsigned int* scp = (const unsigned int*)(rowp + 64*(NB)); \
  const unsigned short* dpp = (const unsigned short*)(rowp + 96*(NB)); \
  float a0 = 0.f, a1 = 0.f, a2 = 0.f; \
  _Pragma("unroll 5") \
  for (int b = 0; b < (NB); ++b) { \
    const float d = __half2float(__ushort_as_half(dpp[b])); \
    const unsigned int sw = scp[8*b + (lane>>2)]; \
    const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
    const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
    const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
    const unsigned int q = qsp[32*b + lane]; \
    const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
    const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
    const int koff = (b << 8) + (lane << 3); \
    LD8(xv0, XB, TS, 0, koff) LD8(xv1, XB, TS, 1, koff) LD8(xv2, XB, TS, 2, koff) \
    const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
    const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
    const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
    const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
    float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, \
                    db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 }; \
    ACC3(xv0, xv1, xv2, wv) \
  } \
  RED3(a0,a1,a2) \
  if (lane == 0) { (OB)[0*(OS)+(RIDX)] = (__half)a0; (OB)[1*(OS)+(RIDX)] = (__half)a1; (OB)[2*(OS)+(RIDX)] = (__half)a2; } }

// ---------------- Q4_K raw v row body, 3 rows ----------------
#define V4ROW3 { \
  const int r = warp - 13312; \
  const unsigned char* rowp = wv4 + (size_t)r * 2880u; \
  float a0 = 0.f, a1 = 0.f, a2 = 0.f; \
  _Pragma("unroll 5") \
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
    LD8(xv0, xh3, DIM, 0, koff) LD8(xv1, xh3, DIM, 1, koff) LD8(xv2, xh3, DIM, 2, koff) \
    float wv[8]; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) { \
      const float qv = (float)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu); \
      wv[j] = d*sc*qv - dm*mn; } \
    ACC3(xv0, xv1, xv2, wv) \
  } \
  RED3(a0,a1,a2) \
  if (lane == 0) { vrow3[0*1024+r] = (__half)a0; vrow3[1*1024+r] = (__half)a1; vrow3[2*1024+r] = (__half)a2; } }

// ---- h_embed3: 3 token slots -> x3 rows (IQ3_S gather+dequant, fp32 out) ----
extern "C" __global__ void __launch_bounds__(256) ffn8_3(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const float* __restrict__ gridf, const __half* __restrict__ hhx3, __half* __restrict__ gact3)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= FFN_N) return;
  const unsigned short* qg = (const unsigned short*)(wg + (size_t)warp * 1960u);
  const unsigned int* sg = (const unsigned int*)(wg + (size_t)warp * 1960u + 1280u);
  const unsigned short* dg = (const unsigned short*)(wg + (size_t)warp * 1960u + 1920u);
  const unsigned short* qu = (const unsigned short*)(wu + (size_t)warp * 1960u);
  const unsigned int* su = (const unsigned int*)(wu + (size_t)warp * 1960u + 1280u);
  const unsigned short* du = (const unsigned short*)(wu + (size_t)warp * 1960u + 1920u);
  float ag[3] = {0.f,0.f,0.f}, au[3] = {0.f,0.f,0.f};
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    LD8(xg0, hhx3, DIM, 0, koff) LD8(xg1, hhx3, DIM, 1, koff) LD8(xg2, hhx3, DIM, 2, koff)
    #define IQ3P2T(QP, SP, DP, X8, IDX) { \
      const float d = __half2float(__ushort_as_half((DP)[b])); \
      const unsigned int sw = (SP)[8*b + (lane>>2)]; \
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
      const unsigned int q = (QP)[32*b + lane]; \
      const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
      const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
      const float wv0 = db*g0.x*sg0, wv1 = db*g0.y*sg1, wv2 = db*g0.z*sg2, wv3 = db*g0.w*sg3; \
      const float wv4 = db*g1.x*sg4, wv5 = db*g1.y*sg5, wv6 = db*g1.z*sg6, wv7 = db*g1.w*sg7; \
      const __half hh[8] = { __float2half(wv0), __float2half(wv1), __float2half(wv2), __float2half(wv3), \
                             __float2half(wv4), __float2half(wv5), __float2half(wv6), __float2half(wv7) }; \
      const float* X8f = (X8); \
      _Pragma("unroll") for (int j = 0; j < 8; ++j) { \
        const float pr = __half2float(__hmul(__float2half(X8f[j]), hh[j])); \
        if ((IDX) == 0) { ag[0] += pr; au[0] += pr; } \
        else if ((IDX) == 1) { ag[1] += pr; au[1] += pr; } \
        else { ag[2] += pr; au[2] += pr; } } }
    // gate weight adds to ag, up weight adds to au (separate calls, per row)
    #define IQ3G(QP, SP, DP, X8, IDX) { \
      const float d = __half2float(__ushort_as_half((DP)[b])); \
      const unsigned int sw = (SP)[8*b + (lane>>2)]; \
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
      const unsigned int q = (QP)[32*b + lane]; \
      const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
      const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
      const float wv0 = db*g0.x*sg0, wv1 = db*g0.y*sg1, wv2 = db*g0.z*sg2, wv3 = db*g0.w*sg3; \
      const float wv4 = db*g1.x*sg4, wv5 = db*g1.y*sg5, wv6 = db*g1.z*sg6, wv7 = db*g1.w*sg7; \
      const __half hh[8] = { __float2half(wv0), __float2half(wv1), __float2half(wv2), __float2half(wv3), \
                             __float2half(wv4), __float2half(wv5), __float2half(wv6), __float2half(wv7) }; \
      const float* X8f = (X8); \
      _Pragma("unroll") for (int j = 0; j < 8; ++j) ag[(IDX)] += __half2float(__hmul(__float2half(X8f[j]), hh[j])); }
    #define IQ3U(QP, SP, DP, X8, IDX) { \
      const float d = __half2float(__ushort_as_half((DP)[b])); \
      const unsigned int sw = (SP)[8*b + (lane>>2)]; \
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
      const unsigned int q = (QP)[32*b + lane]; \
      const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2))); \
      const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2))); \
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f; \
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f; \
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f; \
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f; \
      const float wv0 = db*g0.x*sg0, wv1 = db*g0.y*sg1, wv2 = db*g0.z*sg2, wv3 = db*g0.w*sg3; \
      const float wv4 = db*g1.x*sg4, wv5 = db*g1.y*sg5, wv6 = db*g1.z*sg6, wv7 = db*g1.w*sg7; \
      const __half hh[8] = { __float2half(wv0), __float2half(wv1), __float2half(wv2), __float2half(wv3), \
                             __float2half(wv4), __float2half(wv5), __float2half(wv6), __float2half(wv7) }; \
      const float* X8f = (X8); \
      _Pragma("unroll") for (int j = 0; j < 8; ++j) au[(IDX)] += __half2float(__hmul(__float2half(X8f[j]), hh[j])); }
    IQ3G(qg, sg, dg, xg0, 0) IQ3U(qu, su, du, xg0, 0)
    IQ3G(qg, sg, dg, xg1, 1) IQ3U(qu, su, du, xg1, 1)
    IQ3G(qg, sg, dg, xg2, 2) IQ3U(qu, su, du, xg2, 2)
    #undef IQ3P2T
    #undef IQ3G
    #undef IQ3U
  }
  RED3(ag[0],ag[1],ag[2])
  RED3(au[0],au[1],au[2])
  if (lane == 0) {
    gact3[0*FFN_N + warp] = __hmul(hsilu_h((__half)ag[0]), (__half)au[0]);
    gact3[1*FFN_N + warp] = __hmul(hsilu_h((__half)ag[1]), (__half)au[1]);
    gact3[2*FFN_N + warp] = __hmul(hsilu_h((__half)ag[2]), (__half)au[2]);
  }
}

// ---- down8_3: down GEMV IQ3 packed + residual, 3 rows ----
