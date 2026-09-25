// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-c v2: WIDE-LOAD GEMVs on ALIGNED-REPACKED weights (pack_w1c.py).
// ALIGNMENT LAW (PTX-proven): nvcc merges adjacent narrow loads into wider ones
// (u16 pair -> u32). Every per-lane multi-byte run must sit on natural alignment
// or the dext faults (SM Multiple Warp Errors). Packed layouts make all merges legal:
//   IQ3_XXS row (98B*NB, same size): [qs 64B/blk][scales 32B/blk][d 2B/blk]
//     -> qs u16 @64b+2lane (2B), scale u32 @32b+4s (4B), d u16 @2b (2B)
//   Q6_K row (212B*NB): [pad2][d2][sc16][lo128][qh64] -> 2x u32 runs per lane
//   Q5_K / Q4_K blocks (176B / 144B) are already 8B-aligned -> u64 runs, RAW layout
// Math is BIT-IDENTICAL to W1-b kernels (same decoded values, same fp order).
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define VOCAB 248320
#define FFN_N 17408
#define KVOUT 1024

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x))); }
__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- attention fused qkv: q + k (IQ3 packed) + v (Q4 raw) ----
#define KVBODY8(W) { \
  float acc = 0.f; \
  if ((W) < 13312) { \
    const int r = (W) - 12288; \
    IQ3ROWP(wk + (size_t)r * 1960u, 20, xh, (krow[r] = (__half)acc)) \
  } else { \
    const int r = (W) - 13312; \
    const unsigned char* rowp = wv4 + (size_t)r * 2880u; \
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


// ---------------- Q5_K wide row body (raw layout; row 3520B = 20 x 176B) ----------------
#define Q5ROW(OUT) { \
  const unsigned char* rowp = wq5 + (size_t)warp * 3520u; \
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

// ---------------- IQ3_XXS wide row body (PACKED regions; row 98*NB) ----------------
#define IQ3ROWP(ROWB, NB, XSRC, OUTE) { \
  const unsigned char* rowp = (ROWB); \
  const unsigned short* qsp = (const unsigned short*)(rowp); \
  const unsigned int* scp = (const unsigned int*)(rowp + 64*(NB)); \
  const unsigned short* dpp = (const unsigned short*)(rowp + 96*(NB)); \
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

// ---- q5g8: GDN qkv (Q5 raw, warps [0,10240)) + gate (IQ3 packed, [10240,16384)) ----
extern "C" __global__ void __launch_bounds__(256) q5g8(
    const unsigned char* __restrict__ wq5, const unsigned char* __restrict__ wq3g,
    const float* __restrict__ gridf, const __half* __restrict__ xh,
    __half* __restrict__ qkv_row, __half* __restrict__ gate_row)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  float acc = 0.f;
  if (warp < 10240) {
    Q5ROW(qkv_row)
  } else {
    const int r = warp - 10240;
    IQ3ROWP(wq3g + (size_t)r * 1960u, 20, xh, (gate_row[r] = (__half)acc))
  }
}

// ---- head8: head Q5 GEMV (248320 rows, raw layout) ----
extern "C" __global__ void __launch_bounds__(256) head8(
    const unsigned char* __restrict__ wq5, const __half* __restrict__ xh, __half* __restrict__ logits)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  float acc = 0.f;
  Q5ROW(logits)
}

// ---- ffn8: gate+up IQ3 packed GEMVs + silu-mul ----
extern "C" __global__ void __launch_bounds__(256) ffn8(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const float* __restrict__ gridf, const __half* __restrict__ hhx, __half* __restrict__ gact)
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
  float ag = 0.f, au = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const __half* xb = hhx + (b << 8) + (lane << 3);
    const float4 xa = *(const float4*)(xb);
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #define IQ3P2(QP, SP, DP, ACC) { \
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
      (ACC) += __half2float(__hmul(__float2half(xv[0]), __float2half(wv0))) \
             + __half2float(__hmul(__float2half(xv[1]), __float2half(wv1))); \
      (ACC) += __half2float(__hmul(__float2half(xv[2]), __float2half(wv2))) \
             + __half2float(__hmul(__float2half(xv[3]), __float2half(wv3))); \
      (ACC) += __half2float(__hmul(__float2half(xv[4]), __float2half(wv4))) \
             + __half2float(__hmul(__float2half(xv[5]), __float2half(wv5))); \
      (ACC) += __half2float(__hmul(__float2half(xv[6]), __float2half(wv6))) \
             + __half2float(__hmul(__float2half(xv[7]), __float2half(wv7))); }
    IQ3P2(qg, sg, dg, ag)
    IQ3P2(qu, su, du, au)
    #undef IQ3P2
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { ag += __shfl_down_sync(FULL, ag, o); au += __shfl_down_sync(FULL, au, o); }
  if (lane == 0) gact[warp] = __hmul(hsilu_h((__half)ag), (__half)au);
}

// ---- down8: down GEMV IQ3 packed [5120, 17408] + residual ----
extern "C" __global__ void __launch_bounds__(256) down8(
    const unsigned char* __restrict__ wd, const float* __restrict__ gridf,
    const __half* __restrict__ gact, const float* __restrict__ hh, float* __restrict__ y)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  float acc = 0.f;
  IQ3ROWP(wd + (size_t)warp * 6664u, 68, gact, (y[warp] = hh[warp] + (float)((__half)acc)))
}

// ---- op38: GDN o_proj IQ3 packed ssm_out (24 blocks) ----
extern "C" __global__ void __launch_bounds__(256) op38(
    const unsigned char* __restrict__ wq3, const float* __restrict__ gridf,
    const __half* __restrict__ z, __half* __restrict__ attn_out)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  float acc = 0.f;
  IQ3ROWP(wq3 + (size_t)warp * 2352u, 24, z, (attn_out[warp] = (__half)acc))
}

// Q6_K PACKED row (212B blocks): [pad2][d2 @+2][sc16 @+4][lo128 @+20][qh64 @+148]
extern "C" __global__ void __launch_bounds__(256) aq6k8(
    const unsigned char* __restrict__ wq, const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv4,
    const float* __restrict__ gridf, const __half* __restrict__ xh,
    __half* __restrict__ qrow, __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 12288) {
    const unsigned char* rowp = wq + (size_t)warp * 4240u;
    float acc = 0.f;
    #pragma unroll 5
    for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + b*212;
      const float d = __half2float(*((const __half*)(blk+2)));
      const bool nib_hi = ((lane&15) >= 8);
      const int c2 = (lane>>2)&3;
      const unsigned int loA = *(const unsigned int*)(blk + 20 + 64*(lane>>4) + (lane&7)*8);
      const unsigned int loB = *(const unsigned int*)(blk + 20 + 64*(lane>>4) + (lane&7)*8 + 4);
      const unsigned int qhA = *(const unsigned int*)(blk + 148 + (lane>>4)*32 + (lane&3)*8);
      const unsigned int qhB = *(const unsigned int*)(blk + 148 + (lane>>4)*32 + (lane&3)*8 + 4);
      const int sc8 = (signed char)blk[4 + (lane>>1)];
      const int koff = (b << 8) + (lane << 3);
      const float4 xf0 = *(const float4*)(xh + koff);
      const __half2* h0 = (const __half2*)&xf0;
      float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]), f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
      const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const unsigned int lob = (j < 4) ? (loA >> (8*j)) : (loB >> (8*(j-4)));
        const int xl = nib_hi ? (lob >> 4) & 0xF : lob & 0xF;
        const unsigned int qhb = (j < 4) ? (qhA >> (8*j)) : (qhB >> (8*(j-4)));
        const int xh2 = ((qhb >> (c2<<1)) & 3) << 4;
        const float w = d * (float)sc8 * (float)((signed char)((xl | xh2) - 32));
        acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) qrow[warp] = (__half)acc;
  } else KVBODY8(warp)
}

extern "C" __global__ void __launch_bounds__(256) aq3k8(
    const unsigned char* __restrict__ wq, const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv4,
    const float* __restrict__ gridf, const __half* __restrict__ xh,
    __half* __restrict__ qrow, __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 12288) {
    float acc = 0.f;
    IQ3ROWP(wq + (size_t)warp * 1960u, 20, xh, (qrow[warp] = (__half)acc))
  } else KVBODY8(warp)
}

// ---- ao8: IQ3_S o_proj GEMV [5120, 6144] (RAW layout; byte/u16 loads only, safe) ----
extern "C" __global__ void __launch_bounds__(256) ao8(
    const unsigned char* __restrict__ wo, const float* __restrict__ grid512,
    const __half* __restrict__ ao_in, __half* __restrict__ attn_out)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wo + (size_t)warp * 2640u;
  float acc = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 24; ++b) {
    const unsigned char* blk = rowp + (size_t)b * 110u;
    const float d = __half2float(*((const __half*)blk));
    const int g0 = lane*2, g1 = lane*2 + 1;
    const int sraw = lane >> 2;
    const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
    const unsigned char* sgnb = blk + 74 + lane;
    const int koff = (b << 8) + (lane << 3);
    const float4 xa = *(const float4*)(ao_in + koff);
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lane);
    const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0>>3)] >> (g0&7)) & 1u) << 8));
    const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1>>3)] >> (g1&7)) & 1u) << 8));
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int g = (j < 4) ? qb0 : qb1;
      const float gv = grid512[(g << 2) + (j & 3)];
      const float sgn = ((*sgnb >> j) & 1) ? -1.f : 1.f;
      const float w = d * sc * gv * sgn;
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) attn_out[warp] = (__half)acc;
}
