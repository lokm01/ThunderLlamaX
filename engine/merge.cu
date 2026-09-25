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

extern "C" __global__ void __launch_bounds__(256) k0ab(
    const float* __restrict__ x, const float* __restrict__ nw,
    const float* __restrict__ walpha, const float* __restrict__ wbeta,
    __half* __restrict__ xh, float* __restrict__ alpharaw, float* __restrict__ betaraw)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  // rms (every warp computes it redundantly; needed by all paths)
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = x[i]; ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < DIM; i += 256) xh[i] = __float2half(x[i]*r*nw[i]);
  } else {
    const int w = warp - 8;   // [0,96)
    const float* wr = (w < 48 ? walpha + (size_t)w*DIM : wbeta + (size_t)(w-48)*DIM);
    float acc = 0.f;
    for (int i = lane; i < DIM; i += 32)
      acc += wr[i] * __half2float(__float2half(x[i]*r*nw[i]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) { if (w < 48) alpharaw[w] = acc; else betaraw[w-48] = acc; }
  }
}

// ---- k2s: conv window + silu + L2 norm + T=1 scan + z-phase (k2b_z inlined) ----
extern "C" __global__ void __launch_bounds__(256) k2s(
    const float* __restrict__ conv_in, float* __restrict__ conv_out,
    const __half* __restrict__ qkv_row, const __half* __restrict__ gate_row,
    const float* __restrict__ convw, const float* __restrict__ dtb, const float* __restrict__ ssm_a,
    const float* __restrict__ alpharaw, const float* __restrict__ betaraw,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    float* __restrict__ rec, float* __restrict__ core,
    const float* __restrict__ snw, __half* __restrict__ z)
{
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float al = expf(softplus_f(alpharaw[h] + dtb[h]) * ssm_a[h]);
  const float be = sig_f(betaraw[h]);
  const int kh = h % 16;
  const int qc0 = kh*128, kc0 = QDIM + kh*128, vc0 = 2*QDIM + h*128;
  float qr[4], kr[4], qss = 0.f, kss = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int qq = qc0 + lane*4 + j, kk = kc0 + lane*4 + j, vv = vc0 + lane*4 + j;
    float sq = conv_in[qq]*convw[qq*4+0] + conv_in[CONV_CH+qq]*convw[qq*4+1]
             + conv_in[2*CONV_CH+qq]*convw[qq*4+2] + __half2float(qkv_row[qq])*convw[qq*4+3];
    float sk = conv_in[kk]*convw[kk*4+0] + conv_in[CONV_CH+kk]*convw[kk*4+1]
             + conv_in[2*CONV_CH+kk]*convw[kk*4+2] + __half2float(qkv_row[kk])*convw[kk*4+3];
    float sv = conv_in[vv]*convw[vv*4+0] + conv_in[CONV_CH+vv]*convw[vv*4+1]
             + conv_in[2*CONV_CH+vv]*convw[vv*4+2] + __half2float(qkv_row[vv])*convw[vv*4+3];
    sq *= sig_f(sq); sk *= sig_f(sk); sv *= sig_f(sv);
    qr[j] = sq; kr[j] = sk; v[h*128 + lane*4 + j] = sv;
    qss += sq*sq; kss += sk*sk;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { qss += __shfl_xor_sync(FULL, qss, o); kss += __shfl_xor_sync(FULL, kss, o); }
  const float qn = (1.0f / fmaxf(sqrtf(qss), EPS_Q)) * ISQ128;
  const float kn = 1.0f / fmaxf(sqrtf(kss), EPS_Q);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    q[h*128 + lane*4 + j] = qr[j]*qn;
    k[h*128 + lane*4 + j] = kr[j]*kn;
  }
  const float* vbuf = v + h*128;
  #pragma unroll
  for (int vv2 = 0; vv2 < 16; ++vv2) {
    const int v_idx = warp*16 + vv2;
    float* srow = rec + ((size_t)h*VDIM + v_idx)*KDIM;
    float s0 = srow[lane*4]*al, s1 = srow[lane*4+1]*al, s2 = srow[lane*4+2]*al, s3 = srow[lane*4+3]*al;
    const float k0 = kr[0]*kn, k1 = kr[1]*kn, k2 = kr[2]*kn, k3 = kr[3]*kn;
    float kd = s0*k0 + s1*k1 + s2*k2 + s3*k3;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) kd += __shfl_xor_sync(FULL, kd, o);
    const float dl = (vbuf[v_idx] - kd) * be;
    s0 += dl*k0; s1 += dl*k1; s2 += dl*k2; s3 += dl*k3;
    srow[lane*4] = s0; srow[lane*4+1] = s1; srow[lane*4+2] = s2; srow[lane*4+3] = s3;
    const float q0 = qr[0]*qn, q1 = qr[1]*qn, q2 = qr[2]*qn, q3 = qr[3]*qn;
    float qd = s0*q0 + s1*q1 + s2*q2 + s3*q3;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) qd += __shfl_xor_sync(FULL, qd, o);
    if (lane == 0) core[h*128 + v_idx] = qd;
  }
  {
    const int tid = (h << 8) + threadIdx.x;
    for (int i = tid; i < 3*CONV_CH; i += (NVH << 8)) {
      const int row = i / CONV_CH, c = i - row*CONV_CH;
      conv_out[i] = (row < 2) ? conv_in[(row+1)*CONV_CH + c] : __half2float(qkv_row[c]);
    }
  }
  // ---- z phase (was k2b_z; core[h*128..] written by THIS CTA) ----
  __syncthreads();
  {
    float zz = 0.f;
    for (int i = lane; i < 128; i += 32) { const float c = core[h*128+i]; zz += c*c; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) zz += __shfl_xor_sync(FULL, zz, o);
    const float rz = rsqrtf(zz/128 + EPS_N);
    if (threadIdx.x < 128) {
      const int j = threadIdx.x;
      z[h*128+j] = __float2half((core[h*128+j]*rz*snw[j]) * __half2float(
        __hmul(gate_row[h*128+j], hrcp((__half)1.0f + hexp2(__hmul(gate_row[h*128+j], __float2half(-1.4423828125f)))))));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) a_qkv_q6(
    const unsigned char* __restrict__ wq, const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv4,
    const float* __restrict__ gridf, const __half* __restrict__ xh,
    __half* __restrict__ qrow, __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 12288) {
    const unsigned char* rowp = wq + (size_t)warp * 4200u;
    float acc = 0.f;
    #pragma unroll 4
    for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + b*210;
      const float d = __half2float(*((const __half*)(blk+208)));
      const int lo_off = 64*(lane>>4) + (lane&7)*8;
      const bool nib_hi = ((lane&15) >= 8);
      const int c2 = (lane>>2)&3;
      const unsigned char* qhp = blk + 128 + (lane>>4)*32 + (lane&3)*8;
      const int sc8 = (signed char)blk[192 + (lane>>1)];
      const int koff = (b << 8) + (lane << 3);
      const float4 xf0 = *(const float4*)(xh + koff);
      const __half2* h0 = (const __half2*)&xf0;
      float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]), f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
      const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int lo_byte = blk[lo_off + j];
        const int xl = nib_hi ? (lo_byte >> 4) : (lo_byte & 0xF);
        const int xh2 = ((qhp[j] >> (c2<<1)) & 3) << 4;
        const float w = d * (float)sc8 * (float)((signed char)((xl | xh2) - 32));
        acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) qrow[warp] = (__half)acc;
  } else KV_BODY(warp)
}

extern "C" __global__ void __launch_bounds__(256) a_qkv_iq3(
    const unsigned char* __restrict__ wq, const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv4,
    const float* __restrict__ gridf, const __half* __restrict__ xh,
    __half* __restrict__ qrow, __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 12288) {
    const unsigned char* rowp = wq + (size_t)warp * 1960u;
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
      const __half2* hx = (const __half2*)&xa;
      float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
      const float xvv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
      const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3, db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
      #pragma unroll
      for (int j = 0; j < 8; ++j) acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(wv[j])));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) qrow[warp] = (__half)acc;
  } else KV_BODY(warp)
}

// ---- k1_q5g: GDN qkv (Q5_K 10240 rows) + gate (IQ3_XXS 6144 rows) in one launch ----
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
