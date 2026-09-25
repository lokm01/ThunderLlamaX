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
extern "C" __global__ void __launch_bounds__(256) k2s3(
    float* __restrict__ conv_b, float* __restrict__ rec_b,
    const __half* __restrict__ qkv3, const __half* __restrict__ gate3,
    const float* __restrict__ convw, const float* __restrict__ dtb, const float* __restrict__ ssm_a,
    const float* __restrict__ a3, const float* __restrict__ b3,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    float* __restrict__ core, const float* __restrict__ snw, __half* __restrict__ z3)
{
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kh = h % 16;
  const int qc0 = kh*128, kc0 = QDIM + kh*128, vc0 = 2*QDIM + h*128;
  const float* live = conv_b + 4*(3*CONV_CH);
  for (int t = 0; t < 3; ++t) {
    const float al = expf(softplus_f(a3[t*48+h] + dtb[h]) * ssm_a[h]);
    const float be = sig_f(b3[t*48+h]);
    const __half* qrow_t = qkv3 + t*10240;
    float qr[4], kr[4], qss = 0.f, kss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int qq = qc0 + lane*4 + j, kk = kc0 + lane*4 + j, vv = vc0 + lane*4 + j;
      const float q2 = __half2float(qrow_t[qq]), k2 = __half2float(qrow_t[kk]), v2 = __half2float(qrow_t[vv]);
      float sq, sk, sv;
      if (t == 0) {
        sq = live[qq]*convw[qq*4+0] + live[CONV_CH+qq]*convw[qq*4+1] + live[2*CONV_CH+qq]*convw[qq*4+2] + q2*convw[qq*4+3];
        sk = live[kk]*convw[kk*4+0] + live[CONV_CH+kk]*convw[kk*4+1] + live[2*CONV_CH+kk]*convw[kk*4+2] + k2*convw[kk*4+3];
        sv = live[vv]*convw[vv*4+0] + live[CONV_CH+vv]*convw[vv*4+1] + live[2*CONV_CH+vv]*convw[vv*4+2] + v2*convw[vv*4+3];
      } else {
        float r1q, r1k, r1v, r2q, r2k, r2v;
        if (t == 1) {
          r1q = live[2*CONV_CH+qq]; r1k = live[2*CONV_CH+kk]; r1v = live[2*CONV_CH+vv];
          r2q = __half2float(qkv3[0*10240+qq]); r2k = __half2float(qkv3[0*10240+kk]); r2v = __half2float(qkv3[0*10240+vv]);
        } else {
          r1q = __half2float(qkv3[0*10240+qq]); r1k = __half2float(qkv3[0*10240+kk]); r1v = __half2float(qkv3[0*10240+vv]);
          r2q = __half2float(qkv3[1*10240+qq]); r2k = __half2float(qkv3[1*10240+kk]); r2v = __half2float(qkv3[1*10240+vv]);
        }
        sq = live[t*CONV_CH+qq]*convw[qq*4+0] + r1q*convw[qq*4+1] + r2q*convw[qq*4+2] + q2*convw[qq*4+3];
        sk = live[t*CONV_CH+kk]*convw[kk*4+0] + r1k*convw[kk*4+1] + r2k*convw[kk*4+2] + k2*convw[kk*4+3];
        sv = live[t*CONV_CH+vv]*convw[vv*4+0] + r1v*convw[vv*4+1] + r2v*convw[vv*4+2] + v2*convw[vv*4+3];
      }
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
    const float* rec_in = (t == 0) ? (rec_b + 4*(NVH*VDIM*KDIM) + (size_t)h*VDIM*KDIM)
                                   : (rec_b + (size_t)(t-1)*(NVH*VDIM*KDIM) + (size_t)h*VDIM*KDIM);
    float* rec_out = rec_b + (size_t)t*(NVH*VDIM*KDIM) + (size_t)h*VDIM*KDIM;
    #pragma unroll
    for (int vv2 = 0; vv2 < 16; ++vv2) {
      const int v_idx = warp*16 + vv2;
      const float* srow_in = rec_in + (size_t)v_idx*KDIM;
      float* srow = rec_out + (size_t)v_idx*KDIM;
      float s0 = srow_in[lane*4]*al, s1 = srow_in[lane*4+1]*al, s2 = srow_in[lane*4+2]*al, s3 = srow_in[lane*4+3]*al;
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
      float* dst = conv_b + (size_t)t * (3*CONV_CH);
      const int tid = (h << 8) + threadIdx.x;
      for (int i = tid; i < 3*CONV_CH; i += (NVH << 8)) {
        const int row = i / CONV_CH, c = i - row*CONV_CH;
        float val;
        if (t == 0) val = (row < 2) ? live[(row+1)*CONV_CH + c] : __half2float(qkv3[0*10240 + c]);
        else if (t == 1) val = (row == 0) ? live[2*CONV_CH + c] : __half2float(qkv3[(row-1)*10240 + c]);
        else val = __half2float(qkv3[row*10240 + c]);
        dst[i] = val;
      }
    }
    __syncthreads();
    {
      float zz = 0.f;
      for (int i = lane; i < 128; i += 32) { const float c = core[h*128+i]; zz += c*c; }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) zz += __shfl_xor_sync(FULL, zz, o);
      const float rz = rsqrtf(zz/128 + EPS_N);
      if (threadIdx.x < 128) {
        const int j = threadIdx.x;
        const __half g = gate3[t*6144 + h*128 + j];
        z3[t*6144 + h*128 + j] = __float2half((core[h*128+j]*rz*snw[j]) * __half2float(
          __hmul(g, hrcp((__half)1.0f + hexp2(__hmul(g, __float2half(-1.4423828125f)))))));
      }
    }
    __syncthreads();
  }
}

// ---- op38_3: GDN o proj IQ3 packed ssm_out, 3 rows ----
