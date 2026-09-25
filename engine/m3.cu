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
// R7a norms rung: one ROW per CTA (t = blockIdx.x, grid 3; was grid 1 with a
// serial 3-row t-loop). Per-row math VERBATIM -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) h_embed3(
    const unsigned char* __restrict__ emb, const float* __restrict__ grid512,
    const int* __restrict__ s0, const int* __restrict__ s1, const int* __restrict__ s2,
    float* __restrict__ x3)
{
  const int tid = threadIdx.x;
  const int t = blockIdx.x;
  const int toks[3] = { s0[0], s1[0], s2[0] };
  {
    const unsigned char* row = emb + (size_t)toks[t] * 2200u;
    float* xo = x3 + t*DIM;
    #pragma unroll
    for (int i = 0; i < 20; ++i) {
      const unsigned char* blk = row + i*110;
      const int e = i*256 + tid;
      const float d = __half2float(*((const __half*)blk));
      const int g = tid >> 2, j4 = tid & 3;
      const unsigned int q = (unsigned int)blk[2 + g] + ((((unsigned int)blk[66 + (g>>3)] >> (g&7)) & 1u) << 8);
      const int s8 = tid >> 5;
      const float sc = 1.0f + 2.0f*(float)((blk[106 + (s8>>1)] >> ((s8&1)<<2)) & 0xF);
      const float sgn = ((blk[74 + (tid>>3)] >> (tid & 7)) & 1) ? -1.f : 1.f;
      xo[e] = d * sc * grid512[(q << 2) + j4] * sgn;
    }
  }
}

// ---- k0n3: 3-row RMSNorm ----
// R7a norms rung: one ROW per CTA (t = blockIdx.x, grid 3). VERBATIM -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) k0n3(
    const float* __restrict__ x, const float* __restrict__ nw, __half* __restrict__ xh3)
{
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x;
  {
    const float* xr = x + t*DIM;
    float ss = 0.f;
    for (int i = lane; i < DIM; i += 32) { const float v = xr[i]; ss += v*v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    const float r = rsqrtf(ss/DIM + EPS_N);
    __half* xo = xh3 + t*DIM;
    for (int i = threadIdx.x; i < DIM; i += 256) xo[i] = __float2half(xr[i]*r*nw[i]);
  }
}

// ---- k0ab3: 3-row norm + alpha/beta GEMVs ----
// R7a norms rung: (row t, group g) per CTA — grid 39 = 13*3 (was 13 CTAs each
// serially looping all 3 rows). t = blockIdx.x % 3, g = blockIdx.x / 3; w =
// (g-1)*8 + warpInCTA covers [0,96) exactly once. VERBATIM -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) k0ab3(
    const float* __restrict__ x, const float* __restrict__ nw,
    const float* __restrict__ walpha, const float* __restrict__ wbeta,
    __half* __restrict__ xh3, float* __restrict__ a3, float* __restrict__ b3)
{
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x % 3;
  const int g = blockIdx.x / 3;
  const float* xr = x + t*DIM;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = xr[i]; ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  if (g == 0) {
    __half* xo = xh3 + t*DIM;
    for (int i = threadIdx.x; i < DIM; i += 256) xo[i] = __float2half(xr[i]*r*nw[i]);
  } else {
    const int w = (g - 1) * 8 + (threadIdx.x >> 5);
    const float* wr = (w < 48 ? walpha + (size_t)w*DIM : wbeta + (size_t)(w-48)*DIM);
    float acc = 0.f;
    for (int i = lane; i < DIM; i += 32)
      acc += wr[i] * __half2float(__float2half(xr[i]*r*nw[i]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) { if (w < 48) a3[t*48+w] = acc; else b3[t*48+(w-48)] = acc; }
  }
}

// ---- q5g8_3: qkv (Q5 raw) + gate (IQ3 packed), 3 rows ----
extern "C" __global__ void __launch_bounds__(256) q5g8_3(
    const unsigned char* __restrict__ wq5, const unsigned char* __restrict__ wq3g,
    const float* __restrict__ gridf, const __half* __restrict__ xh3,
    __half* __restrict__ qkv3, __half* __restrict__ gate3)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 10240) {
    Q5ROW3(xh3, DIM, qkv3, 10240, warp)
  } else {
    const int r = warp - 10240;
    IQ3ROWP3(wq3g + (size_t)r * 1960u, 20, xh3, DIM, gate3, 6144, r)
  }
}

// ---- k2s3: conv+silu+L2+delta-scan 3 sequential steps + per-step rec/conv slots + z3 ----
// conv_b points at block blk's [5][3*CONV_CH] region of conv4 [48][5][3*CONV_CH]:
// slot 4 = live input window, slots 0..2 = per-step output windows (3 unused).
// rec_b same for rec4 [48][5][48*128*128]. No scalar args (empty-sig vals gotcha).
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
extern "C" __global__ void __launch_bounds__(256) op38_3(
    const unsigned char* __restrict__ wq3, const float* __restrict__ gridf,
    const __half* __restrict__ z3, __half* __restrict__ attn_out3)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  IQ3ROWP3(wq3 + (size_t)warp * 2352u, 24, z3, INNER, attn_out3, DIM, warp)
}

// ---- k3ao3: GDN o proj Q8_0 raw ssm_out, 3 rows ----
extern "C" __global__ void __launch_bounds__(256) k3ao3(
    const unsigned char* __restrict__ wq8, const __half* __restrict__ z3, __half* __restrict__ attn_out3)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wq8 + (size_t)warp * 6528u;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 192; ++b) {
    const unsigned char* blk = rowp + b*34;
    const float d = __half2float(*((const __half*)blk));
    const float w = d * (float)((signed char)blk[2+lane]);
    const __half wh = __float2half(w);
    a0 += __half2float(__hmul(z3[0*INNER + (b<<5)+lane], wh));
    a1 += __half2float(__hmul(z3[1*INNER + (b<<5)+lane], wh));
    a2 += __half2float(__hmul(z3[2*INNER + (b<<5)+lane], wh));
  }
  RED3(a0,a1,a2)
  if (lane == 0) { attn_out3[0*DIM+warp] = (__half)a0; attn_out3[1*DIM+warp] = (__half)a1; attn_out3[2*DIM+warp] = (__half)a2; }
}

// ---- hh3: 3-row residual merge + ffn norm ----
// R7a norms rung: one ROW per CTA (t = blockIdx.x, grid 3). VERBATIM -> bit-identical.
extern "C" __global__ void __launch_bounds__(256) hh3(
    const float* __restrict__ x, const __half* __restrict__ attn_out3, const float* __restrict__ nw2,
    float* __restrict__ hh3, __half* __restrict__ hhx3)
{
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x;
  const float* xr = x + t*DIM;
  const __half* ar = attn_out3 + t*DIM;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = xr[i] + __half2float(ar[i]); ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  float* ho = hh3 + t*DIM; __half* hx = hhx3 + t*DIM;
  for (int i = threadIdx.x; i < DIM; i += 256) {
    const float v = xr[i] + __half2float(ar[i]);
    ho[i] = v;
    hx[i] = __float2half(v*r*nw2[i]);
  }
}

// ---- ffn8_3: gate+up IQ3 packed GEMVs + silu-mul, 3 rows ----
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
extern "C" __global__ void __launch_bounds__(256) down8_3(
    const unsigned char* __restrict__ wd, const float* __restrict__ gridf,
    const __half* __restrict__ gact3, const float* __restrict__ hh3, float* __restrict__ y3)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wd + (size_t)warp * 6664u;
  const unsigned short* qsp = (const unsigned short*)(rowp);
  const unsigned int* scp = (const unsigned int*)(rowp + 64*68);
  const unsigned short* dpp = (const unsigned short*)(rowp + 96*68);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 68; ++b) {
    const float d = __half2float(__ushort_as_half(dpp[b]));
    const unsigned int sw = scp[8*b + (lane>>2)];
    const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
    const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu;
    const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
    const unsigned int q = qsp[32*b + lane];
    const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2)));
    const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2)));
    const int koff = (b << 8) + (lane << 3);
    LD8(xv0, gact3, FFN_N, 0, koff) LD8(xv1, gact3, FFN_N, 1, koff) LD8(xv2, gact3, FFN_N, 2, koff)
    const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
    const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
    const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
    const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
    float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3,
                    db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
    ACC3(xv0, xv1, xv2, wv)
  }
  RED3(a0,a1,a2)
  if (lane == 0) {
    y3[0*DIM+warp] = hh3[0*DIM+warp] + (float)((__half)a0);
    y3[1*DIM+warp] = hh3[1*DIM+warp] + (float)((__half)a1);
    y3[2*DIM+warp] = hh3[2*DIM+warp] + (float)((__half)a2);
  }
}
