// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 R5-K5: T=6 PROBE trunk kernels (M=6; k2s6 slots 0..4 + rec6x t=5; conv slots 0..3 + conv5x t=4 + conv6x t=5; [48][5] layout + live=slot-4 PRESERVED); k2s5 writes per-step slots 0..4 (slot 4 = live, sequential in-CTA); per-row fp op order IDENTICAL
// to the T=1/T=3 kernels -> rows bit-identical). v2 half2 cores + fat CTAs.
// Laws: flat indexing, sequential loops, no gridDim reads, per-kernel cubins.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define NVH 48
#define CONV_CH 10240
#define QDIM 2048
#define INNER 6144
#define FFN_N 17408
#define VOCAB 248320
#define EPS_N 1e-6f
#define EPS_Q 1e-6f
#define ISQ128 0.08838834764831845f

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x))); }
__device__ __forceinline__ __half hsilu_h4(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

#define LDH2(NM, XB, TS, T, KO) const uint4 NM##_raw = *(const uint4*)((XB) + (T)*(TS) + (KO)); \
  const __half2* NM = (const __half2*)&NM##_raw;

#define RED6(A0,A1,A2,A3,A4,A5) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); A3 += __shfl_down_sync(FULL, A3, o); A4 += __shfl_down_sync(FULL, A4, o); A5 += __shfl_down_sync(FULL, A5, o); } }

// per-acc element order: 2k, 2k+1 ascending == j ascending (m3 ACC3 order)
#define ACC6H2(X0, X1, X2, X3, X4, X5, WV, A0, A1, A2, A3, A4, A5) { \
  const __half2 w01 = __halves2half2(__float2half((WV)[0]), __float2half((WV)[1])); \
  const __half2 w23 = __halves2half2(__float2half((WV)[2]), __float2half((WV)[3])); \
  const __half2 w45 = __halves2half2(__float2half((WV)[4]), __float2half((WV)[5])); \
  const __half2 w67 = __halves2half2(__float2half((WV)[6]), __float2half((WV)[7])); \
  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); const __half2* x3 = (X3); const __half2* x4 = (X4); const __half2* x5 = (X5); \
  { const float2 p = __half22float2(__hmul2(x0[0], w01)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[1], w23)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[2], w45)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[3], w67)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[0], w01)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[1], w23)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[2], w45)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[3], w67)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[0], w01)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[1], w23)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[2], w45)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[3], w67)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x3[0], w01)); A3 += p.x; A3 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x3[1], w23)); A3 += p.x; A3 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x3[2], w45)); A3 += p.x; A3 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x3[3], w67)); A3 += p.x; A3 += p.y; } \
    { const float2 p = __half22float2(__hmul2(x4[0], w01)); A4 += p.x; A4 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x4[1], w23)); A4 += p.x; A4 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x4[2], w45)); A4 += p.x; A4 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x4[3], w67)); A4 += p.x; A4 += p.y; ; } \
    { const float2 p = __half22float2(__hmul2(x5[0], w01)); A5 += p.x; A5 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x5[1], w23)); A5 += p.x; A5 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x5[2], w45)); A5 += p.x; A5 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x5[3], w67)); A5 += p.x; A5 += p.y; } }

// ---- h_embed4 ----
extern "C" __global__ void __launch_bounds__(256) k2s6(
    float* __restrict__ conv_b, float* __restrict__ rec_b, float* __restrict__ conv5x, float* __restrict__ conv6x, float* __restrict__ rec6x,
    const __half* __restrict__ qkv4, const __half* __restrict__ gate4,
    const float* __restrict__ convw, const float* __restrict__ dtb, const float* __restrict__ ssm_a,
    const float* __restrict__ a4, const float* __restrict__ b4,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    float* __restrict__ core, const float* __restrict__ snw, __half* __restrict__ z4)
{
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kh = h % 16;
  const int qc0 = kh*128, kc0 = QDIM + kh*128, vc0 = 2*QDIM + h*128;
  const float* live = conv_b + 4*(3*CONV_CH);
  for (int t = 0; t < 6; ++t) {
    const float al = expf(softplus_f(a4[t*48+h] + dtb[h]) * ssm_a[h]);
    const float be = sig_f(b4[t*48+h]);
    const __half* qrow_t = qkv4 + t*10240;
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
        const float r1q = (t == 1) ? live[2*CONV_CH+qq] : __half2float(qkv4[(t-2)*10240+qq]);
        const float r1k = (t == 1) ? live[2*CONV_CH+kk] : __half2float(qkv4[(t-2)*10240+kk]);
        const float r1v = (t == 1) ? live[2*CONV_CH+vv] : __half2float(qkv4[(t-2)*10240+vv]);
        const float r2q = __half2float(qkv4[(t-1)*10240+qq]);
        const float r2k = __half2float(qkv4[(t-1)*10240+kk]);
        const float r2v = __half2float(qkv4[(t-1)*10240+vv]);
        const float r0q = (t < 3) ? live[t*CONV_CH+qq] : __half2float(qkv4[(t-3)*10240+qq]);
        const float r0k = (t < 3) ? live[t*CONV_CH+kk] : __half2float(qkv4[(t-3)*10240+kk]);
        const float r0v = (t < 3) ? live[t*CONV_CH+vv] : __half2float(qkv4[(t-3)*10240+vv]);
        sq = r0q*convw[qq*4+0] + r1q*convw[qq*4+1] + r2q*convw[qq*4+2] + q2*convw[qq*4+3];
        sk = r0k*convw[kk*4+0] + r1k*convw[kk*4+1] + r2k*convw[kk*4+2] + k2*convw[kk*4+3];
        sv = r0v*convw[vv*4+0] + r1v*convw[vv*4+1] + r2v*convw[vv*4+2] + v2*convw[vv*4+3];
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
    const float* rec_in = (t == 0) ? (rec_b + 4*(NVH*128*128) + (size_t)h*128*128)
                                   : (rec_b + (size_t)(t-1)*(NVH*128*128) + (size_t)h*128*128);
    float* rec_out = (t == 5) ? (rec6x + (size_t)h*128*128)
                                : (rec_b + (size_t)t*(NVH*128*128) + (size_t)h*128*128);
    #pragma unroll
    for (int vv2 = 0; vv2 < 16; ++vv2) {
      const int v_idx = warp*16 + vv2;
      const float* srow_in = rec_in + (size_t)v_idx*128;
      float* srow = rec_out + (size_t)v_idx*128;
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
      // R4 RACE FIX: t==4 would write conv slot 4 = the LIVE window other CTAs
      // still read at their t=0..2 (conv writes are cross-head strided — no CTA
      // owns its channels). Route t==4's window to conv5x (unread in-kernel);
      // acceptsel5k copies conv5x -> slot 4 only when m==4. rec slot writes are
      // head-local slices (race-free) and stay in-kernel for all t.
      float* dst = (t == 4) ? conv5x : (t == 5) ? conv6x : (conv_b + (size_t)t * (3*CONV_CH));
      const int tid = (h << 8) + threadIdx.x;
      for (int i = tid; i < 3*CONV_CH; i += (NVH << 8)) {
        const int row = i / CONV_CH, c = i - row*CONV_CH;
        float val;
        if (t == 0) val = (row < 2) ? live[(row+1)*CONV_CH + c] : __half2float(qkv4[0*10240 + c]);
        else if (t == 1) val = (row == 0) ? live[2*CONV_CH + c] : __half2float(qkv4[(row-1)*10240 + c]);
        else val = __half2float(qkv4[(row + t - 2)*10240 + c]);
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
        const __half g = gate4[t*6144 + h*128 + j];
        z4[t*6144 + h*128 + j] = __float2half((core[h*128+j]*rz*snw[j]) * __half2float(
          __hmul(g, hrcp((__half)1.0f + hexp2(__hmul(g, __float2half(-1.4423828125f)))))));
      }
    }
    __syncthreads();
  }
}

// ---- op38nw32_4 ----
