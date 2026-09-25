// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-a: fully-fused GDN block (T=1), Qwen3.8-27B real dims.
// HARD RULES: flat indexing only, NO gridDim/blockDim reads (hardcoded 256 thr),
// single-array smem only (none used), nwarps/nthreads passed as needed.
// Numerics replicate stock tinygrad dtype flow: half(xh) * half(w) fp32-acc GEMVs
// with half outputs; fp32 alpha/beta GEMVs; fp32 conv/scan; half silu on gates.
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
__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- K0: RMS(x) -> xh = half(x*r*nw) ; 1 CTA, per-warp redundant reduce ----
extern "C" __global__ void __launch_bounds__(256) k2_scan(
    const float* __restrict__ conv_in, float* __restrict__ conv_out,
    const __half* __restrict__ qkv_row, const __half* __restrict__ gate_row,
    const float* __restrict__ convw, const float* __restrict__ dtb, const float* __restrict__ ssm_a,
    const float* __restrict__ alpharaw, const float* __restrict__ betaraw,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    float* __restrict__ rec, float* __restrict__ core)
{
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float al = expf(softplus_f(alpharaw[h] + dtb[h]) * ssm_a[h]);
  const float be = sig_f(betaraw[h]);
  const int kh = h % 16;  // tinygrad repeat=TILE: v-head h <- k-head h%nk (np.repeat would be h/3)
  const int qc0 = kh*128, kc0 = QDIM + kh*128, vc0 = 2*QDIM + h*128;
  // phase A (per warp, redundant): q/k/v slices
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
  // scan: 8 warps x 16 v-rows, exact gdn_scan_m T=1 semantics
  const float* vbuf = v + h*128;
  #pragma unroll
  for (int vv = 0; vv < 16; ++vv) {
    const int v_idx = warp*16 + vv;
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
  // new conv state (ping-pong dst; disjoint from conv_in reads)
  {
    const int tid = (h << 8) + threadIdx.x;
    for (int i = tid; i < 3*CONV_CH; i += (NVH << 8)) {
      const int row = i / CONV_CH, c = i - row*CONV_CH;
      conv_out[i] = (row < 2) ? conv_in[(row+1)*CONV_CH + c] : __half2float(qkv_row[c]);
    }
  }
}

// ---- K2b: gated norm z = half(ssm_norm(core) * hsilu(gate)) ----
