// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P2 pSCAN-M: GDN prefill — per-block conv + delta-rule scan over M=16
// sequential steps in ONE kernel (the k2s3 T=3 pattern extended to 16).
// Differences vs k2s3 (all documented in P2_PREFILL_ATTN.md):
//   1. STATE IN REGISTERS: each warp owns 16 v_idx rows (v_idx = warp*16+vv2),
//      lane owns k-dims [lane*4, lane*4+4) -> sreg[16][4] = 64 floats/thread,
//      loaded ONCE from the live rec slot before the t loop and stored back
//      ONCE after it. The k2s3 per-step GLOBAL rec slot roundtrips (which at
//      T=16 would move 4.7GB/chunk through global) are GONE; per-step op order
//      within the delta rule is byte-identical (loads/stores replaced by reg
//      reads/writes of the same values in the same order).
//   2. Conv window generalized: row(t-k) = qkv16[(t-k)*10240] for t-k >= 0,
//      live[(t-k+3)*CONV_CH] for t-k < 0 (the k2s3 t=0/1/2 special cases are
//      exactly this formula at T=3).
//   3. Intermediate conv_b slot writes are dead within the kernel (steps >= 1
//      read the raw qkv16 rows) -> only the FINAL window (rows 13,14,15) is
//      written back to the live slot at t == TROWS-1. In-place live is safe:
//      all live reads happen at t <= 2, the write at t = 15, with
//      __syncthreads() barriers inside every t iteration.
//   4. araw16/braw16 are [16][48] (per-chunk-row alpha/beta from pfk_ab16);
//      z output is z16[16][6144] fp16.
// Everything else (norms, softplus/sigmoid, ISQ128 scaling, the kd/dl/state
// update/qd order, the z gate epilogue) is k2s3 VERBATIM.
// grid (48,1,1) 256thr; LAWS: no gridDim reads, full masks, unrolled vv2
// (static reg indices), sequential t loop, hardcoded sizes.
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
#define TROWS 16
#define EPS_N 1e-6f
#define EPS_Q 1e-6f
#define ISQ128 0.08838834764831845f

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x))); }

extern "C" __global__ void __launch_bounds__(256) pfs16(
    float* __restrict__ conv_live, float* __restrict__ rec_live,
    const __half* __restrict__ qkv16, const __half* __restrict__ gate16,
    const float* __restrict__ convw, const float* __restrict__ dtb, const float* __restrict__ ssm_a,
    const float* __restrict__ a16, const float* __restrict__ b16,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    float* __restrict__ core, const float* __restrict__ snw, __half* __restrict__ z16)
{
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kh = h % 16;
  const int qc0 = kh*128, kc0 = QDIM + kh*128, vc0 = 2*QDIM + h*128;

  // register state slice (loaded once)
  float sreg[16][4];
  #pragma unroll
  for (int vv2 = 0; vv2 < 16; ++vv2) {
    const float* srow_in = rec_live + (size_t)h*VDIM*KDIM + (size_t)(warp*16 + vv2)*KDIM;
    #pragma unroll
    for (int j = 0; j < 4; ++j) sreg[vv2][j] = srow_in[lane*4 + j];
  }

  for (int t = 0; t < TROWS; ++t) {
    const float al = expf(softplus_f(a16[t*48+h] + dtb[h]) * ssm_a[h]);
    const float be = sig_f(b16[t*48+h]);
    const __half* qrow_t = qkv16 + (size_t)t*10240;
    float qr[4], kr[4], qss = 0.f, kss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int qq = qc0 + lane*4 + j, kk = kc0 + lane*4 + j, vv = vc0 + lane*4 + j;
      const float q2 = __half2float(qrow_t[qq]), k2 = __half2float(qrow_t[kk]), v2 = __half2float(qrow_t[vv]);
      // window rows t-3, t-2, t-1 (live fallback for negative rows — k2s3 general form)
      const float w0q = (t >= 3) ? __half2float(qkv16[(size_t)(t-3)*10240 + qq]) : conv_live[(size_t)(t+0)*CONV_CH + qq];
      const float w1q = (t >= 2) ? __half2float(qkv16[(size_t)(t-2)*10240 + qq]) : conv_live[(size_t)(t+1)*CONV_CH + qq];
      const float w2q = (t >= 1) ? __half2float(qkv16[(size_t)(t-1)*10240 + qq]) : conv_live[(size_t)(t+2)*CONV_CH + qq];
      const float w0k = (t >= 3) ? __half2float(qkv16[(size_t)(t-3)*10240 + kk]) : conv_live[(size_t)(t+0)*CONV_CH + kk];
      const float w1k = (t >= 2) ? __half2float(qkv16[(size_t)(t-2)*10240 + kk]) : conv_live[(size_t)(t+1)*CONV_CH + kk];
      const float w2k = (t >= 1) ? __half2float(qkv16[(size_t)(t-1)*10240 + kk]) : conv_live[(size_t)(t+2)*CONV_CH + kk];
      const float w0v = (t >= 3) ? __half2float(qkv16[(size_t)(t-3)*10240 + vv]) : conv_live[(size_t)(t+0)*CONV_CH + vv];
      const float w1v = (t >= 2) ? __half2float(qkv16[(size_t)(t-2)*10240 + vv]) : conv_live[(size_t)(t+1)*CONV_CH + vv];
      const float w2v = (t >= 1) ? __half2float(qkv16[(size_t)(t-1)*10240 + vv]) : conv_live[(size_t)(t+2)*CONV_CH + vv];
      float sq = w0q*convw[qq*4+0] + w1q*convw[qq*4+1] + w2q*convw[qq*4+2] + q2*convw[qq*4+3];
      float sk = w0k*convw[kk*4+0] + w1k*convw[kk*4+1] + w2k*convw[kk*4+2] + k2*convw[kk*4+3];
      float sv = w0v*convw[vv*4+0] + w1v*convw[vv*4+1] + w2v*convw[vv*4+2] + v2*convw[vv*4+3];
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
      float s0 = sreg[vv2][0]*al, s1 = sreg[vv2][1]*al, s2 = sreg[vv2][2]*al, s3 = sreg[vv2][3]*al;
      const float k0 = kr[0]*kn, k1 = kr[1]*kn, k2 = kr[2]*kn, k3 = kr[3]*kn;
      float kd = s0*k0 + s1*k1 + s2*k2 + s3*k3;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) kd += __shfl_xor_sync(FULL, kd, o);
      const float dl = (vbuf[v_idx] - kd) * be;
      s0 += dl*k0; s1 += dl*k1; s2 += dl*k2; s3 += dl*k3;
      sreg[vv2][0] = s0; sreg[vv2][1] = s1; sreg[vv2][2] = s2; sreg[vv2][3] = s3;
      const float q0 = qr[0]*qn, q1 = qr[1]*qn, q2b = qr[2]*qn, q3 = qr[3]*qn;
      float qd = s0*q0 + s1*q1 + s2*q2b + s3*q3;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) qd += __shfl_xor_sync(FULL, qd, o);
      if (lane == 0) core[h*128 + v_idx] = qd;
    }
    __syncthreads();
    // final conv window write (rows 13,14,15) at the last step only
    if (t == TROWS - 1) {
      const int tid = (h << 8) + threadIdx.x;
      for (int i = tid; i < 3*CONV_CH; i += (NVH << 8)) {
        const int row = i / CONV_CH, c = i - row*CONV_CH;
        conv_live[i] = __half2float(qkv16[(size_t)(13 + row)*10240 + c]);
      }
    }
    {
      float zz = 0.f;
      for (int i = lane; i < 128; i += 32) { const float c = core[h*128+i]; zz += c*c; }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) zz += __shfl_xor_sync(FULL, zz, o);
      const float rz = rsqrtf(zz/128 + EPS_N);
      if (threadIdx.x < 128) {
        const int j = threadIdx.x;
        const __half g = gate16[(size_t)t*6144 + h*128 + j];
        z16[(size_t)t*6144 + h*128 + j] = __float2half((core[h*128+j]*rz*snw[j]) * __half2float(
          __hmul(g, hrcp((__half)1.0f + hexp2(__hmul(g, __float2half(-1.4423828125f)))))));
      }
    }
    __syncthreads();
  }

  // store the register state back to the live slot (once)
  #pragma unroll
  for (int vv2 = 0; vv2 < 16; ++vv2) {
    float* srow = rec_live + (size_t)h*VDIM*KDIM + (size_t)(warp*16 + vv2)*KDIM;
    #pragma unroll
    for (int j = 0; j < 4; ++j) srow[lane*4 + j] = sreg[vv2][j];
  }
}
