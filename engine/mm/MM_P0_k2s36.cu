// MM_P0 D5: mm_k2s36 — the k2s GDN recurrence-core bench at qwen35moe dims.
//
// Today's production k2s (Qwen3.8): 48 heads x [128 v][128 k] fp32 state,
// CTA=head (256 thr = 8 warps), warp covers 16 v-rows, lanes cover 128 k.
// Qwen3.6-35B-A3B: 32 heads x [128 v][128 k] = 2 MiB/layer (SMALLER than
// today's 3 MiB — MM_PLAN's "32x-state sleeper" premise is tested here).
//
// Core math + op order VERBATIM from k2s11 (silu gates, norm, delta-rule
// update, xor-shfl trees, sequential t-chain t reads t-1's slab). conv/z
// stripped (not the sleeper). Outputs rec_out + core -> nothing elidable.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define NVH 32
#ifndef TMAX
#define TMAX 11
#endif
#define EPS_Q 1e-6f
#define ISQ128 0.08838834764831845f

__device__ __forceinline__ float sig_f(float x){ return 1.0f/(1.0f+exp2f(x*(-1.4426950408889634f))); }
__device__ __forceinline__ float softplus_f(float x){ return fmaxf(x,0.f)+log1pf(expf(-fabsf(x))); }

extern "C" __global__ void __launch_bounds__(256) mm_k2s36(
    const __half* __restrict__ qkv,     // [T][NVH*384] per head: q[128] k[128] v[128]
    const float* __restrict__ abdt,     // [T][NVH*3] (a, b, dt)
    const float* __restrict__ ssm_a,    // [NVH]
    const float* __restrict__ dtb,      // [NVH]
    const float* __restrict__ rec_in,   // [NVH][128][128] fp32 live state
    float* __restrict__ rec_out,        // [T][NVH][128][128] fp32 per-t slabs
    float* __restrict__ core)           // [T][NVH][128]
{
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int c0 = h*384;
  const float* st = rec_in + (size_t)h*16384;
  for (int t = 0; t < TMAX; ++t) {
    const float al = expf(softplus_f(abdt[t*NVH*3 + h*3 + 2] + dtb[h]) * ssm_a[h]);
    const float be = sig_f(abdt[t*NVH*3 + h*3 + 1]);
    const __half* xrow = qkv + (size_t)t*NVH*384 + c0;
    float qr[4], kr[4], qss = 0.f, kss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float q2 = __half2float(xrow[lane*4 + j]);
      const float k2 = __half2float(xrow[128 + lane*4 + j]);
      const float sq = q2*sig_f(q2), sk = k2*sig_f(k2);
      qr[j] = sq; kr[j] = sk;
      qss += sq*sq; kss += sk*sk;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { qss += __shfl_xor_sync(FULL, qss, o); kss += __shfl_xor_sync(FULL, kss, o); }
    const float qn = (1.0f / fmaxf(sqrtf(qss), EPS_Q)) * ISQ128;
    const float kn = 1.0f / fmaxf(sqrtf(kss), EPS_Q);
    #pragma unroll
    for (int j = 0; j < 4; ++j) { qr[j] *= qn; kr[j] *= kn; }
    float* rec_out_t = rec_out + (size_t)t*NVH*16384 + (size_t)h*16384;
    #pragma unroll
    for (int vv2 = 0; vv2 < 16; ++vv2) {
      const int v_idx = warp*16 + vv2;
      const float v2 = __half2float(xrow[256 + v_idx]);
      const float sv = v2*sig_f(v2);
      const float* srow_in = st + (size_t)v_idx*128;
      float* srow = rec_out_t + (size_t)v_idx*128;
      float s0 = srow_in[lane*4]*al, s1 = srow_in[lane*4+1]*al, s2 = srow_in[lane*4+2]*al, s3 = srow_in[lane*4+3]*al;
      const float k0 = kr[0], k1 = kr[1], k2 = kr[2], k3 = kr[3];
      float kd = s0*k0 + s1*k1 + s2*k2 + s3*k3;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) kd += __shfl_xor_sync(FULL, kd, o);
      const float dl = (sv - kd) * be;
      s0 += dl*k0; s1 += dl*k1; s2 += dl*k2; s3 += dl*k3;
      srow[lane*4] = s0; srow[lane*4+1] = s1; srow[lane*4+2] = s2; srow[lane*4+3] = s3;
      const float q0 = qr[0], q1 = qr[1], q2 = qr[2], q3 = qr[3];
      float qd = s0*q0 + s1*q1 + s2*q2 + s3*q3;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) qd += __shfl_xor_sync(FULL, qd, o);
      if (lane == 0) core[(size_t)t*NVH*128 + h*128 + v_idx] = qd;
    }
    st = rec_out_t;
  }
}
