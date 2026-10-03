// MM SESSION E (item 3): k2s36h_{TMAX} -- the SPLIT-ROW gated-delta-rule
// scan. Stock k2s36 = 1 CTA per head (32 CTAs -- 60% of the GPU idle at
// 82 SMs; the t-chain is serial so ROWS are the only parallel dimension
// short of the full WY chunked reformulation). This port splits the 128
// v-rows across 2 CTAs (grid 2x32=64): per-row math VERBATIM stock (same
// loads, same op order); the ONLY numerics change is the output-norm sum
// regrouping (yss = half0_sum + half1_sum instead of the single 8-warp
// serial sum) -- Tier-2 reassociation, the F-bank/rebase16 class.
// The gated-RMSNorm epilogue moves to k2nz36 (cross-CTA by construction):
//   k2s36h writes raw qd -> YQB[t][4096] + per-(t,head,half) yss -> YSSB
//   k2nz36  applies ((qd*rstd)*wn)*silu(z) per (t,head) -- same formula
//           and op order as the stock fused epilogue.
// Buffers: YQB [TMAX][4096] f32, YSSB [TMAX][64] f32 (dedicated scratch).
#ifndef TMAX
#define TMAX 256
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(k2s36h_, TMAX)
#define NVH 32
#define EPS_F 9.999999974752427e-07f
#define ISQ128 0.08838834764831845f
#define ROWS_PER_CTA 64
#define ROWS_PER_WARP 8

extern "C" __global__ void __launch_bounds__(256) KSYM(
    const float* __restrict__ qkvs,     // [T][8192] post-conv silu'd
    const float* __restrict__ ab,       // [T][64]
    const float* __restrict__ alog,     // [32]
    const float* __restrict__ dtb,      // [32]
    float* __restrict__ S,              // [32][128][128] fp32 (in-place)
    float* __restrict__ YQB,            // [T][4096] raw qd
    float* __restrict__ YSSB)           // [T][64] (half*32+h) yss partials
{
  const int h = blockIdx.x >> 1;              // head
  const int half = blockIdx.x & 1;            // row half
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* st = S + (size_t)h * 16384 + (size_t)half * 64 * 128;
  __shared__ float wsum[8];
  for (int t = 0; t < TMAX; ++t) {
    const float* xq = qkvs + (size_t)t * 8192 + (size_t)(h & 15) * 128;
    const float* xk = qkvs + (size_t)t * 8192 + 2048 + (size_t)(h & 15) * 128;
    const float* xv = qkvs + (size_t)t * 8192 + 4096 + (size_t)h * 128
                    + half * ROWS_PER_CTA;
    float qr[4], kr[4], qss = 0.f, kss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float q2 = xq[lane*4 + j];
      const float k2 = xk[lane*4 + j];
      qr[j] = q2; kr[j] = k2;
      qss += q2*q2; kss += k2*k2;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { qss += __shfl_xor_sync(0xffffffffu, qss, o); kss += __shfl_xor_sync(0xffffffffu, kss, o); }
    const float qn = 1.0f / sqrtf(qss + EPS_F);
    const float kn = 1.0f / sqrtf(kss + EPS_F);
    #pragma unroll
    for (int j = 0; j < 4; ++j) { qr[j] = qr[j]*qn; qr[j] = qr[j]*ISQ128; kr[j] = kr[j]*kn; }
    const float be = 1.0f/(1.0f + expf(-ab[t*64 + 32 + h]));
    const float spv = fmaxf(ab[t*64 + h] + dtb[h], 0.f) + log1pf(expf(-fabsf(ab[t*64 + h] + dtb[h])));
    const float al = expf(alog[h] * spv);
    float pw = 0.f;
    #pragma unroll
    for (int vv2 = 0; vv2 < ROWS_PER_WARP; ++vv2) {
      const int vi = warp * ROWS_PER_WARP + vv2;        // within the half
      const float v2 = xv[vi];
      float s0 = st[(size_t)vi*128 + lane*4+0] * al;
      float s1 = st[(size_t)vi*128 + lane*4+1] * al;
      float s2 = st[(size_t)vi*128 + lane*4+2] * al;
      float s3 = st[(size_t)vi*128 + lane*4+3] * al;
      float kd = s0*kr[0] + s1*kr[1]; kd = kd + s2*kr[2]; kd = kd + s3*kr[3];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) kd += __shfl_xor_sync(0xffffffffu, kd, o);
      const float dl = (v2 - kd) * be;
      s0 += dl*kr[0]; s1 += dl*kr[1]; s2 += dl*kr[2]; s3 += dl*kr[3];
      st[(size_t)vi*128 + lane*4+0] = s0;
      st[(size_t)vi*128 + lane*4+1] = s1;
      st[(size_t)vi*128 + lane*4+2] = s2;
      st[(size_t)vi*128 + lane*4+3] = s3;
      float qd = s0*qr[0] + s1*qr[1]; qd = qd + s2*qr[2]; qd = qd + s3*qr[3];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) qd += __shfl_xor_sync(0xffffffffu, qd, o);
      if (lane == 0) YQB[(size_t)t*4096 + h*128 + half*64 + vi] = qd;
      pw += qd*qd;
    }
    if (lane == 0) wsum[warp] = pw;
    __syncthreads();
    if (threadIdx.x == 0) {
      float yss = wsum[0];
      #pragma unroll
      for (int w_ = 1; w_ < 8; ++w_) yss += wsum[w_];
      YSSB[(size_t)t*64 + half*32 + h] = yss;
    }
    __syncthreads();
  }
}

// MM SESSION E (item 3): k2nz36 -- the gated-RMSNorm apply (the cross-CTA
// epilogue of k2s36h). One CTA per (t, head), 128 threads; yss = half0 +
// half1 (FIXED order); same formula/op order as the stock fused epilogue.
extern "C" __global__ void __launch_bounds__(128) k2nz36(
    const float* __restrict__ YQB,       // [T][4096]
    const float* __restrict__ YSSB,      // [T][64]
    const float* __restrict__ wn,        // [128]
    const float* __restrict__ z,         // [T][4096]
    float* __restrict__ y,               // [T][4096]
    const int T)
{
  const int th = blockIdx.x;
  const int t = th >> 5, h = th & 31;
  const int d = threadIdx.x;
  const float yss = YSSB[(size_t)t*64 + h] + YSSB[(size_t)t*64 + 32 + h];
  const float ms = yss * (1.0f/128.0f);
  const float rstd = 1.0f / sqrtf(ms + EPS_F);
  const float qd = YQB[(size_t)t*4096 + h*128 + d];
  const float zg = z[(size_t)t*4096 + h*128 + d];
  const float sg = 1.0f/(1.0f + expf(-zg));
  y[(size_t)t*4096 + h*128 + d] = ((qd*rstd)*wn[d]) * (zg*sg);
}
