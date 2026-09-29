// MM P34: k2s36 -- the REAL recurrent gated-delta-rule port at qwen35moe dims
// (32 v-heads x [128 v][128 k] fp32 state; k-head = v-head>>1) + the GATED
// RMSNorm epilogue (plain w, silu(z)) folded in. CTA = v-head, 256 thr,
// warp owns 16 v-rows, lane covers k-dims {l*4+j}.
//
// Math (verbatim the torch_recurrent_gated_delta_rule reference):
//   q,k = post-conv silu'd rows; qn = 1/sqrt(sum(q^2)+eps); qr = (q*qn)*ISQ128
//   kn = 1/sqrt(sum(k^2)+eps); kr = k*kn
//   beta = 1/(1+expf(b)); al = expf(alog*softplus(a+dtb))  [ssm_a = -exp(A_log) folded]
//   per v-row: s = S*al (decay on read); kd = tree(s*kr); dl = (v-kd)*beta;
//              s += dl*kr; S=s; qd = tree(s*qr)
//   epilogue: yss = sum warp-asc (sum vv2-asc qd^2); ms = yss*(1/128);
//             rstd = 1/sqrtf(ms+eps); out = ((qd*rstd)*w)*silu(z)
// eps = 9.999999974752427e-07f. IEEE sqrtf + division. -fmad=false.
#ifndef TMAX
#define TMAX 3
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(k2s36_, TMAX)
#define NVH 32
#define EPS_F 9.999999974752427e-07f
#define ISQ128 0.08838834764831845f

extern "C" __global__ void __launch_bounds__(256) KSYM(
    const float* __restrict__ qkvs,     // [T][8192] post-conv silu'd
    const float* __restrict__ ab,       // [T][64] (a logits | b logits)
    const float* __restrict__ alog,     // [32]
    const float* __restrict__ dtb,      // [32]
    const float* __restrict__ wn,       // [128] ssm_norm weight
    const float* __restrict__ z,        // [T][4096]
    float* __restrict__ S,              // [32][128][128] fp32 (in-place)
    float* __restrict__ y)              // [T][4096] post gated-norm
{
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* st = S + (size_t)h*16384;
  __shared__ float wsum[8];
  __shared__ float ysm[128];
  __shared__ float rstd_s;
  for (int t = 0; t < TMAX; ++t) {
    const float* xq = qkvs + (size_t)t*8192 + (size_t)(h&15)*128;   // TILED: head t pairs k-head t%16
    const float* xk = qkvs + (size_t)t*8192 + 2048 + (size_t)(h&15)*128;
    const float* xv = qkvs + (size_t)t*8192 + 4096 + (size_t)h*128;
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
    const float al = expf(alog[h] * spv);   // ssm_a = -exp(A_log) FOLDED in the GGUF
    // v-rows: warp w owns rows 16w..16w+15 (sequential vv2 asc)
    float pw = 0.f;
    #pragma unroll
    for (int vv2 = 0; vv2 < 16; ++vv2) {
      const int vi = warp*16 + vv2;
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
      if (lane == 0) { ysm[vi] = qd; }
      pw += qd*qd;
    }
    if (lane == 0) wsum[warp] = pw;
    __syncthreads();
    if (threadIdx.x == 0) {
      float yss = wsum[0];
      #pragma unroll
      for (int w_ = 1; w_ < 8; ++w_) yss += wsum[w_];
      const float ms = yss * (1.0f/128.0f);
      rstd_s = 1.0f / sqrtf(ms + EPS_F);
    }
    __syncthreads();
    const float rstd = rstd_s;
    const int d = threadIdx.x;
    if (d < 128) {
      const float qd = ysm[d];
      const float zg = z[(size_t)t*4096 + h*128 + d];
      const float sg = 1.0f/(1.0f + expf(-zg));
      y[(size_t)t*4096 + h*128 + d] = ((qd*rstd)*wn[d]) * (zg*sg);
    }
    __syncthreads();
  }
}
