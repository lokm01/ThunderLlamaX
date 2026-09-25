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
extern "C" __global__ void __launch_bounds__(256) k0_norm(
    const float* __restrict__ x, const float* __restrict__ nw, __half* __restrict__ xh)
{
  const int lane = threadIdx.x & 31;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = x[i]; ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  for (int i = threadIdx.x; i < DIM; i += 256) xh[i] = __float2half(x[i]*r*nw[i]);
}

// ---- K1: x-GEMVs. warps: [0,10240) qkv Q5_K | [10240,16384) gate IQ3 |
//      [16384,16432) alpha f32 | [16432,16480) beta f32 ----
extern "C" __global__ void __launch_bounds__(256) k1_q5(
    const unsigned char* __restrict__ wq5, const __half* __restrict__ xh, __half* __restrict__ qkv_row)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const unsigned char* rowp = wq5 + (size_t)warp * 3520u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + b*176;
    const float d = __half2float(*((const __half*)blk));
    const float dm = __half2float(*((const __half*)(blk+2)));
    const int s = lane >> 2;
    float sc, mn;
    if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
    else { sc = (float)((blk[s+8] & 0xF) | ((blk[s] >> 6) << 4));
           mn = (float)((blk[s+8] >> 4) | ((blk[s+4] >> 6) << 4)); }
    // TRUTH-FITTED (stock tinygrad dequant): byte = (k>>6)*32 + (k&31), nibble = (k>>5)&1,
    // qh byte = k&31 bit = k>>5, scale sub = k>>5 (k = lane*8+j within the 256-block)
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
}

extern "C" __global__ void __launch_bounds__(256) k1_iq3(
    const unsigned char* __restrict__ wq3g, const float* __restrict__ gridf,
    const __half* __restrict__ xh, __half* __restrict__ gate_row)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const unsigned char* rowp = wq3g + (size_t)warp * 1960u;
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
    const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
    const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
    const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
    const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xvv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3,
                          db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(wv[j])));
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) gate_row[warp] = (__half)acc;
}

extern "C" __global__ void __launch_bounds__(256) k1_ab(
    const float* __restrict__ walpha, const float* __restrict__ wbeta,
    const __half* __restrict__ xh, float* __restrict__ alpharaw, float* __restrict__ betaraw)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp < 48) {
    const float* wr = walpha + (size_t)warp * DIM;
    float acc = 0.f;
    for (int i = lane; i < DIM; i += 32) acc += wr[i] * __half2float(xh[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) alpharaw[warp] = acc;
  } else {
    const float* wr = wbeta + (size_t)(warp-48) * DIM;
    float acc = 0.f;
    for (int i = lane; i < DIM; i += 32) acc += wr[i] * __half2float(xh[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) betaraw[warp-48] = acc;
  }
}

// ---- K2: conv window + silu + L2 norm q/k + activations + T=1 scan (in-place rec)
//      + new conv state (ping-pong dst). grid=(48,), per-warp redundant prep. ----
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
extern "C" __global__ void __launch_bounds__(256) k2b_z(
    const float* __restrict__ core, const __half* __restrict__ gate_row,
    const float* __restrict__ snw, __half* __restrict__ z)
{
  const int h = blockIdx.x;
  const int lane = threadIdx.x & 31;
  float ss = 0.f;
  for (int i = lane; i < 128; i += 32) { const float c = core[h*128+i]; ss += c*c; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/128 + EPS_N);
  if (threadIdx.x < 128) {
    const int j = threadIdx.x;
    z[h*128+j] = __float2half((core[h*128+j]*r*snw[j]) * __half2float(hsilu_h(gate_row[h*128+j])));
  }
}

// ---- K3a: o_proj Q8_0 GEMV [5120, 6144] ----
extern "C" __global__ void __launch_bounds__(256) k3a_oproj(
    const unsigned char* __restrict__ wq8, const __half* __restrict__ z, __half* __restrict__ attn_out)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wq8 + (size_t)warp * 6528u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 192; ++b) {          // sequential blocks (lane-strided start miscompiles on this dext)
    const unsigned char* blk = rowp + b*34;
    const float d = __half2float(*((const __half*)blk));
    const float w = d * (float)((signed char)blk[2+lane]);
    acc += __half2float(__hmul(z[(b<<5)+lane], __float2half(w)));
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) attn_out[warp] = (__half)acc;
}

// ---- K3m: hh = x + attn_out ; hhx = half(ffn_norm(hh)) ----
extern "C" __global__ void __launch_bounds__(256) k3m_hh(
    const float* __restrict__ x, const __half* __restrict__ attn_out, const float* __restrict__ nw2,
    float* __restrict__ hh, __half* __restrict__ hhx)
{
  const int lane = threadIdx.x & 31;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) {
    const float v = x[i] + __half2float(attn_out[i]);
    ss += v*v;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  for (int i = threadIdx.x; i < DIM; i += 256) {
    const float v = x[i] + __half2float(attn_out[i]);
    hh[i] = v;
    hhx[i] = __float2half(v*r*nw2[i]);
  }
}

// ---- K3b: ffn gate+up IQ3 GEMVs + silu-mul ----
extern "C" __global__ void __launch_bounds__(256) k3b_ffn(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const float* __restrict__ gridf, const __half* __restrict__ hhx, __half* __restrict__ gact)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= FFN_N) return;
  float ag = 0.f, au = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const __half* xb = hhx + (b << 8) + (lane << 3);
    const float4 xa = *(const float4*)(xb);
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #define IQ3ACC(ROWP, ACC) { \
      const unsigned char* blk = (ROWP) + (size_t)b * 98u; \
      const float d = __half2float(*((const __half*)blk)); \
      const unsigned short* scw = (const unsigned short*)(blk + 66); \
      const unsigned int sw = ((unsigned int)scw[2*(lane>>2)]) | (((unsigned int)scw[2*(lane>>2)+1]) << 16); \
      const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f; \
      const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu; \
      const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u; \
      const unsigned int q = ((unsigned int)blk[2 + 2*lane]) | (((unsigned int)blk[3 + 2*lane]) << 8); \
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
    IQ3ACC(wg + (size_t)warp * 1960u, ag)
    IQ3ACC(wu + (size_t)warp * 1960u, au)
    #undef IQ3ACC
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { ag += __shfl_down_sync(FULL, ag, o); au += __shfl_down_sync(FULL, au, o); }
  if (lane == 0) gact[warp] = __hmul(hsilu_h((__half)ag), (__half)au);
}

// ---- K3c: down GEMV IQ3 [5120, 17408] + residual ----
extern "C" __global__ void __launch_bounds__(256) k3c_down(
    const unsigned char* __restrict__ wd, const float* __restrict__ gridf,
    const __half* __restrict__ gact, const float* __restrict__ hh, float* __restrict__ y)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wd + (size_t)warp * 6664u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 68; ++b) {
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
    const float4 xa = *(const float4*)(gact + koff);
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
    const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
    const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
    const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
    const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3,
                          db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(wv[j])));
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) y[warp] = hh[warp] + (float)((__half)acc);
}
