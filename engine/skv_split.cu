// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W3-100k: split-KV attention for ctx 100k. Replaces aattn3 (T=3 probe) and
// a_attn/aattn_d (T=1 trunk/draft) when the l-loop becomes BW-bound. THREE kernels:
//   KPRE: per-head q-norm + partial-rope -> qw workspace (fp32, qe*0.0625 folded);
//         k-norm + rope + fp16 KV append at pos..pos+ROWS-1 (verbatim aattn3 phase-1).
//   K1S : grid (4 kv-groups x S splits). GQA-shared: ONE KV read serves all 6 q-heads
//         x ROWS rows. 8 warps lockstep over l (stride 1, tile-unrolled); warp w owns
//         rows w, w+8, (w+16); per-row online softmax in registers; partials
//         (m, s, acc[256]) to workspace. Empty split -> identity partials.
//   K2S : grid 24 heads; combine S partials sequentially (fixed order, same epilogue
//         structure as aattn3), sigmoid gate, write ao.
// Per-row op order matches aattn3/a_attn exactly (Tier-1: T=3 rows bit-identical to
// T=1 rows). Laws: sequential loops, FULL-mask shuffles only on full warps, flat
// indexing, naturally-aligned float4 loads (KV rows are 512B-aligned, lane*16B).
// -DKNAME -DCTXK -DROWS (3|1) -DS (splits, power of 2) -DCH (chunk = CTXK/S) -DUNROLL
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define EPS_N 1e-6f
#define RMAX (6*ROWS)

#define LDH8(NM, P) float NM[8]; { \
  const float4 xa = *(const float4*)((P)); \
  const __half2* hx = (const __half2*)&xa; \
  float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]); \
  NM[0]=f0.x; NM[1]=f0.y; NM[2]=f1.x; NM[3]=f1.y; NM[4]=f2.x; NM[5]=f2.y; NM[6]=f3.x; NM[7]=f3.y; }

// ============================== KPRE ==============================
extern "C" __global__ void __launch_bounds__(256) KPRE(
    const __half* __restrict__ qrow, const __half* __restrict__ krow, const __half* __restrict__ vrow,
    const float* __restrict__ qnw, const float* __restrict__ knw, const float* __restrict__ freqs,
    __half* __restrict__ kv, const int* __restrict__ pos_slot, float* __restrict__ qw)
{
  const int h = blockIdx.x;
  const int kvh = h / 6;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int d = threadIdx.x;
  __shared__ float sm[768];
  const int pos = pos_slot[0];
  for (int t = 0; t < ROWS; ++t) {
    const float ang = (d < 64) ? (float)(pos + t) * freqs[d & 31] : 0.0f;
    const float cs = cosf(ang), sn = sinf(ang);
    float qv = __half2float(qrow[(ROWS==1 ? 0 : t*12288) + h*512 + d]);
    {
      float ss = qv*qv;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
      if (lane == 0) sm[warp] = ss;
      __syncthreads();
      if (warp == 0 && lane < 8) { float v = sm[lane]; for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffu, v, o); if (lane == 0) sm[0] = v; }
      __syncthreads();
      const float r = rsqrtf(sm[0]/256.f + EPS_N);
      qv = __half2float(__float2half(qv*r)) * qnw[d];
      sm[256 + d] = qv;
      __syncthreads();
      const float qe = (d < 32) ? qv*cs - sm[256 + d + 32]*sn
                     : (d < 64) ? qv*cs + sm[256 + d - 32]*sn
                     : qv;
      qw[(ROWS==1 ? 0 : t*24)*256 + h*256 + d] = qe * 0.0625f;
    }
    {
      float kvv = __half2float(krow[(ROWS==1 ? 0 : t*1024) + kvh*256 + d]);
      float ss = kvv*kvv;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
      if (lane == 0) sm[warp] = ss;
      __syncthreads();
      if (warp == 0 && lane < 8) { float v = sm[lane]; for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffu, v, o); if (lane == 0) sm[0] = v; }
      __syncthreads();
      const float r = rsqrtf(sm[0]/256.f + EPS_N);
      kvv = __half2float(__float2half(kvv*r)) * knw[d];
      sm[256 + d] = kvv;
      __syncthreads();
      const float ko = (d < 32) ? kvv*cs - sm[256 + d + 32]*sn
                     : (d < 64) ? kvv*cs + sm[256 + d - 32]*sn
                     : kvv;
      if ((h % 6) == 0) {
        __half* Kc = kv + (size_t)kvh * (CTXK*256);
        __half* Vc = kv + (size_t)(4 + kvh) * (CTXK*256);
        Kc[(size_t)(pos+t)*256 + d] = __float2half(ko);
        Vc[(size_t)(pos+t)*256 + d] = vrow[(ROWS==1 ? 0 : t*1024) + kvh*256 + d];
      }
    }
    __syncthreads();
  }
}

// ============================== K1S ==============================
extern "C" __global__ void __launch_bounds__(256) K1S(
    const __half* __restrict__ kv, const float* __restrict__ qw, const int* __restrict__ pos_slot,
    float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
{
  const int g = blockIdx.x / S;
  const int s = blockIdx.x % S;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = pos_slot[0];
  const int l0 = s * CH;
  const int l1 = (l0 + CH) < (pos + ROWS) ? (l0 + CH) : (pos + ROWS);
  const __half* Kc = kv + (size_t)g * (CTXK*256);
  const __half* Vc = kv + (size_t)(4 + g) * (CTXK*256);
  // rows owned by this warp (sequential row list, warp-uniform)
  const int rr0 = warp;
  const int nrr = (rr0 < RMAX) ? 1 : 0;
  float qe0[8], qe1[8], qe2[8];
  float m0[1] = {-1e30f}, m1[1] = {-1e30f}, m2[1] = {-1e30f};
  float s0[1] = {0.f}, s1[1] = {0.f}, s2[1] = {0.f};
  float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0}, a2[8] = {0,0,0,0,0,0,0,0};
  // row r -> t = r % ROWS, hl = r / ROWS; qw row index = t*24 + g*6 + hl
  #define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )
  #define LDQ8(DST, R) { const float4 qa = *(const float4*)(qw + (size_t)QROWIDX(R)*256 + lane*8); \
    const float4 qb = *(const float4*)(qw + (size_t)QROWIDX(R)*256 + lane*8 + 4); \
    DST[0]=qa.x; DST[1]=qa.y; DST[2]=qa.z; DST[3]=qa.w; DST[4]=qb.x; DST[5]=qb.y; DST[6]=qb.z; DST[7]=qb.w; }
  if (rr0     < RMAX) LDQ8(qe0, rr0)
  if (rr0 + 8 < RMAX) LDQ8(qe1, rr0 + 8)
  if (rr0 +16 < RMAX) LDQ8(qe2, rr0 + 16)
  const int nr = nrr + ((rr0 + 8 < RMAX) ? 1 : 0) + ((rr0 +16 < RMAX) ? 1 : 0);
  if (nr > 0 && l0 < l1) {
    #pragma unroll UNROLL
    for (int l = l0; l < l1; ++l) {
      LDH8(kr, (Kc + (size_t)l*256 + lane*8))
      LDH8(vr, (Vc + (size_t)l*256 + lane*8))
      if (rr0 < RMAX) {
        const int t0 = rr0 % ROWS;
        if (l <= pos + t0) {
          float sc = 0.f;
          _Pragma("unroll") for (int j = 0; j < 8; ++j) sc += qe0[j] * kr[j];
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(FULL, sc, o);
          if (sc > m0[0]) { const float cor = expf(m0[0] - sc); s0[0] *= cor; _Pragma("unroll") for (int j=0;j<8;++j) a0[j] *= cor; m0[0] = sc; }
          const float p = expf(sc - m0[0]); s0[0] += p;
          _Pragma("unroll") for (int j = 0; j < 8; ++j) a0[j] += p * vr[j];
        }
      }
      if (rr0 + 8 < RMAX) {
        const int r1_ = rr0 + 8; const int t1_ = r1_ % ROWS;
        if (l <= pos + t1_) {
          float sc = 0.f;
          _Pragma("unroll") for (int j = 0; j < 8; ++j) sc += qe1[j] * kr[j];
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(FULL, sc, o);
          if (sc > m1[0]) { const float cor = expf(m1[0] - sc); s1[0] *= cor; _Pragma("unroll") for (int j=0;j<8;++j) a1[j] *= cor; m1[0] = sc; }
          const float p = expf(sc - m1[0]); s1[0] += p;
          _Pragma("unroll") for (int j = 0; j < 8; ++j) a1[j] += p * vr[j];
        }
      }
      if (rr0 +16 < RMAX) {
        const int r2_ = rr0 + 16; const int t2_ = r2_ % ROWS;
        if (l <= pos + t2_) {
          float sc = 0.f;
          _Pragma("unroll") for (int j = 0; j < 8; ++j) sc += qe2[j] * kr[j];
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(FULL, sc, o);
          if (sc > m2[0]) { const float cor = expf(m2[0] - sc); s2[0] *= cor; _Pragma("unroll") for (int j=0;j<8;++j) a2[j] *= cor; m2[0] = sc; }
          const float p = expf(sc - m2[0]); s2[0] += p;
          _Pragma("unroll") for (int j = 0; j < 8; ++j) a2[j] += p * vr[j];
        }
      }
    }
  }
  const size_t pb0 = (size_t)(g*S + s)*RMAX;
  if (rr0 < RMAX) {
    const size_t b = pb0 + rr0;
    if (lane == 0) { pm[b] = m0[0]; ps[b] = s0[0]; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a0[j];
  }
  if (rr0 + 8 < RMAX) {
    const size_t b = pb0 + rr0 + 8;
    if (lane == 0) { pm[b] = m1[0]; ps[b] = s1[0]; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a1[j];
  }
  if (rr0 +16 < RMAX) {
    const size_t b = pb0 + rr0 + 16;
    if (lane == 0) { pm[b] = m2[0]; ps[b] = s2[0]; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a2[j];
  }
}

// ============================== K2S ==============================
extern "C" __global__ void __launch_bounds__(256) K2S(
    const float* __restrict__ pm, const float* __restrict__ ps, const float* __restrict__ pA,
    const __half* __restrict__ qrow, __half* __restrict__ ao)
{
  const int h = blockIdx.x;
  const int g = h / 6, hl = h % 6;
  const int d = threadIdx.x;
  for (int t = 0; t < ROWS; ++t) {
    const int r = hl*ROWS + t;
    const size_t pb = (size_t)(g*S)*RMAX + r;
    float M = -1e30f;
    for (int s2 = 0; s2 < S; ++s2) M = fmaxf(M, pm[pb + (size_t)s2*RMAX]);
    float out = 0.f, Ssum = 0.f;
    for (int s2 = 0; s2 < S; ++s2) {
      const float ex = expf(pm[pb + (size_t)s2*RMAX] - M);
      Ssum += ps[pb + (size_t)s2*RMAX] * ex;
      out += pA[(pb + (size_t)s2*RMAX)*256 + d] * ex;
    }
    out /= Ssum;
    const float gf = __half2float(qrow[(ROWS==1 ? 0 : t*12288) + h*512 + 256 + d]);
    const float sg = 1.0f / (1.0f + expf(-gf));
    ao[(ROWS==1 ? 0 : t*6144) + h*256 + d] = __float2half(out * sg);
  }
}
