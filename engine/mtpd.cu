// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2-MTP: draft-chain + accept/commit kernels. Draft = blk.64 (all Q4_0,
// repacked two-region per row: [qs NGRP*16B][d NGRP*8*2B] — all loads naturally
// aligned per the ALIGNMENT LAW). Draft numerics are heuristic-only (no bit-exact
// contract); the ACCEPT kernel implements the exact mtp_v3 emission contract.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define INNER 6144
#define FFN_N 17408
#define EPS_N 1e-6f
#define NH 24
#define CTXK 2304
#define SLICE 40960

__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- dnorm2: enorm(e) || hnorm(hm) -> cat[10240] halves ----
extern "C" __global__ void __launch_bounds__(256) dnorm2(
    const float* __restrict__ e, const float* __restrict__ hm,
    const float* __restrict__ enw, const float* __restrict__ hnw,
    __half* __restrict__ cat)
{
  const int lane = threadIdx.x & 31;
  float ss0 = 0.f, ss1 = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v0 = e[i]; ss0 += v0*v0; const float v1 = hm[i]; ss1 += v1*v1; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { ss0 += __shfl_xor_sync(FULL, ss0, o); ss1 += __shfl_xor_sync(FULL, ss1, o); }
  const float r0 = rsqrtf(ss0/DIM + EPS_N), r1 = rsqrtf(ss1/DIM + EPS_N);
  for (int i = threadIdx.x; i < DIM; i += 256) {
    cat[i] = __float2half(e[i]*r0*enw[i]);
    cat[DIM + i] = __float2half(hm[i]*r1*hnw[i]);
  }
}

// ---- dfgu: draft FFN gate+up Q4 GEMVs + silu-mul ----
extern "C" __global__ void __launch_bounds__(256) dfgu(
    const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
    const __half* __restrict__ hhx, __half* __restrict__ gact)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= FFN_N) return;
  const unsigned char* rg = wg + (size_t)warp * 2880u;
  const unsigned char* ru = wu + (size_t)warp * 2880u;
  float ag = 0.f, au = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    const float4 xf = *(const float4*)(hhx + koff);
    const __half2* hx = (const __half2*)&xf;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    {
      const int subi = (b<<3) + (lane>>2);
      const unsigned long long qs8 = *(const unsigned long long*)(rg + subi*16 + ((lane&1)<<3));
      const float d = __half2float(*((const __half*)(rg + 2560 + subi*2)));
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float qv = (float)(((qs8 >> (8*j)) >> (((lane>>1)&1)<<2)) & 0xFu);
        ag += __half2float(__hmul(__float2half(xv[j]), __float2half(d*(qv - 8.f))));
      }
    }
    {
      const int subi = (b<<3) + (lane>>2);
      const unsigned long long qs8 = *(const unsigned long long*)(ru + subi*16 + ((lane&1)<<3));
      const float d = __half2float(*((const __half*)(ru + 2560 + subi*2)));
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float qv = (float)(((qs8 >> (8*j)) >> (((lane>>1)&1)<<2)) & 0xFu);
        au += __half2float(__hmul(__float2half(xv[j]), __float2half(d*(qv - 8.f))));
      }
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { ag += __shfl_down_sync(FULL, ag, o); au += __shfl_down_sync(FULL, au, o); }
  if (lane == 0) gact[warp] = __hmul(hsilu_h((__half)ag), (__half)au);
}

// ---- dkv: draft k + v Q4 GEMVs in one launch (warps [0,1024) k, [1024,2048) v) ----
extern "C" __global__ void __launch_bounds__(256) dkv(
    const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv,
    const __half* __restrict__ xh, __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const bool isk = (warp < 1024);
  const int r = isk ? warp : warp - 1024;
  const unsigned char* rowp = (isk ? wk : wv) + (size_t)r * 2880u;
  float acc = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const int koff = (b << 8) + (lane << 3);
    const float4 xf = *(const float4*)(xh + koff);
    const __half2* hx = (const __half2*)&xf;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    const int subi = (b<<3) + (lane>>2);
    const unsigned long long qs8 = *(const unsigned long long*)(rowp + subi*16 + ((lane&1)<<3));
    const float d = __half2float(*((const __half*)(rowp + 2560 + subi*2)));
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float qv = (float)(((qs8 >> (8*j)) >> (((lane>>1)&1)<<2)) & 0xFu);
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(d*(qv - 8.f))));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) { if (isk) krow[r] = (__half)acc; else vrow[r] = (__half)acc; }
}

// ---- aattn_d: draft attention (a_attn clone, CTX=2304, pos from arg buffer) ----
extern "C" __global__ void __launch_bounds__(256) aattn_d(
    const __half* __restrict__ qrow, const __half* __restrict__ krow, const __half* __restrict__ vrow,
    const float* __restrict__ qnw, const float* __restrict__ knw, const float* __restrict__ freqs,
    __half* __restrict__ kv, const int* __restrict__ pos_slot, __half* __restrict__ ao)
{
  const int h = blockIdx.x;
  const int kvh = h / 6;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int d = threadIdx.x;
  __shared__ float sm[3088];
  const int pos = pos_slot[0];
  const float ang = (d < 64) ? (float)pos * freqs[d & 31] : 0.0f;
  const float cs = cosf(ang), sn = sinf(ang);
  float qv = __half2float(qrow[h*512 + d]);
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
    sm[512 + d] = qe * 0.0625f;
  }
  {
    float kvv = __half2float(krow[kvh*256 + d]);
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
      Kc[(size_t)pos*256 + d] = __float2half(ko);
      Vc[(size_t)pos*256 + d] = vrow[kvh*256 + d];
    }
  }
  __syncthreads();
  {
    const __half* Kc = kv + (size_t)kvh * (CTXK*256);
    const __half* Vc = kv + (size_t)(4 + kvh) * (CTXK*256);
    float m = -1e30f, sacc = 0.f; float acc[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for (int l = warp; l <= pos; l += 8) {
      const __half* kr_ = Kc + (size_t)l*256;
      float sc = 0.f;
      #pragma unroll
      for (int j = 0; j < 8; ++j) sc += sm[512 + lane*8 + j] * __half2float(kr_[lane*8 + j]);
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(FULL, sc, o);
      if (sc > m) {
        const float cor = expf(m - sc);
        sacc *= cor;
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] *= cor;
        m = sc;
      }
      const float p = expf(sc - m);
      sacc += p;
      const __half* vr_ = Vc + (size_t)l*256;
      #pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += p * __half2float(vr_[lane*8 + j]);
    }
    if (lane == 0) { sm[3072 + warp] = m; sm[3080 + warp] = sacc; }
    #pragma unroll
    for (int j = 0; j < 8; ++j) sm[1024 + warp*256 + lane*8 + j] = acc[j];
    __syncthreads();
    float M = -1e30f;
    #pragma unroll
    for (int w2 = 0; w2 < 8; ++w2) M = fmaxf(M, sm[3072 + w2]);
    float out = 0.f, S = 0.f;
    #pragma unroll
    for (int w2 = 0; w2 < 8; ++w2) {
      const float ex = expf(sm[3072 + w2] - M);
      S += sm[3080 + w2] * ex;
      out += sm[1024 + w2*256 + d] * ex;
    }
    out /= S;
    const float gf = __half2float(qrow[h*512 + 256 + d]);
    const float sg = 1.0f / (1.0f + expf(-gf));
    ao[h*256 + d] = __float2half(out * sg);
  }
}

// ---- shead: draft head over the 40960-row Q5_K slice ----
extern "C" __global__ void __launch_bounds__(256) shead(
    const unsigned char* __restrict__ wq5, const __half* __restrict__ xh, __half* __restrict__ slogits)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= SLICE) return;
  const unsigned char* rowp = wq5 + (size_t)warp * 3520u;
  float acc = 0.f;
  #pragma unroll 5
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + b*176;
    const float d = __half2float(*((const __half*)blk));
    const float dm = __half2float(*((const __half*)(blk+2)));
    const int s = lane >> 2;
    float sc, mn;
    if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
    else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
           mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
    const unsigned long long qs8 = *(const unsigned long long*)(blk + 48 + ((lane >> 3) << 5) + ((lane & 3) << 3));
    const unsigned long long qh8 = *(const unsigned long long*)(blk + 16 + ((lane & 3) << 3));
    const int nsh = ((lane >> 2) & 1) << 2;
    const int koff = (b << 8) + (lane << 3);
    const float4 xf0 = *(const float4*)(xh + koff);
    const __half2* h0 = (const __half2*)&xf0;
    float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
           f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int qv = (int)(((unsigned int)(qs8 >> (8*j)) >> nsh) & 0xFu);
      qv += (int)((((unsigned int)(qh8 >> (8*j)) >> s) & 1u) << 4);
      const float w = d*sc*(float)qv - dm*mn;
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) slogits[warp] = (__half)acc;
}

// ---- samx: slice argmax + id-table lookup -> dring[j] ----
extern "C" __global__ void __launch_bounds__(256) samx(
    const __half* __restrict__ slogits, const int* __restrict__ stab, int* __restrict__ out_tok)
{
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  __shared__ int sm[16];
  float best = -1e30f; int bidx = 0;
  for (int i = tid; i < SLICE; i += 256) {
    const float v = __half2float(slogits[i]);
    if (v > best || (v == best && i < bidx)) { best = v; bidx = i; }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(FULL, best, o);
    const int oi = __shfl_down_sync(FULL, bidx, o);
    if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
  }
  if (lane == 0) { sm[warp] = __float_as_int(best); sm[8+warp] = bidx; }
  __syncthreads();
  if (tid == 0) {
    for (int w = 1; w < 8; ++w)
      if (__int_as_float(sm[w]) > __int_as_float(sm[0]) || (__int_as_float(sm[w]) == __int_as_float(sm[0]) && sm[8+w] < sm[8])) { sm[0] = sm[w]; sm[8] = sm[8+w]; }
    out_tok[0] = stab[sm[8]];
  }
}

// ---- dposadd: dst[0] = src[0] + 1 ----
extern "C" __global__ void __launch_bounds__(256) dposadd(
    const int* __restrict__ src, int* __restrict__ dst)
{
  if (threadIdx.x == 0) dst[0] = src[0] + 1;
}

// ---- accept: m = longest prefix match; emit, advance, select h_seed ----
extern "C" __global__ void __launch_bounds__(256) accept(
    const int* __restrict__ amds, const int* __restrict__ dring0, const int* __restrict__ dring1, const float* __restrict__ xA,
    int* __restrict__ m_slot, int* __restrict__ m_hist, int* __restrict__ cyc_slot,
    int* __restrict__ pos_slot, int* __restrict__ cur_slot, int* __restrict__ tok_hist,
    float* __restrict__ h_seed)
{
  __shared__ int ms;
  if (threadIdx.x == 0) {
    int m = 0;
    if (amds[0] == dring0[0]) { m = 1; if (amds[1] == dring1[0]) m = 2; }
    const int pos = pos_slot[0];
    for (int t = 0; t <= m; ++t) tok_hist[pos + t] = amds[t];
    cur_slot[0] = amds[m];
    pos_slot[0] = pos + m + 1;
    m_slot[0] = m;
    m_hist[cyc_slot[0]] = m;
    cyc_slot[0] = cyc_slot[0] + 1;
    ms = m;
  }
  __syncthreads();
  const int m = ms;
  for (int i = threadIdx.x; i < DIM; i += 256) h_seed[i] = xA[m*DIM + i];
}

// ---- acceptsel: copy per-block rec/conv slot m -> live slot 4 ----
// rec4: [48][5][786432] f32; conv4: [48][5][30720] f32 (block-major); grid 48 CTAs.
extern "C" __global__ void __launch_bounds__(256) acceptsel(
    float* __restrict__ rec4, float* __restrict__ conv4, const int* __restrict__ m_slot)
{
  const int b = blockIdx.x;
  const int m = m_slot[0];
  {
    const float* src = rec4 + ((size_t)b*5 + m) * 786432u;
    float* dst = rec4 + ((size_t)b*5 + 4) * 786432u;
    for (int i = threadIdx.x; i < 786432; i += 256) dst[i] = src[i];
  }
  {
    const float* src = conv4 + ((size_t)b*5 + m) * 30720u;
    float* dst = conv4 + ((size_t)b*5 + 4) * 30720u;
    for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
  }
}
