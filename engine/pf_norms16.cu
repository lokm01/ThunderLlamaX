// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P2 prefill M=16 elementwise family (KSEL-dispatched, ONE kernel per cubin —
// the per-kernel cubin law). All math copied verbatim from the T=1 kernels
// (h_embed / k0_norm / k0ab / k3m_hh / dnorm2) with per-row op order unchanged:
// per-warp redundant RMS reductions (strided i = lane; i < DIM; i += 32 + xor
// butterfly — the k0ab order), fp16 rounding at the same points.
//   KSEL 1: pfk_emb16  — 16 token ids -> x16f[16][5120] fp32 (IQ3_S gather)
//   KSEL 2: pfk_n16    — RMS norm fp32 row -> fp16 row (attn blocks / head)
//   KSEL 3: pfk_ab16   — RMS norm + alpha/beta GEMV -> xh16 + araw16/braw16
//   KSEL 4: pfk_hh16   — hh = xin + f32(attn_out) (fp32 out) + hhx normed fp16
//   KSEL 5: pfd_dnorm16 — draft enorm(e)||hnorm(hm) -> cat16[16][10240] fp16
// LAWS: flat indexing, no gridDim reads, full masks, 256thr, hardcoded sizes.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define EPS_N 1e-6f

#if KSEL == 1
// grid (16): one CTA per row; tok = ids16[blockIdx.x]; h_embed math verbatim.
extern "C" __global__ void __launch_bounds__(256) pfk_emb16(
    const unsigned char* __restrict__ emb, const float* __restrict__ grid512,
    const int* __restrict__ ids16, float* __restrict__ x16f)
{
  const int tid = threadIdx.x;
  const int tok = ids16[blockIdx.x];
  float* x = x16f + (size_t)blockIdx.x*DIM;
  const unsigned char* row = emb + (size_t)tok * 2200u;
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
    x[e] = d * sc * grid512[(q << 2) + j4] * sgn;
  }
}

#elif KSEL == 2
// grid (16): one CTA per row. k0_norm math (per-warp redundant RMS).
extern "C" __global__ void __launch_bounds__(256) pfk_n16(
    const float* __restrict__ x16f, const float* __restrict__ nw, __half* __restrict__ xh16)
{
  const int lane = threadIdx.x & 31;
  const float* x = x16f + (size_t)blockIdx.x*DIM;
  __half* xh = xh16 + (size_t)blockIdx.x*DIM;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = x[i]; ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  for (int i = threadIdx.x; i < DIM; i += 256) xh[i] = __float2half(x[i]*r*nw[i]);
}

#elif KSEL == 3
// grid (16*13): CTA (row*13 + b); b==0 -> norm row; b>=1 -> warps 8..104 do the
// 96 alpha/beta f32 GEMV rows for that chunk row (k0ab math verbatim).
extern "C" __global__ void __launch_bounds__(256) pfk_ab16(
    const float* __restrict__ x16f, const float* __restrict__ nw,
    const float* __restrict__ walpha, const float* __restrict__ wbeta,
    __half* __restrict__ xh16, float* __restrict__ araw16, float* __restrict__ braw16)
{
  const int brow = blockIdx.x / 13, b = blockIdx.x % 13;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* x = x16f + (size_t)brow*DIM;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = x[i]; ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  if (b == 0) {
    __half* xh = xh16 + (size_t)brow*DIM;
    for (int i = threadIdx.x; i < DIM; i += 256) xh[i] = __float2half(x[i]*r*nw[i]);
  } else {
    const int w = (b - 1)*8 + warp;   // global warp [0,96) across CTAs 1..12
    const float* wr = (w < 48) ? walpha + (size_t)w*DIM : wbeta + (size_t)(w-48)*DIM;
    float acc = 0.f;
    for (int i = lane; i < DIM; i += 32)
      acc += wr[i] * __half2float(__float2half(x[i]*r*nw[i]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) {
      if (w < 48) araw16[(size_t)brow*48 + w] = acc;
      else braw16[(size_t)brow*48 + (w-48)] = acc;
    }
  }
}

#elif KSEL == 4
// grid (16): hh = x + f32(attn_out) fp32, hhx = half(hh*r*nw2) — k3m_hh verbatim.
extern "C" __global__ void __launch_bounds__(256) pfk_hh16(
    const float* __restrict__ x16f, const __half* __restrict__ attn_out16, const float* __restrict__ nw2,
    float* __restrict__ hh16, __half* __restrict__ hhx16)
{
  const int lane = threadIdx.x & 31;
  const float* x = x16f + (size_t)blockIdx.x*DIM;
  const __half* ao = attn_out16 + (size_t)blockIdx.x*DIM;
  float* hh = hh16 + (size_t)blockIdx.x*DIM;
  __half* hhx = hhx16 + (size_t)blockIdx.x*DIM;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) {
    const float v = x[i] + __half2float(ao[i]);
    ss += v*v;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  for (int i = threadIdx.x; i < DIM; i += 256) {
    const float v = x[i] + __half2float(ao[i]);
    hh[i] = v;
    hhx[i] = __float2half(v*r*nw2[i]);
  }
}

#elif KSEL == 5
// grid (16): draft dnorm2 for M rows — cat16 = [enorm(e) || hnorm(hm)] per row.
// Norm math in the k0ab/dnorm2 class (per-warp redundant RMS, half at the end).
extern "C" __global__ void __launch_bounds__(256) pfd_dnorm16(
    const float* __restrict__ e16f, const float* __restrict__ hm16f,
    const float* __restrict__ enw, const float* __restrict__ hnw, __half* __restrict__ cat16)
{
  const int lane = threadIdx.x & 31;
  const int brow = blockIdx.x;
  const float* e = e16f + (size_t)brow*DIM;
  const float* hm = hm16f + (size_t)brow*DIM;
  __half* cat = cat16 + (size_t)brow*(2*DIM);
  float sse = 0.f, ssh = 0.f;
  for (int i = lane; i < DIM; i += 32) { sse += e[i]*e[i]; ssh += hm[i]*hm[i]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) { sse += __shfl_xor_sync(FULL, sse, o); ssh += __shfl_xor_sync(FULL, ssh, o); }
  const float re = rsqrtf(sse/DIM + EPS_N), rh = rsqrtf(ssh/DIM + EPS_N);
  for (int i = threadIdx.x; i < DIM; i += 256) {
    cat[i] = __float2half(e[i]*re*enw[i]);
    cat[DIM + i] = __float2half(hm[i]*rh*hnw[i]);
  }
}

#elif KSEL == 6
// P4 Stage 2: record 16 trunk hidden rows (fp32) into the draft window ring:
// rec[1..16] = x32 rows 0..15; recseed[0] = x32 row 15 (next window hm seed).
extern "C" __global__ void __launch_bounds__(256) pfk_rec16(
    const float* __restrict__ x32, float* __restrict__ rec, float* __restrict__ recseed)
{
  const int b = blockIdx.x;
  const float* src = x32 + (size_t)(b & 15)*DIM;
  float* dst = (b < 16) ? (rec + (size_t)(b + 1)*DIM) : recseed;
  for (int i = threadIdx.x; i < DIM; i += 256) dst[i] = src[i];
}
#elif KSEL == 7
// R4: the M64 fat-CTA reshape of pfk_ab16 -- grid (rows*3), 1024 thr/CTA:
// 96 warps/row = the 96 alpha/beta GEMV jobs (warp-local math VERBATIM: the
// same lane-strided dot + shfl_down tree -> bit-identical per job); CTA 0
// additionally writes the normed xh row (elementwise-identical values). The
// 832-CTA original pays 10 waves at the dext 1-CTA/SM; this is 192 CTAs with
// 32 resident warps to hide the L2 latency.
extern "C" __global__ void __launch_bounds__(1024) pfk_ab16w(
    const float* __restrict__ x16f, const float* __restrict__ nw,
    const float* __restrict__ walpha, const float* __restrict__ wbeta,
    __half* __restrict__ xh16, float* __restrict__ araw16, float* __restrict__ braw16)
{
  const int brow = blockIdx.x / 3, c = blockIdx.x % 3;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* x = x16f + (size_t)brow*DIM;
  float ss = 0.f;
  for (int i = lane; i < DIM; i += 32) { const float v = x[i]; ss += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss/DIM + EPS_N);
  if (c == 0) {
    __half* xh = xh16 + (size_t)brow*DIM;
    for (int i = threadIdx.x; i < DIM; i += 1024) xh[i] = __float2half(x[i]*r*nw[i]);
  }
  const int w = c * 32 + warp;   // global warp [0,96)
  const float* wr = (w < 48) ? walpha + (size_t)w*DIM : wbeta + (size_t)(w-48)*DIM;
  float acc = 0.f;
  for (int i = lane; i < DIM; i += 32)
    acc += wr[i] * __half2float(__float2half(x[i]*r*nw[i]));
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) {
    if (w < 48) araw16[(size_t)brow*48 + w] = acc;
    else braw16[(size_t)brow*48 + (w-48)] = acc;
  }
}
#endif
