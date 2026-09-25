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
    float qv = __half2float(qrow[(ROWS==3 ? t*12288 : 0) + h*512 + d]);
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
      qw[(ROWS==3 ? t*24 : 0)*256 + h*256 + d] = qe * 0.0625f;
    }
    {
      float kvv = __half2float(krow[(ROWS==3 ? t*1024 : 0) + kvh*256 + d]);
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
        Vc[(size_t)(pos+t)*256 + d] = vrow[(ROWS==3 ? t*1024 : 0) + kvh*256 + d];
      }
    }
    __syncthreads();
  }
}

// ============================== K1S ==============================
