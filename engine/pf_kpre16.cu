// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P2 pKPRE-M: M=16 KPRE for prefill chunks (KV8 int8-KV canonical path).
// Per chunk of 16 new positions [pos, pos+16): per q-head q-norm + partial-RoPE
// (rope_dim=64, pairs (i, i+32), theta from freqs table) -> qw16 fp16 rows
// (qe*0.0625 folded, exact in fp16 range for our magnitudes); k-norm + rope +
// BIASED-UINT8 KV append with per-(row,32ch) fp16 scales — quantizer math
// VERBATIM from spk_preqh.cu (the T=1 canonical): identical per-row op order
// => appends bit-identical to 16 T=1 KPRE runs, qw16 bit-identical per row.
// Laws: full-mask shuffles, sequential t loop, no gridDim reads, smem single array.
// grid (24,1,1) 256thr; one CTA per q-head h; kvh = h/6 appends on (h%6)==0.
// -DCTXK (slab stride). Name carries no warp token (256thr default class).
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define EPS_N 1e-6f
#define TROWS 16

extern "C" __global__ void __launch_bounds__(256) pfk_pre16(
    const __half* __restrict__ qrow16, const __half* __restrict__ krow16, const __half* __restrict__ vrow16,
    const float* __restrict__ qnw, const float* __restrict__ knw, const float* __restrict__ freqs,
    unsigned char* __restrict__ kv, __half* __restrict__ sc, const int* __restrict__ pos_slot,
    __half* __restrict__ qw16)
{
  const int h = blockIdx.x;
  const int kvh = h / 6;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int d = threadIdx.x;
  __shared__ float sm[768];
  const int pos = pos_slot[0];
  for (int t = 0; t < TROWS; ++t) {
    const float ang = (d < 64) ? (float)(pos + t) * freqs[d & 31] : 0.0f;
    const float cs = cosf(ang), sn = sinf(ang);
    float qv = __half2float(qrow16[(size_t)t*12288 + h*512 + d]);
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
      qw16[(size_t)t*6144 + h*256 + d] = __float2half(qe * 0.0625f);
    }
    {
      float kvv = __half2float(krow16[(size_t)t*1024 + kvh*256 + d]);
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
        unsigned char* Kc = kv + (size_t)kvh * (CTXK*256);
        unsigned char* Vc = kv + (size_t)(4 + kvh) * (CTXK*256);
        __half* Ksc = sc + (size_t)kvh * (CTXK*8);
        __half* Vsc = sc + (size_t)(4 + kvh) * (CTXK*8);
        const float kf = __half2float(__float2half(ko));
        const float vf = __half2float(vrow16[(size_t)t*1024 + kvh*256 + d]);
        float mk = fabsf(kf), mv = fabsf(vf);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          mk = fmaxf(mk, __shfl_xor_sync(FULL, mk, o));
          mv = fmaxf(mv, __shfl_xor_sync(FULL, mv, o));
        }
        const float sk = (mk > 0.f) ? mk * (1.f/127.f) : 1e-8f;
        const float sv = (mv > 0.f) ? mv * (1.f/127.f) : 1e-8f;
        const __half skh = __float2half(sk), svh = __float2half(sv);
        Ksc[(size_t)(pos+t)*8 + (d >> 5)] = skh;
        Vsc[(size_t)(pos+t)*8 + (d >> 5)] = svh;
        const float qk = kf / __half2float(skh), qv8 = vf / __half2float(svh);
        const int iq = __float2int_rn(fminf(fmaxf(qk, -127.f), 127.f));
        const int iv = __float2int_rn(fminf(fmaxf(qv8, -127.f), 127.f));
        Kc[(size_t)(pos+t)*256 + d] = (unsigned char)(iq + 128);
        Vc[(size_t)(pos+t)*256 + d] = (unsigned char)(iv + 128);
      }
    }
    __syncthreads();
  }
}
