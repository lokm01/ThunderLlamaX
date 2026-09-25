// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-b: attention blocks (16x, Qwen3.8-27B). T=1 decode.
// dims: 24 q-heads x 256, 4 kv-heads x 256, gate interleaved [q|gate] per head,
// qk RMSNorm per head (eps 1e-6), NeoX rope (pairs (i, i+128)), GQA h->h/6,
// scale 1/16, fp16 KV cache [2][4][2048][256], sigmoid gate, o_proj 6144->5120.
// Weight types: q = Q6_K or IQ3_XXS (per block), k = IQ3_XXS, v = Q4_K, o = IQ3_S.
// HARD RULES: flat indexing, no gridDim reads, sequential loops, single-array smem,
// byte/uint16 loads on Q-buffers, per-kernel cubins.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define NH 24
#define NKV 4
#define HD 256
#define CTX 2048
#define EPS_N 1e-6f
#define QOUT 12288
#define KVOUT 1024

// ---- a_q6: Q6_K GEMV [12288 rows x 5120 in], row 4200B (20 blocks x 210B) ----
extern "C" __global__ void __launch_bounds__(256) a_attn(
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
  const float ang = (d < 64) ? (float)pos * freqs[d & 31] : 0.0f;   // partial rope: first 64 dims only
  const float cs = cosf(ang), sn = sinf(ang);
  // --- q norm (fp32 math, half round-back like stock RMSNorm) ---
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
    sm[512 + d] = qe * 0.0625f;      // fold the 1/sqrt(256) scale into q
  }
  // --- k norm + rope; store K/V for this kv head (h%6==0 stores) ---
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
    sm[256 + d] = kvv;                // qn consumed into qe (sm[512]) above; sync above separates
    __syncthreads();
    const float ko = (d < 32) ? kvv*cs - sm[256 + d + 32]*sn
                   : (d < 64) ? kvv*cs + sm[256 + d - 32]*sn
                   : kvv;
    if ((h % 6) == 0) {
      __half* Kc = kv + (size_t)kvh * (CTX*256);
      __half* Vc = kv + (size_t)(NKV + kvh) * (CTX*256);
      Kc[(size_t)pos*256 + d] = __float2half(ko);
      Vc[(size_t)pos*256 + d] = vrow[kvh*256 + d];
    }
  }
  __syncthreads();
  // --- attention over l = 0..pos; 8 warps split l (warp-uniform), 8 dims per lane ---
  {
    const __half* Kc = kv + (size_t)kvh * (CTX*256);
    const __half* Vc = kv + (size_t)(NKV + kvh) * (CTX*256);
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
    // combine 8 warps through smem
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
    // sigmoid gate; fp32 math then half store (best match to stock elementwise promotion)
    const float gf = __half2float(qrow[h*512 + 256 + d]);
    const float sg = 1.0f / (1.0f + expf(-gf));
    ao[h*256 + d] = __float2half(out * sg);
  }
}

// ---- a_o: IQ3_S GEMV [5120 rows x 6144 in], row 2640B = 24 blocks x 110B ----
