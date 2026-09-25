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
extern "C" __global__ void __launch_bounds__(256) a_q6(
    const unsigned char* __restrict__ wq6, const __half* __restrict__ xh, __half* __restrict__ qrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const unsigned char* rowp = wq6 + (size_t)warp * 4200u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + b*210;
    const float d = __half2float(*((const __half*)(blk+208)));
    // element e = b*256 + lane*8 + j;  h = e>>7, i = e&127
    const int lo_off = 64*(lane>>4) + (lane&7)*8;
    const bool nib_hi = ((lane&15) >= 8);
    const int c2 = (lane>>2)&3;   // 2-bit chunk = i>>5 = (lane*8+j)>>5, j<8 -> (lane>>2), mask to 0..3
    const unsigned char* qhp = blk + 128 + (lane>>4)*32 + (lane&3)*8;
    const int sc8 = (signed char)blk[192 + (lane>>1)];
    const int koff = (b << 8) + (lane << 3);
    const float4 xf0 = *(const float4*)(xh + koff);
    const __half2* h0 = (const __half2*)&xf0;
    float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
           f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int lo_byte = blk[lo_off + j];
      const int xl = nib_hi ? (lo_byte >> 4) : (lo_byte & 0xF);
      const int xh = ((qhp[j] >> (c2<<1)) & 3) << 4;
      const float w = d * (float)sc8 * (float)((signed char)((xl | xh) - 32));
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) qrow[warp] = (__half)acc;
}

// ---- a_kv: k GEMV (IQ3_XXS 1024 rows) + v GEMV (Q4_K 1024 rows) in one launch ----
extern "C" __global__ void __launch_bounds__(256) a_kv(
    const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv,
    const float* __restrict__ gridf, const __half* __restrict__ xh,
    __half* __restrict__ krow, __half* __restrict__ vrow)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  float acc = 0.f;
  if (warp < KVOUT) {
    // IQ3_XXS row (identical math to k1_iq3)
    const unsigned char* rowp = wk + (size_t)warp * 1960u;
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
      const __half2* hx = (const __half2*)&xa;
      float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
      const float xvv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
      const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
      const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
      const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
      const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
      const float wv[8] = { db*g0.x*sg0, db*g0.y*sg1, db*g0.z*sg2, db*g0.w*sg3,
                            db*g1.x*sg4, db*g1.y*sg5, db*g1.z*sg6, db*g1.w*sg7 };
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        acc += __half2float(__hmul(__float2half(xvv[j]), __float2half(wv[j])));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) krow[warp] = (__half)acc;
  } else {
    // Q4_K row (2880B = 20 blocks x 144B); scale decode identical to Q5_K, no qh, qs at +16
    const int r = warp - KVOUT;
    const unsigned char* rowp = wv + (size_t)r * 2880u;
    #pragma unroll 4
    for (int b = 0; b < 20; ++b) {
      const unsigned char* blk = rowp + b*144;
      const float d = __half2float(*((const __half*)blk));
      const float dm = __half2float(*((const __half*)(blk+2)));
      const int s = lane >> 2;
      float sc, mn;
      if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
      else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
             mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
      const unsigned char* qsb = blk + 16 + ((lane >> 3) << 5) + ((lane & 3) << 3);
      const int nsh = ((lane >> 2) & 1) << 2;
      const int koff = (b << 8) + (lane << 3);
      const float4 xf0 = *(const float4*)(xh + koff);
      const __half2* h0 = (const __half2*)&xf0;
      float2 f0 = __half22float2(h0[0]), f1 = __half22float2(h0[1]),
             f2 = __half22float2(h0[2]), f3 = __half22float2(h0[3]);
      const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float qv = (float)((qsb[j] >> nsh) & 0xF);
        const float w = d*sc*qv - dm*mn;
        acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
    if (lane == 0) vrow[r] = (__half)acc;
  }
}

// ---- a_attn: qk-norm + rope + KV append + gated softmax attention. grid=(24,) ----
// CTA = q head h; kv head kvh = h/6 (each CTA re-derives its kv head's k/v; only
// h%6==0 stores -> no cross-CTA dependency on the freshly written row).
// smem (single array): [0..15] reduce partials/bcast; [256..511] qn then kn stash;
// [512..767] qe (x 0.0625); [1024..3071] per-warp partial out (8x256);
// [3072..3087] per-warp (m, s). All write-after-read reuses are syncthreads-separated.
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
extern "C" __global__ void __launch_bounds__(256) a_o(
    const unsigned char* __restrict__ wo, const float* __restrict__ grid512,
    const __half* __restrict__ ao_in, __half* __restrict__ attn_out)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wo + (size_t)warp * 2640u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 24; ++b) {
    const unsigned char* blk = rowp + (size_t)b * 110u;
    const float d = __half2float(*((const __half*)blk));
    // element e = b*256 + lane*8 + j
    const int g0 = lane*2, g1 = lane*2 + 1;
    const int sraw = lane >> 2;                     // e>>5 constant per lane
    const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
    const unsigned char* sgnb = blk + 74 + lane;    // sign byte index = e>>3 mod 256
    const int koff = (b << 8) + (lane << 3);
    const float4 xa = *(const float4*)(ao_in + koff);
    const __half2* hx = (const __half2*)&xa;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int g = (j < 4) ? g0 : g1;
      const unsigned int q = (unsigned int)blk[2 + g] + ((((unsigned int)blk[66 + (g>>3)] >> (g&7)) & 1u) << 8);
      const float gv = grid512[(q << 2) + (j & 3)];
      const float sgn = ((*sgnb >> j) & 1) ? -1.f : 1.f;
      const float w = d * sc * gv * sgn;
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(w)));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) attn_out[warp] = (__half)acc;
}
