// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-b: embed (IQ3_S gather+dequant), argmax over half logits.
// The head GEMV reuses k0_norm (output_norm -> xh half) + k1_q5 (Q5_K 248320-row
// GEMV, grid 31040) from W1-a cubins.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define VOCAB 248320

// ---- h_embed: read token id from device slot, gather+dequant its IQ3_S row.
// grid=(1,), 256 threads; row = 2200B = 20 blocks x 110B. Output fp32 x[5120]. ----
extern "C" __global__ void __launch_bounds__(256) h_embed(
    const unsigned char* __restrict__ emb, const float* __restrict__ grid512,
    const int* __restrict__ tok_slot, float* __restrict__ x)
{
  const int tid = threadIdx.x;
  const int tok = tok_slot[0];
  const unsigned char* row = emb + (size_t)tok * 2200u;
  #pragma unroll
  for (int i = 0; i < 20; ++i) {
    const unsigned char* blk = row + i*110;
    const int e = i*256 + tid;             // element within the 5120-row
    // IQ3_S fields are WITHIN-BLOCK (256 elems): q g=e&255>>2, scale e&255>>5, sign byte e&255>>3 bit e&7
    const float d = __half2float(*((const __half*)blk));
    const int g = tid >> 2, j4 = tid & 3;
    const unsigned int q = (unsigned int)blk[2 + g] + ((((unsigned int)blk[66 + (g>>3)] >> (g&7)) & 1u) << 8);
    const int s8 = tid >> 5;
    const float sc = 1.0f + 2.0f*(float)((blk[106 + (s8>>1)] >> ((s8&1)<<2)) & 0xF);
    const float sgn = ((blk[74 + (tid>>3)] >> (tid & 7)) & 1) ? -1.f : 1.f;
    x[e] = d * sc * grid512[(q << 2) + j4] * sgn;
  }
}

// ---- h_argmax: greedy pick over half logits; grid=(1,), block-wide reduce.
// Tie-safe (lower index wins on equal values). Writes tok_slot and tok_hist[pos],
// bumps pos_slot. All device-resident (no host roundtrip in the decode loop). ----
extern "C" __global__ void __launch_bounds__(256) h_argmax(
    const __half* __restrict__ logits, int* __restrict__ tok_slot,
    int* __restrict__ pos_slot, int* __restrict__ tok_hist)
{
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  __shared__ int sm[16];   // single-array smem rule: [0..7] float-bits of val, [8..15] idx
  float best = -1e30f; int bidx = 0;
  for (int i = tid; i < VOCAB; i += 256) {
    const float v = __half2float(logits[i]);
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
    const int bidx_f = sm[8];
    const int pos = pos_slot[0];
    tok_slot[0] = bidx_f;
    tok_hist[pos] = bidx_f;
    pos_slot[0] = pos + 1;
  }
}

// ---- k3a_iq3: GDN o_proj for the 24 blocks whose ssm_out is IQ3_XXS (6144 in -> 5120 out,
//      row 2352B = 24 blocks x 98B) ----
extern "C" __global__ void __launch_bounds__(256) k3a_iq3(
    const unsigned char* __restrict__ wq3, const float* __restrict__ gridf,
    const __half* __restrict__ z, __half* __restrict__ attn_out)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= 5120) return;
  const unsigned char* rowp = wq3 + (size_t)warp * 2352u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 24; ++b) {
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
    const float4 xa = *(const float4*)(z + koff);
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
  if (lane == 0) attn_out[warp] = (__half)acc;
}
