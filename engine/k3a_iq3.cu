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
