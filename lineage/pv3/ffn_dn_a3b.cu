// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
#define A3B_N 17408
#define A3B_K 5120
#define A3B_BLOCKS 20
#define A3B_BATCH 1
extern "C" __global__ void r_544_8_8_4_20_4_2_4_a3b(float* data0_17408, float* data1_5120, float* data2_1, unsigned char* data3_20480, unsigned char* data4_34119680, float* data5_1024, unsigned char* data6_11141120) {
  float* a3b_y = data0_17408;
  const float* a3b_x = data1_5120;
  const float* a3b_s = data2_1;
  const float* a3b_nw = (const float*)data3_20480;
  const unsigned char* a3b_W = data4_34119680;
  const float* a3b_g = data5_1024;
// A3b warp-per-row IQ3_XXS dequant-GEMV body (A3a design; math == gguf.py case 18).
// Inserted into a kernel with the tinygrad-generated signature (arg order preserved).
// One warp per output row; thread `lane` handles elements 8*lane..8*lane+7 of each
// 256-elem block. Signs are computed INLINE from the scale words: sign bit b (b<7) of
// the 7-bit slot index, bit 7 = parity of the index (XOR form of even_signs[] table,
// bit-exact). No blockDim/gridDim reads (cbuf_0 NTID is zero on this driver stack).
// Placeholders for the x-load, epilogue and warps-per-CTA count are filled by the fork hook.
  const int warp = (blockIdx.x * 4) + (threadIdx.x >> 5);
  if (warp >= A3B_N * 1) return;
  const int lane = threadIdx.x & 31;
  const unsigned char* row = a3b_W + (size_t)(warp % A3B_N) * (size_t)A3B_BLOCKS * 98u;
  const int xrow = (warp / A3B_N) * A3B_K;
  float acc = 0.0f;
  #pragma unroll 4
  for (int b = 0; b < A3B_BLOCKS; ++b) {
    const unsigned char* blk = row + (size_t)b * 98u;
    const float d = __half2float(*((const __half*)blk));
    const unsigned short* sc = (const unsigned short*)(blk + 66);
    const unsigned int sw = ((unsigned int)sc[2*(lane>>2)]) | (((unsigned int)sc[2*(lane>>2)+1]) << 16);
    const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
    const unsigned int sidx = (sw >> (7u * (unsigned int)(lane & 3))) & 0x7Fu;
    const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
    const unsigned int q = ((unsigned int)blk[2 + 2*lane]) | (((unsigned int)blk[3 + 2*lane]) << 8);
    const float4 g0 = *((const float4*)(a3b_g + ((q & 0xFFu) << 2)));
    const float4 g1 = *((const float4*)(a3b_g + ((q >> 8) << 2)));
    const int koff = xrow + (int)(((size_t)b << 8) + ((unsigned int)lane << 3));
    const float4 xa = *((const float4*)(a3b_x + koff));
    const float4 xb = *((const float4*)(a3b_x + koff + 4));
    const int eoff = koff - xrow;   // nw is per-input-dim (not batched)
    const float4 na = *((const float4*)(a3b_nw + eoff));
    const float4 nb = *((const float4*)(a3b_nw + eoff + 4));
    const float x0 = xa.x*na.x, x1 = xa.y*na.y, x2 = xa.z*na.z, x3 = xa.w*na.w;
    const float x4 = xb.x*nb.x, x5 = xb.y*nb.y, x6 = xb.z*nb.z, x7 = xb.w*nb.w;
    const float sg0 = ((sidx>>0)&1) ? -1.0f : 1.0f;
    const float sg1 = ((sidx>>1)&1) ? -1.0f : 1.0f;
    const float sg2 = ((sidx>>2)&1) ? -1.0f : 1.0f;
    const float sg3 = ((sidx>>3)&1) ? -1.0f : 1.0f;
    const float sg4 = ((sidx>>4)&1) ? -1.0f : 1.0f;
    const float sg5 = ((sidx>>5)&1) ? -1.0f : 1.0f;
    const float sg6 = ((sidx>>6)&1) ? -1.0f : 1.0f;
    const float sg7 = spar ? -1.0f : 1.0f;
    acc += db * ( g0.x*sg0*x0 + g0.y*sg1*x1 + g0.z*sg2*x2 + g0.w*sg3*x3
                + g1.x*sg4*x4 + g1.y*sg5*x5 + g1.z*sg6*x6 + g1.w*sg7*x7 );
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    float u = acc * (1.0f/a3b_s[warp/A3B_N]); a3b_y[warp] = u / (1.0f + expf(-u));
  }

}
