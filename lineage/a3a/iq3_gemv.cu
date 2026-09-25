// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// A3a: standalone IQ3_XXS dequant-GEMV microbenchmark kernels for sm_86 (RTX 3090).
// Math ported EXACTLY from tinygrad fork llm/gguf.py ggml_data_to_tensor case 18:
//   block = 98 bytes / 256 elems: d fp16[0:2]; qs[2:66] (each byte -> grid[256][4],
//   4 consecutive values); scales[66:98] = 8 uint32: top nibble = 4-bit db scale,
//   4x7-bit indices (shifts 0,7,14,21) into even_signs[128] table (byte bits ->
//   per-element signs, bit==0 -> +1); db = d*(s+0.5)*0.5 per 32-elem group.
//   grid byte values are used RAW (4..62), NOT offset by 32.
// y[N] = W[N,K] . x[K]; W row-major GGUF IQ3_XXS blocks (256 elems/98B) along K.
// x is fp16 (matches model activations), accumulate fp32.
#include <cuda_fp16.h>

typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;

// v1: one warp per output row; each thread handles 8 consecutive elements of a
// 256-elem block (elements 8*lane..8*lane+7 = exactly one sign-group slot).
extern "C" __global__ __launch_bounds__(512) void iq3_gemv(
    const u8*    __restrict__ w,      // raw IQ3_XXS blocks, N*(K/256)*98 bytes
    const __half* __restrict__ x,     // K halves
    float*       __restrict__ y,      // N floats
    const float* __restrict__ gridf,  // iq3xxs_grid as float[256*4]
    const u8*    __restrict__ esign,  // even_signs table, 128 bytes
    int N, int K)
{
  const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (warp >= N) return;
  const int lane = threadIdx.x & 31;
  const int nb = K >> 8;                       // 98B blocks per row
  const u8* row = w + (size_t)warp * (size_t)nb * 98u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < nb; ++b) {
    const u8* blk = row + (size_t)b * 98u;
    const float d = __half2float(*(const __half*)blk);
    const u16* sc = (const u16*)(blk + 66);
    const int g = lane >> 2;                   // 32-elem scale/sign group
    const u32 sw = (u32)sc[2*g] | ((u32)sc[2*g+1] << 16);
    const float db = d * ((float)(sw >> 28) + 0.5f) * 0.5f;
    const u8 sb = esign[(sw >> (7 * (lane & 3))) & 0x7Fu];
    const u32 q = (u32)blk[2 + 2*lane] | ((u32)blk[3 + 2*lane] << 8);
    const float4 gv0 = *(const float4*)(gridf + ((q & 0xFFu) << 2));
    const float4 gv1 = *(const float4*)(gridf + ((q >> 8) << 2));
    const uint4 xw = *(const uint4*)(x + ((size_t)b << 8) + (lane << 3));
    const float2 f0 = __half22float2(*((const __half2*)&xw.x));
    const float2 f1 = __half22float2(*((const __half2*)&xw.y));
    const float2 f2 = __half22float2(*((const __half2*)&xw.z));
    const float2 f3 = __half22float2(*((const __half2*)&xw.w));
    acc += db * ( gv0.x*(1.f-2.f*((sb>>0)&1))*f0.x + gv0.y*(1.f-2.f*((sb>>1)&1))*f0.y
                + gv0.z*(1.f-2.f*((sb>>2)&1))*f1.x + gv0.w*(1.f-2.f*((sb>>3)&1))*f1.y
                + gv1.x*(1.f-2.f*((sb>>4)&1))*f2.x + gv1.y*(1.f-2.f*((sb>>5)&1))*f2.y
                + gv1.z*(1.f-2.f*((sb>>6)&1))*f3.x + gv1.w*(1.f-2.f*((sb>>7)&1))*f3.y );
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) y[warp] = acc;
}

// v2: split-K variant — one warp per (row, split); partials then reduced.
// Requires nb % splits == 0.
extern "C" __global__ __launch_bounds__(512) void iq3_gemv_sk(
    const u8*    __restrict__ w,
    const __half* __restrict__ x,
    float*       __restrict__ part,   // N*splits floats
    const float* __restrict__ gridf,
    const u8*    __restrict__ esign,
    int N, int K, int splits)
{
  const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (wid >= N * splits) return;
  const int n = wid / splits, s = wid - n * splits;
  const int lane = threadIdx.x & 31;
  const int nb = K >> 8, b0 = s * (nb / splits), b1 = (s + 1) * (nb / splits);
  const u8* row = w + (size_t)n * (size_t)nb * 98u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = b0; b < b1; ++b) {
    const u8* blk = row + (size_t)b * 98u;
    const float d = __half2float(*(const __half*)blk);
    const u16* sc = (const u16*)(blk + 66);
    const int g = lane >> 2;
    const u32 sw = (u32)sc[2*g] | ((u32)sc[2*g+1] << 16);
    const float db = d * ((float)(sw >> 28) + 0.5f) * 0.5f;
    const u8 sb = esign[(sw >> (7 * (lane & 3))) & 0x7Fu];
    const u32 q = (u32)blk[2 + 2*lane] | ((u32)blk[3 + 2*lane] << 8);
    const float4 gv0 = *(const float4*)(gridf + ((q & 0xFFu) << 2));
    const float4 gv1 = *(const float4*)(gridf + ((q >> 8) << 2));
    const uint4 xw = *(const uint4*)(x + ((size_t)b << 8) + (lane << 3));
    const float2 f0 = __half22float2(*((const __half2*)&xw.x));
    const float2 f1 = __half22float2(*((const __half2*)&xw.y));
    const float2 f2 = __half22float2(*((const __half2*)&xw.z));
    const float2 f3 = __half22float2(*((const __half2*)&xw.w));
    acc += db * ( gv0.x*(1.f-2.f*((sb>>0)&1))*f0.x + gv0.y*(1.f-2.f*((sb>>1)&1))*f0.y
                + gv0.z*(1.f-2.f*((sb>>2)&1))*f1.x + gv0.w*(1.f-2.f*((sb>>3)&1))*f1.y
                + gv1.x*(1.f-2.f*((sb>>4)&1))*f2.x + gv1.y*(1.f-2.f*((sb>>5)&1))*f2.y
                + gv1.z*(1.f-2.f*((sb>>6)&1))*f3.x + gv1.w*(1.f-2.f*((sb>>7)&1))*f3.y );
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) part[(size_t)n * splits + s] = acc;
}

extern "C" __global__ void iq3_reduce(const float* __restrict__ part, float* __restrict__ y, int N, int splits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float a = 0.f;
  const float* p = part + (size_t)i * splits;
  for (int s = 0; s < splits; ++s) a += p[s];
  y[i] = a;
}
