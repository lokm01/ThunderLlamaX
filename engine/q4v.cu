// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2-MTP: generic Q4_0 packed GEMV template. Compiled per-use with
// -DNOUT=<rows> -DNGRP=<input/256> (-DADDHH=1 for the down-proj + residual).
// Packed row layout: [qs: NGRP*8 sub-blocks x 16B][d: NGRP*8 sub-blocks x 2B]
// row bytes = NGRP*128 + NGRP*16 (same 18B/sub-block). All loads aligned.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#ifndef NGRP
#define NGRP 20
#endif
#ifndef NOUT
#define NOUT 5120
#endif
#ifndef KNAME
#define KNAME q4v
#endif

extern "C" __global__ void __launch_bounds__(256) KNAME(
    const unsigned char* __restrict__ w, const __half* __restrict__ x,
    const float* __restrict__ hh,
#ifdef ADDHH
    float* __restrict__ out
#else
    __half* __restrict__ out
#endif
)
{
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= NOUT) return;
  const unsigned char* rowp = w + (size_t)warp * (NGRP*144u);
  const unsigned char* qsreg = rowp;
  const unsigned char* dreg = rowp + NGRP*128u;
  float acc = 0.f;
  #pragma unroll 5
  for (int b = 0; b < NGRP; ++b) {
    const int koff = (b << 8) + (lane << 3);
    const float4 xf = *(const float4*)(x + koff);
    const __half2* hx = (const __half2*)&xf;
    float2 f0 = __half22float2(hx[0]), f1 = __half22float2(hx[1]), f2 = __half22float2(hx[2]), f3 = __half22float2(hx[3]);
    const float xv[8] = { f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y };
    const int subi = (b<<3) + (lane>>2);
    // fork Q4_0 truth: element e (within 32-sub-block) = byte (e&15), nibble (e>>4)
    // -> lane covers e = lane*8..+7: chunk = lane&1, byte-in-chunk = j, nibble = (lane>>1)&1
    const unsigned long long qs8 = *(const unsigned long long*)(qsreg + subi*16 + ((lane&1)<<3));
    const float d = __half2float(*((const __half*)(dreg + subi*2)));
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float qv = (float)(((qs8 >> (8*j)) >> ((((lane>>1)&1))<<2)) & 0xFu);
      acc += __half2float(__hmul(__float2half(xv[j]), __float2half(d*(qv - 8.f))));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
#ifdef ADDHH
  if (lane == 0) out[warp] = hh[warp] + (float)((__half)acc);
#else
  if (lane == 0) out[warp] = (__half)acc;
#endif
}
