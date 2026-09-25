// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2G L3: half2-core port of q4v.cu (LDH2+ACC-pairs, the W2F head8v_3
// pattern). Per-acc element add order preserved (pairs 2k,2k+1 ascending ==
// j ascending), products fp16 exactly like q4v (the x side keeps its native
// fp16 bits; w side = float2half(d*(qv-8)) identical) -> BIT-IDENTICAL
// outputs. Same signature/grid as q4v. Compiled per-use with
// -DNOUT -DNGRP (-DADDHH=1) and -DKNAME (ehprojh/dqh/doprojh/ddownh).
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#ifndef NGRP
#define NGRP 20
#endif
#ifndef NOUT
#define NOUT 5120
#endif
#ifndef KNAME
#define KNAME q4vh
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
    const int subi = (b<<3) + (lane>>2);
    const unsigned long long qs8 = *(const unsigned long long*)(qsreg + subi*16 + ((lane&1)<<3));
    const float d = __half2float(*((const __half*)(dreg + subi*2)));
    float wv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float qv = (float)(((qs8 >> (8*j)) >> ((((lane>>1)&1))<<2)) & 0xFu);
      wv[j] = d * (qv - 8.f);
    }
    const __half2 w01 = __halves2half2(__float2half(wv[0]), __float2half(wv[1]));
    const __half2 w23 = __halves2half2(__float2half(wv[2]), __float2half(wv[3]));
    const __half2 w45 = __halves2half2(__float2half(wv[4]), __float2half(wv[5]));
    const __half2 w67 = __halves2half2(__float2half(wv[6]), __float2half(wv[7]));
    { const float2 p = __half22float2(__hmul2(hx[0], w01)); acc += p.x; acc += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[1], w23)); acc += p.x; acc += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[2], w45)); acc += p.x; acc += p.y; }
    { const float2 p = __half22float2(__hmul2(hx[3], w67)); acc += p.x; acc += p.y; }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
#ifdef ADDHH
  if (lane == 0) out[warp] = hh[warp] + (float)((__half)acc);
#else
  if (lane == 0) out[warp] = (__half)acc;
#endif
}
