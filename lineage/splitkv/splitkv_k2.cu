// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// K2 (dumb-simple): grid=(48,1,1), 128 threads; thread d owns output dim d.
// Serial loop over slots — obviously correct; optimize later.
#include <cuda_fp16.h>
struct __align__(8) half4 { half x, y, z, w; };
#define NEG_INF (__int_as_float(0xff800000))
extern "C" __global__ void __launch_bounds__(128) splitkv_k2(
    const float* __restrict__ oacc, const float* __restrict__ lse,
    half* __restrict__ out, const int slots)
{
  const int row = blockIdx.x;            // 0..47 = h*3 + t
  const int d = threadIdx.x;             // 0..127
  float mx = NEG_INF;
  for (int s = 0; s < slots; ++s) { const float v = lse[s*48 + row]; if (v > mx) mx = v; }
  if (mx == NEG_INF) { out[row*128 + d] = __float2half(0.0f); return; }
  float tot = 0.0f, acc = 0.0f;
  for (int s = 0; s < slots; ++s) {
    const float lv = lse[s*48 + row];
    const float sc = (lv == NEG_INF) ? 0.0f : exp2f(lv - mx);
    tot += sc;
    acc += sc * oacc[(s*48 + row)*128 + d];
  }
  if (tot <= 0.0f) tot = 1.0f;
  out[row*128 + d] = __float2half(acc / tot);
}
