// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cstdint>
// device-side buffer fill: uint4 i -> (i, ~i, i^0x55, i+7) — varied content for
// the dependent-load arm; no host DMA (the DART >=1GB-copy danger law).
extern "C" __global__ void __launch_bounds__(256) e1_fill(uint4* src)
{
  const size_t n = (size_t)176*1024*1024*2;   // byte size passed baked (2*176MB=352MB worst)
  const size_t NU4_MAX = 16777216ull;          // 256MB in uint4
  const size_t i0 = (size_t)blockIdx.x * 8192ull;
  #pragma unroll
  for (int k = 0; k < 32; ++k) {
    const size_t i = i0 + (size_t)k * 256 + threadIdx.x;
    if (i >= NU4_MAX) return;
    const unsigned v = (unsigned)(i * 2654435761ull);
    uint4 x; x.x = v; x.y = ~v; x.z = v ^ 0x55555555u; x.w = v + 7u;
    src[i] = x;
  }
}
