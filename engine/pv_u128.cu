// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// probe pv_u128

#include <cuda_fp16.h>
#define FULL 0xffffffffu

extern "C" __global__ void __launch_bounds__(256) pv_u128(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = blockIdx.x << 7 | (threadIdx.x >> 1);     // 1 uint4 per 2 threads (128 thr/CTA used)
  const uint4 v = *(const uint4*)(w + ((size_t)i << 4));
  float fv = (float)v.x + (float)v.y + (float)v.z + (float)v.w;
  fv = (float)((unsigned)v.x & 0xFF) + (float)((unsigned)v.y & 0xFF) + (float)((unsigned)v.z & 0xFF) + (float)(v.w & 0xFF);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}
