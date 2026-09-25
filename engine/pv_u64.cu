// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// probe pv_u64

#include <cuda_fp16.h>
#define FULL 0xffffffffu

extern "C" __global__ void __launch_bounds__(256) pv_u64(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = blockIdx.x << 8 | threadIdx.x;            // 1 u64 per thread
  const unsigned long long v = *(const unsigned long long*)(w + ((size_t)i << 3));
  float fv = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) fv += (float)((v >> (8*j)) & 0xFF);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}
