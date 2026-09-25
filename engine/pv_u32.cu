// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// probe pv_u32

#include <cuda_fp16.h>
#define FULL 0xffffffffu

extern "C" __global__ void __launch_bounds__(256) pv_u32(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = (blockIdx.x << 8) + threadIdx.x;          // 1 u32 per thread
  const unsigned int v = *(const unsigned int*)(w + ((size_t)i << 2));
  float fv = (float)(v & 0xFF) + (float)((v >> 8) & 0xFF) + (float)((v >> 16) & 0xFF) + (float)(v >> 24);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}
