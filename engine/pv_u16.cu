// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// probe pv_u16

#include <cuda_fp16.h>
#define FULL 0xffffffffu

extern "C" __global__ void __launch_bounds__(256) pv_u16(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = ((blockIdx.x << 8) + threadIdx.x) << 1;   // 1 u16 per thread
  const unsigned short v = *(const unsigned short*)(w + i);
  float fv = (float)(v & 0xFF) + (float)(v >> 8);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}
