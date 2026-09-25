// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// probe pv_row32

#include <cuda_fp16.h>
#define FULL 0xffffffffu

extern "C" __global__ void __launch_bounds__(256) pv_row32(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const unsigned char* rowp = w + (size_t)warp * 3520u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + b*176;
    const unsigned int qa = *(const unsigned int*)(blk + 48 + lane*8);
    const unsigned int qb = *(const unsigned int*)(blk + 48 + lane*8 + 4);
    const unsigned int qh = *(const unsigned int*)(blk + 16 + ((lane & 3) << 3));
    acc += (float)(qa & 0xFF) + (float)(qb & 0xFF) + (float)(qh & 0xFF);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) out[warp] = acc;
}
