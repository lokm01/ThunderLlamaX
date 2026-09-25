// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R4 acceptsel5k: deep-set state select. rec: slot m -> live 4 (head-local
// per-block CTAs; m==4 is a self-copy no-op). conv: m<4 copies slot m as the
// legacy acceptsel; m==4 copies conv5x (k2s5's race-safe t=4 window) -> slot 4.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) acceptsel5k(
    float* __restrict__ rec4, float* __restrict__ conv4, const int* __restrict__ m_slot,
    const float* __restrict__ conv5x)
{
  const int b = blockIdx.x;
  const int m = m_slot[0];
  {
    const float* src = rec4 + ((size_t)b*5 + m) * 786432u;
    float* dst = rec4 + ((size_t)b*5 + 4) * 786432u;
    for (int i = threadIdx.x; i < 786432; i += 256) dst[i] = src[i];
  }
  if (m == 4) {
    const float* src = conv5x + (size_t)b * 30720u;
    float* dst = conv4 + ((size_t)b*5 + 4) * 30720u;
    for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
  } else {
    const float* src = conv4 + ((size_t)b*5 + m) * 30720u;
    float* dst = conv4 + ((size_t)b*5 + 4) * 30720u;
    for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
  }
}
