// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R5 K=5 acceptsel6k: deep-set state select. [48][5] rec4/conv4 layout and
// live=slot-4 PRESERVED (no trunk/serve surgery). rec: m<5 copies slot m -> 4
// (head-local; m==4 self-copy no-op); m==5 copies rec6x (k2s6's t=5 scratch)
// -> 4. conv: m<4 slot m (legacy); m==4 conv5x (t=4 window); m==5 conv6x (t=5).
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) acceptsel6k(
    float* __restrict__ rec4, float* __restrict__ conv4, const int* __restrict__ m_slot,
    const float* __restrict__ conv5x, const float* __restrict__ conv6x, const float* __restrict__ rec6x)
{
  const int b = blockIdx.x;
  const int m = m_slot[0];
  {
    const float* src = (m == 5) ? (rec6x + (size_t)b * 786432u) : (rec4 + ((size_t)b*5 + m) * 786432u);
    float* dst = rec4 + ((size_t)b*5 + 4) * 786432u;
    for (int i = threadIdx.x; i < 786432; i += 256) dst[i] = src[i];
  }
  if (m >= 4) {
    const float* src = (m == 4) ? (conv5x + (size_t)b * 30720u) : (conv6x + (size_t)b * 30720u);
    float* dst = conv4 + ((size_t)b*5 + 4) * 30720u;
    for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
  } else {
    const float* src = conv4 + ((size_t)b*5 + m) * 30720u;
    float* dst = conv4 + ((size_t)b*5 + 4) * 30720u;
    for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
  }
}
