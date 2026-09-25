// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R5d K=7 acceptsel8k: deep-set state select. [48][5] rec4/conv4 layout and
// live=slot-4 PRESERVED. rec: m<5 slot m -> 4; m==5 rec6x; m==6 rec7x; m==7 rec8x.
// conv: m<4 slot m; m==4 conv5x; m==5 conv6x; m==6 conv7x; m==7 conv8x.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) acceptsel8k(
    float* __restrict__ rec4, float* __restrict__ conv4, const int* __restrict__ m_slot,
    const float* __restrict__ conv5x, const float* __restrict__ conv6x, const float* __restrict__ conv7x, const float* __restrict__ conv8x,
    const float* __restrict__ rec6x, const float* __restrict__ rec7x, const float* __restrict__ rec8x)
{
  const int b = blockIdx.x;
  const int m = m_slot[0];
  {
    const float* src;
    if (m == 5) src = rec6x + (size_t)b * 786432u;
    else if (m == 6) src = rec7x + (size_t)b * 786432u;
    else if (m == 7) src = rec8x + (size_t)b * 786432u;
    else src = rec4 + ((size_t)b*5 + m) * 786432u;
    float* dst = rec4 + ((size_t)b*5 + 4) * 786432u;
    for (int i = threadIdx.x; i < 786432; i += 256) dst[i] = src[i];
  }
  {
    const float* src;
    if (m == 4) src = conv5x + (size_t)b * 30720u;
    else if (m == 5) src = conv6x + (size_t)b * 30720u;
    else if (m == 6) src = conv7x + (size_t)b * 30720u;
    else if (m == 7) src = conv8x + (size_t)b * 30720u;
    else src = conv4 + ((size_t)b*5 + m) * 30720u;
    float* dst = conv4 + ((size_t)b*5 + 4) * 30720u;
    for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
  }
}
