// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// T=1 combine: out[d + h*256] = (sum_s ws[s][g][h6][:256][d]) * sigmoid(gate[d + h*512 + 256])
#include <cuda_fp16.h>
#ifndef S
#define S 24
#endif
extern "C" __global__ void __launch_bounds__(256) k2t1(
    float* __restrict__ out, float* __restrict__ P, const __half* __restrict__ KV,
    float* __restrict__ gate, const float* __restrict__ ws)
{
  const int h = blockIdx.x;             // 24 heads
  const int d = threadIdx.x;
  const int g = h / 6, h6 = h % 6;
  float acc = 0.0f;
  for (int s = 0; s < S; s++)
    for (int wv = 0; wv < 8; wv++)
      acc += ws[(((size_t)(s * 4 + g) * 6) + h6) * 8 * 256 + (size_t)wv * 256 + d];
  const float gt = gate[d + (size_t)h * 512 + 256];
  out[d + (size_t)h * 256] = acc * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293335f)));
}
