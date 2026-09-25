// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pv3(
    float* __restrict__ out, float* __restrict__ P, float* __restrict__ mx, float* __restrict__ sm,
    half* __restrict__ V, float* __restrict__ gate, const int dummy)
{
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x + (blockIdx.z << 7);
  const float m_ = mx[(g*3) + r];
  const float s_ = sm[(g*3) + r];
  const float* P_g = P + ((size_t)g * 301056) + ((size_t)r * 100352);
  const __half* V_g = (const __half*)V + (((size_t)(g/6)) * 25690112LL) + 102760448LL;
  float acc = 0.0f;
  for (int p = 0; p < 100352; ++p)
    acc += exp2f((P_g[p] - m_) * 1.4426950216293334f) * __half2float(V_g[((size_t)p << 8) + d]);
  const float gt = gate[d + ((size_t)g << 9) + 256 + ((size_t)r * 12288)];
  out[d + ((size_t)g << 8) + ((size_t)r * 6144)] =
    (acc / s_) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));
}
