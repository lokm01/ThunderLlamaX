// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// K2-RS combine for the row-shared K1: ws slot = ((s*4+g)*18 + h6*3 + r) * 258.
#include <cuda_fp16.h>
#define NEG_INF (__int_as_float(0xff800000))
#ifndef LMAX
#define LMAX 100352
#endif
#ifndef S
#define S 41
#endif
extern "C" __global__ void __launch_bounds__(256) k2rs(
    float* __restrict__ out, float* __restrict__ P, float* __restrict__ mx, float* __restrict__ sm,
    const __half* __restrict__ KV, float* __restrict__ gate, const float* __restrict__ ws)
{
  const int hr = blockIdx.x;            // 72 = h*3 + r
  const int h = hr / 3, r = hr % 3, d = threadIdx.x;
  const int g = h / 6, h6 = h % 6;
  const size_t slot = ((size_t)g * 18 + h6 * 3 + r) * 258;
  float M = NEG_INF;
  for (int s = 0; s < S; s++) M = fmaxf(M, ws[(size_t)s * (4 * 18) * 258 + slot + 256]);
  float L_ = 0.0f, O = 0.0f;
  for (int s = 0; s < S; s++) {
    const float* w = ws + (size_t)s * (4 * 18) * 258 + slot;
    const float wt = exp2f(w[256] - M);
    L_ += wt * w[257];
    O += wt * w[d];
  }
  const float gt = gate[d + (size_t)h * 512 + (size_t)r * 12288 + 256];
  out[d + (size_t)h * 256 + (size_t)r * 6144] = (O / L_) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));
}
