// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
extern "C" __global__ void __launch_bounds__(256) ps_probe_t(
    float* __restrict__ b0, float* __restrict__ b1, float* __restrict__ b2,
    float* __restrict__ b3, float* __restrict__ b4, float* __restrict__ b5,
    float* __restrict__ b6, float* __restrict__ b7, const int T)
{
  if (threadIdx.x == 0) {
    b0[0] = (float)T;
    b0[1] = (float)(int)(size_t)b1;
    b0[2] = 1.0f;
  }
}
