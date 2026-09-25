// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
extern "C" __global__ void test_write2(float* out, int n) {
  if (threadIdx.x < n) out[threadIdx.x] = 42.0f + (float)threadIdx.x;
}
