// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// A3a smoke test: trivial kernel to prove hand-made cubins load+run via fork NVProgram.
typedef unsigned int u32;
extern "C" __global__ void smoke(const float* __restrict__ x, float* __restrict__ y, int n, int c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + (float)c;
}
