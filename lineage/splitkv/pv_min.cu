// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
extern "C" __global__ void __launch_bounds__(128) pv_min(float* out, const int v) {
  out[threadIdx.x + (blockIdx.z << 7)] = (float)v;
}
