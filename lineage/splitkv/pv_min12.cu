// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
extern "C" __global__ void __launch_bounds__(128) pv_min12(
    float* o, float* a1, float* a2, float* a3, float* a4, float* a5, float* a6,
    float* a7, float* a8, float* a9, float* a10, float* a11, const int v) {
  o[threadIdx.x + (blockIdx.z << 7)] = (float)v + a1[0] + a11[1];
}
