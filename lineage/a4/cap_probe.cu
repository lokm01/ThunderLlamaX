// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01

extern "C" __global__ void __launch_bounds__(32) cap_probe(const float* __restrict__ a, float* __restrict__ b, const int dummy)
{ if (threadIdx.x == 0) b[0] = a[0]; }
