// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
extern "C" __global__ void __launch_bounds__(256) k_triv(float* __restrict__ a, float* __restrict__ o, const int n, const int nthreads)
{
    const int tid = (blockIdx.x << 8) + threadIdx.x;
    if (tid >= nthreads) return;
    float acc = 0.f;
    for (int i = tid; i < n; i += nthreads) acc += a[i] + 1.0f;
    o[tid & 1023] = acc;
}
