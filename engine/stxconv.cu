// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 M1-A: GDN conv-state copy, one block per launch (grid=1, 256 thr).
// conv4 slot window <-> trunk conv{i}_par (30720 f32 = 3*10240).
extern "C" __global__ void __launch_bounds__(256) stxconv(
    const float* __restrict__ src, float* __restrict__ dst)
{
  for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
}
