// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 M1-A: GDN recurrent-state copy, one block per launch (grid=1, 256 thr).
// rec4 slot window <-> trunk rec{i} (786432 f32 = 48*128*128). Direction-agnostic
// (src->dst); python passes the right window. Sequential strided loop, full warp.
extern "C" __global__ void __launch_bounds__(256) stxrec(
    const float* __restrict__ src, float* __restrict__ dst)
{
  for (int i = threadIdx.x; i < 786432; i += 256) dst[i] = src[i];
}
