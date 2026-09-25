// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R7 LAW CONTROL: the same b5_ld1 body as r7_e1bis5.cu, built as a
// SINGLE-KERNEL cubin — runs clean, while the multi-kernel r7_e1bis5.cubin
// faulted with "Out Of Range Register" on every SM (wrong-code execution;
// the fork's per-.text.<name> addressing is broken for multi-kernel cubins).
// THE -DKNAME ONE-KERNEL-PER-CUBIN CONVENTION IS LOAD-BEARING.
#include <cstdint>
extern "C" __global__ void __launch_bounds__(256) b5_ld1(const uint4* src, unsigned* out)
{
  const uint4 v = src[threadIdx.x];
  if ((v.x ^ v.y ^ v.z ^ v.w) == 0xdeadbeefu) out[blockIdx.x] = v.x;
}
