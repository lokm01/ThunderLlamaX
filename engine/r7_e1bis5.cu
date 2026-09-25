// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cstdint>
extern "C" __global__ void __launch_bounds__(256) b5_ld1(const uint4* src, unsigned* out)
{
  const uint4 v = src[threadIdx.x];
  if ((v.x ^ v.y ^ v.z ^ v.w) == 0xdeadbeefu) out[blockIdx.x] = v.x;
}
extern "C" __global__ void __launch_bounds__(256) b6_ld4(const unsigned* src, unsigned* out)
{
  const unsigned v = src[threadIdx.x];
  if (v == 0xdeadbeefu) out[blockIdx.x] = v;
}
extern "C" __global__ void __launch_bounds__(256) b7_ld8(const uint2* src, unsigned* out)
{
  const uint2 v = src[threadIdx.x];
  if ((v.x ^ v.y) == 0xdeadbeefu) out[blockIdx.x] = v.x;
}
