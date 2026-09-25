// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// device-side fill: slot i -> uint4(i, 3i+1, 5i+2, 7i+3) (mod 2^32)
// flat indexing, hardcoded 256 threads/CTA (blockDim reads 0 on the dext).
extern "C" __global__ void __launch_bounds__(256) r7b_fill(uint4* __restrict__ dst)
{
  const unsigned long long i = (unsigned long long)blockIdx.x * 256 + threadIdx.x;
  const unsigned a = (unsigned)i;
  dst[i] = make_uint4(a, a * 3u + 1u, a * 5u + 2u, a * 7u + 3u);
}
