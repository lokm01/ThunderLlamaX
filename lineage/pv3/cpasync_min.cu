// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// cp.async forensics: minimal single-CTA test. out[0] = smem roundtrip of in[0..15].
#include <cuda_fp16.h>
#define CPASYNC(dst, src) asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"((unsigned long long)(src)))
extern "C" __global__ void __launch_bounds__(32) cpasync_min(const float* __restrict__ in, float* __restrict__ out, const int dummy)
{
  __shared__ __align__(16) float sm[16];
  const int tid = threadIdx.x;
  if (tid == 0) {
    CPASYNC(&sm[0], &in[0]);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
  }
  __syncthreads();
  out[tid] = sm[tid] + 1.0f;
}
