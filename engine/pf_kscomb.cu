// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P4 Stage 1b: K-split partials combine. out16[m][n] = half(p0+p1+p2+p3)
// summed in ks order 0..3 (the documented accumulation order; each partial is
// itself the in-order fp32 sum over its K-span, so the global order is the
// same left-to-right stream as the single-CTA kernel). Hardcoded M=16 N=1024
// KS=4. Flat indexing, 256thr, full guard.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) pf_kscomb4(
    const float* __restrict__ part, __half* __restrict__ out16)
{
  const int i = threadIdx.x + blockIdx.x * 256;
  if (i < 16*1024) {
    const float* p = part + i;
    const float acc = p[0*16384] + p[1*16384] + p[2*16384] + p[3*16384];
    out16[i] = __float2half(acc);
  }
}
