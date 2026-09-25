// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// smem-image probe: after staging tile 0 (8 positions), write the sV rows
// directly to ws. ws[(0,0,h6)] = sV[h6] image, ws[(0,1,h6)] = sV[6+h6] for h6<2.
#include <cuda_fp16.h>
#ifndef LMAX
#define LMAX 100352
#endif
extern "C" __global__ void __launch_bounds__(256) k1dbg(
    float* __restrict__ P, const __half* __restrict__ KV, float* __restrict__ ws, const int sp)
{
  const int g = blockIdx.x & 3;
  const int start = 0;
  const int cend = min(8, sp + 1);
  const __half* Vg = KV + (size_t)g * (LMAX * 256) + (size_t)LMAX * 1024;
  __shared__ __align__(16) __half sV[8][256];
  const int tid = threadIdx.x, lane = tid & 31;
  const int spr = tid >> 5, sdc = (tid & 31) << 3;
  struct __align__(16) u4 { unsigned x, y, z, w; };
  if (spr < cend) *(u4*)&sV[spr][sdc] = *(const u4*)(Vg + (size_t)(start + spr) * 256 + sdc);
  __syncthreads();
  const int pi = tid >> 5;
  if (pi < cend) {
    float* w = ws + ((pi < 6) ? ((size_t)(0*4+0)*6 + pi) * 256 : ((size_t)(0*4+1)*6 + (pi-6)) * 256);
    #pragma unroll
    for (int j = 0; j < 8; j++) w[lane + (j << 5)] = __half2float(sV[pi][lane + (j << 5)]);
  }
}
