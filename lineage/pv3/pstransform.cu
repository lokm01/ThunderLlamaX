// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(49) pstransform(float* data0, float* data1, float* data2) {
  __shared__ __align__(16) float buf1[49];
  int gidx0 = blockIdx.x; /* 72 */
  float val0 = (*(data2+gidx0));
  int gidx1 = blockIdx.y; /* 256 */
  int lidx0 = threadIdx.x; /* 49 */
  int alu0 = ((gidx0*100352)+(gidx1*392)+(lidx0<<3));
  float s = 0.0f;
  #pragma unroll
  for (int k = 0; k < 8; k++) {
    float e = exp2f(((data1[alu0+k]-val0)*1.4426950216293335f));
    data1[alu0+k] = e;
    s += e;
  }
  buf1[lidx0] = s;
  __syncthreads();
  if (lidx0 == 0) {
    float t = 0.0f;
    for (int i = 0; i < 49; i++) t += buf1[i];
    *(data0+(gidx1+(gidx0<<8))) = t;
  }
}
