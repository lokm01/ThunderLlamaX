// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
extern "C" __global__ void valcheck(float* o, const int a, const int b, const int c, const int d) {
  if (threadIdx.x == 0 && blockIdx.x == 0) { o[0]=a; o[1]=b; o[2]=c; o[3]=d; }
}
