// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
__device__ __forceinline__ unsigned h2u(const __half2 h) { return *reinterpret_cast<const unsigned*>(&h); }
extern "C" __global__ void mma_micro(const __half* A, const __half* B, float* D) {
  // A[16][16], B[16][8], D[16][8]; single warp; lane-agnostic fill per current assumption
  const int lane = threadIdx.x & 31;
  const unsigned a0 = *(const unsigned*)(A + (lane>>2)*16 + (lane&3)*2);
  const unsigned a1 = *(const unsigned*)(A + (lane>>2)*16 + (lane&3)*2 + 8);
  const unsigned a2 = *(const unsigned*)(A + ((lane>>2)+8)*16 + (lane&3)*2);
  const unsigned a3 = *(const unsigned*)(A + ((lane>>2)+8)*16 + (lane&3)*2 + 8);
  const unsigned b0 = h2u(__halves2half2(B[((lane&3)*2)*8 + (lane>>2)], B[(((lane&3)*2)+1)*8 + (lane>>2)]));
  const unsigned b1 = h2u(__halves2half2(B[(((lane&3)*2)+8)*8 + (lane>>2)], B[(((lane&3)*2)+9)*8 + (lane>>2)]));
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  const int m = lane >> 2, n = (lane & 3)*2;
  D[m*8 + n] = c0; D[m*8 + n + 1] = c1;
  D[(m+8)*8 + n] = c2; D[(m+8)*8 + n + 1] = c3;
}
