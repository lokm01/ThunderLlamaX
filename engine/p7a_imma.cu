// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P7-A Stage-0 probe 1: INT8 TENSOR-CORE MMA (IMMA) on the TinyGPU dext.
// Primary: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
// Fallback: mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32  (-DMMA8816=1)
// U8 variant: -DU8=1 swaps .s8 operands for .u8.
// Laws: hardcoded NTHR (no blockDim reads), sequential loops, no smem,
// per-kernel cubin, nw-token in the cubin name, grid/iters as launch args.
// Fragments are pre-packed by the host (per-lane register bytes), so the
// kernel is shape-exact: validate iters=1 vs numpy int32 reference.
#include <cuda_runtime.h>
#include <cstdint>

#ifndef NTHR
#define NTHR 128
#endif
#ifndef ILP
#define ILP 1
#endif

#if MMA8816
__device__ __forceinline__ void imma8816(int &c0, int &c1,
                                         const uint32_t a0, const uint32_t b0) {
#if U8
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "
#else
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
#endif
               "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
               : "+r"(c0), "+r"(c1)
               : "r"(a0), "r"(b0));
}
#else
__device__ __forceinline__ void imma16832(int &c0, int &c1, int &c2, int &c3,
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1) {
#if U8
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
#else
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
#endif
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const uint32_t* __restrict__ afr, const uint32_t* __restrict__ bfr,
    int* __restrict__ cout, int iters) {
  const int tid = threadIdx.x;
#if MMA8816
  // A 8x16 s8 (1 reg/lane), B 16x8 s8 (1 reg/lane), C 8x8 s32 (2 regs/lane)
  const uint32_t a0 = afr[tid];
  const uint32_t b0 = bfr[tid];
#if ILP >= 2
  const uint32_t e0 = a0 ^ 0x01010101u;
#endif
  int c0 = 0, c1 = 0;
#if ILP >= 2
  int d0 = 0, d1 = 0;
#endif
#if ILP >= 4
  int f0 = 0, f1 = 0, g0 = 0, g1 = 0;
#endif
  for (int i = 0; i < iters; ++i) {
    imma8816(c0, c1, a0, b0);
#if ILP >= 2
    imma8816(d0, d1, e0, b0);
#endif
#if ILP >= 4
    imma8816(f0, f1, a0, e0);   // operand-swap chain, still consumed below
    imma8816(g0, g1, e0, e0);
#endif
  }
#if ILP >= 2
  c0 += d0; c1 += d1;
#endif
#if ILP >= 4
  c0 += f0 + g0; c1 += f1 + g1;
#endif
  cout[tid*2+0] = c0; cout[tid*2+1] = c1;
#else
  // A 16x32 s8 (4 regs/lane), B 32x8 s8 (2 regs/lane), C 16x8 s32 (4 regs/lane)
  const uint32_t a0 = afr[tid*4+0], a1 = afr[tid*4+1];
  const uint32_t a2 = afr[tid*4+2], a3 = afr[tid*4+3];
  const uint32_t b0 = bfr[tid*2+0], b1 = bfr[tid*2+1];
#if ILP >= 2
  const uint32_t e0 = a0 ^ 0x01010101u, e1 = a1 ^ 0x01010101u;
  const uint32_t e2 = a2 ^ 0x01010101u, e3 = a3 ^ 0x01010101u;
#endif
  int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if ILP >= 2
  int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#endif
#if ILP >= 4
  int f0 = 0, f1 = 0, f2 = 0, f3 = 0;
  int g0 = 0, g1 = 0, g2 = 0, g3 = 0;
#endif
  for (int i = 0; i < iters; ++i) {
    imma16832(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#if ILP >= 2
    imma16832(d0, d1, d2, d3, e0, e1, e2, e3, b0, b1);
#endif
#if ILP >= 4
    imma16832(f0, f1, f2, f3, a0, a1, a2, a3, b1, b0);
    imma16832(g0, g1, g2, g3, e0, e1, e2, e3, b1, b0);
#endif
  }
#if ILP >= 2
  c0 += d0; c1 += d1; c2 += d2; c3 += d3;
#endif
#if ILP >= 4
  c0 += f0 + g0; c1 += f1 + g1; c2 += f2 + g2; c3 += f3 + g3;
#endif
  cout[tid*4+0] = c0; cout[tid*4+1] = c1;
  cout[tid*4+2] = c2; cout[tid*4+3] = c3;
#endif
}
