// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P7-A Stage-0 probe 4: LDMATRIX — ldmatrix.sync.aligned.m8n8.x4.shared.b16
// vs 4x plain LDS.32 fragment loads (the HMMA-feeding path).
// smem: 512B = 4 tiles x 8 rows x 16B (8 halfs). x4 addressing: lane L supplies
// the row address of matrix (L>>3), row (L&7): byte (L>>3)*64 + (L&7)*8.
// Result: lane L reg j (j=0..3) = u32 at tile j, row (L>>2), col-pair
// (L&3): byte j*128 + (L>>2)*16 + (L&3)*4. LDS.32 mode reads the same 4 u32s.
// mode 0 = ldmatrix loop, mode 1 = 4x ld.shared.u32 loop. All integer
// checksums (u32 sums) -> numpy-exact.
// Laws: hardcoded NTHR, single smem array 16B-aligned, sequential loops,
// per-kernel cubin, nw-token name.
#include <cuda_runtime.h>
#include <cstdint>

#ifndef NTHR
#define NTHR 256
#endif
#define NW (NTHR / 32)

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const uint4* __restrict__ src, uint32_t* __restrict__ out4,
    uint32_t* __restrict__ outacc, int iters, int mode) {
  __shared__ __align__(16) char SM[512];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  // stage 256B = 16 uint4, coalesced
  for (int i = tid; i < 32; i += NTHR) ((uint4*)SM)[i] = src[i];
  __syncthreads();
  const unsigned laddr =
      (unsigned)__cvta_generic_to_shared(SM + (lane >> 3) * 128 + (lane & 7) * 16);
  // LDS.32 fragment addresses: matrix j, row (L>>2), col-pair (L&3)
  unsigned a0, a1, a2, a3;
  {
    const unsigned b0 = (unsigned)__cvta_generic_to_shared(
        SM + (lane >> 2) * 16 + (lane & 3) * 4);
        a0 = b0; a1 = b0 + 128; a2 = b0 + 256; a3 = b0 + 384;
  }
  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  uint32_t acc = 0;
  if (mode == 0) {
    for (int i = 0; i < iters; ++i) {
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
          : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
          : "r"(laddr));
      acc += r0 + r1 + r2 + r3;
    }
  } else {
    for (int i = 0; i < iters; ++i) {
      asm volatile("ld.shared.u32 %0, [%4];\nld.shared.u32 %1, [%5];\n"
                   "ld.shared.u32 %2, [%6];\nld.shared.u32 %3, [%7];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3));
      acc += r0 + r1 + r2 + r3;
    }
  }
  out4[tid * 4 + 0] = r0; out4[tid * 4 + 1] = r1;
  out4[tid * 4 + 2] = r2; out4[tid * 4 + 3] = r3;
  outacc[tid] = acc;
}
