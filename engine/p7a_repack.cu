// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P7-A Stage-0 probe 2: quant-word LOAD STREAM — strided (shipped pGEMM
// stage_w pattern) vs OFFLINE-REPACKED k-chunk-major contiguous 16B runs.
// Real FFN gate IQ3_XXS packed rows: KDIM=5120 -> 20 blocks/row, ROWBYTES=1960
//   row layout: [20x64B qs u16[32]][20x32B sc u32[8]][20x2B d u16] = 1960B.
// Strided kernel replicates stage_w exactly: lane(r=lane>>2, c=lane&3) reads
//   per (row, block b, cc in 0..7): q u16 @ qs[32*b + c*8+cc],
//   sc u32 @ scp[8*b + ((c*8+cc)>>2)]; c==0 lane also reads d u16[b].
// Repacked layout (host permute): unit[b*49+i][rowgroup of 8] u16, where
//   i<32 -> qs u16[32*b+i]; 32<=i<48 -> sc u16[16*b+(i-32)]; i=48 -> d u16[b].
//   49 u16 x 8 rows = 15680B = EXACTLY the 8 rows' bytes (no padding).
//   Warp streams its 8-row group as 980 contiguous uint4 with a RING-deep
//   unrolled register ring (compile-time indexed).
// Laws: hardcoded NTHR, no grid-stride, sequential loops, single smem array
// 16B aligned, full-warp shfl reduce, per-kernel cubin, nw-token name.
#include <cuda_runtime.h>
#include <cstdint>

#ifndef NTHR
#define NTHR 256
#endif
#define NW (NTHR / 32)
#define NROWS 17408
#define ROWB 1960
#define UPG 980   // uint4 units per 8-row group

#if KERNEL == 0
// ---------------- strided baseline ----------------
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w, uint32_t* __restrict__ out, int reps) {
  __shared__ __align__(16) char SM[NW * 64];
  const int warp = blockIdx.x * NW + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int r = lane >> 2, c = lane & 3;
  const int r0 = warp * 8;
  uint32_t acc = 0;
  if (r0 + 8 <= NROWS) {
    const unsigned char* rowp = w + (size_t)(r0 + r) * ROWB;
    for (int rep = 0; rep < reps; ++rep) {
      _Pragma("unroll")
      for (int b = 0; b < 20; ++b) {
        _Pragma("unroll")
        for (int cc = 0; cc < 8; ++cc) {
          const int lc = c * 8 + cc;
          acc += *(const uint16_t*)(rowp + 2 * (32 * b + lc));
          acc += ((const uint32_t*)(rowp + 1280))[8 * b + (lc >> 2)];
        }
        if (c == 0) acc += *(const uint16_t*)(rowp + 1920 + 2 * b);
      }
    }
  }
  _Pragma("unroll")
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  acc = __shfl_sync(0xffffffffu, acc, 0);   // broadcast warp total from lane 0
  if (lane == 0) ((uint32_t*)SM)[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t s = 0;
    _Pragma("unroll")
    for (int i = 0; i < NW; ++i) s += ((uint32_t*)SM)[i];
    out[blockIdx.x] = s;
  }
}
#else
// ---------------- repacked contiguous ----------------
#ifndef RING
#define RING 8
#endif
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const uint4* __restrict__ wr, uint32_t* __restrict__ out, int reps) {
  __shared__ __align__(16) char SM[NW * 64];
  const int warp = blockIdx.x * NW + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int g = warp;               // 8-row group index
  uint32_t acc = 0;
  if (g < NROWS / 8) {
    const uint4* base = wr + (size_t)g * UPG;
    const int steps = UPG >> 5;   // 32 lanes x consecutive uint4 per step
    for (int rep = 0; rep < reps; ++rep) {
      _Pragma("unroll 8")
      for (int s = 0; s < steps; ++s) {
        const int u = (s << 5) + lane;
        const uint4 v = base[u];
        acc += v.x + v.y + v.z + v.w;
      }
      {  // tail: 980 = 30*32 + 20
        const int u = (steps << 5) + lane;
        if (u < UPG) {
          const uint4 v = base[u];
          acc += v.x + v.y + v.z + v.w;
        }
      }
    }
  }
  _Pragma("unroll")
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  acc = __shfl_sync(0xffffffffu, acc, 0);   // broadcast warp total from lane 0
  if (lane == 0) ((uint32_t*)SM)[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t s = 0;
    _Pragma("unroll")
    for (int i = 0; i < NW; ++i) s += ((uint32_t*)SM)[i];
    out[blockIdx.x] = s;
  }
}
#endif
