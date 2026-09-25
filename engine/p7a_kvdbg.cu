// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P7-A Stage-0 probe 3: INT8-KV SLAB STREAM, pfa16 layout.
//   kv[g][row][256B] with g in 0..7 (K groups 0-3, V groups 4-7), CTXK rows
//   per group; sc[g][row][8 halfs] scales (group stride CTXK*8 HALF units ->
//   sc linear HALF index = global_row*8 + seg, no division needed).
//   8*CTXK*256B = 205.6MB @CTXK=100352 — the representative 200MB slice.
// Each CTA streams a CONTIGUOUS linear slice; each warp a contiguous
// sub-slice; 32 lanes x uint4 = 512B (2 rows) per warp-step; #pragma unroll 8
// keeps 8 loads in flight per lane (ring-free: guarded range, exact coverage).
// Consumption per uint4: u32 word sums + 4x dp4a(int8) + the u16 scale
// (integer-only checksum -> numpy-exact replication).
// Laws: hardcoded NTHR (blockDim/gridDim never read — n_ctas is an arg),
// no grid-stride, sequential loops, single smem array, full-warp shfl.
#include <cuda_runtime.h>
#include <cstdint>

#ifndef NTHR
#define NTHR 1024
#endif
#ifndef CTXK
#define CTXK 100352
#endif
#define NW (NTHR / 32)

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ kv, const unsigned char* __restrict__ sc,
    uint32_t* __restrict__ out, int n_ctas, int reps) {
  __shared__ __align__(16) char SM[NW * 64];
  const int warp = blockIdx.x * NW + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int warps_total = n_ctas * NW;
  const int TOTU = 8 * CTXK * 16;   // total uint4 units in the slab
  const int per = TOTU / warps_total;
  const int u0 = warp * per;
  const int u1 = (warp == warps_total - 1) ? TOTU : u0 + per;
  const uint4* base = (const uint4*)kv;
  const int q8 = 0x01010101u;   // dp4a operand -> sum of signed bytes
  uint32_t acc = 0;
  for (int rep = 0; rep < reps; ++rep) {
    _Pragma("unroll 8")
    for (int u = u0; u < u1; ++u) {
      const uint4 v = base[u];
#if P7A_NODP
      int dp = 0;
#else
      int dp = __dp4a((int)v.x, q8, 0);
      dp = __dp4a((int)v.y, q8, dp);
      dp = __dp4a((int)v.z, q8, dp);
      dp = __dp4a((int)v.w, q8, dp);
#endif
      acc += v.x + v.y + v.z + v.w + (uint32_t)dp;
#if !P7A_NOSC
      const int rg = u >> 4;          // global row (kv linear, group-major)
      const int seg = (u & 15) >> 1;  // 32B scale segment within the row
#if P7A_SC32
      // batched: one u32 per seg-PAIR (halves scale-load count)
      acc += ((const uint32_t*)(sc + (size_t)rg * 16))[seg >> 1];
#else
      acc += *(const uint16_t*)(sc + 2 * ((size_t)rg * 8 + seg));
#endif
#endif
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
    _Pragma("unroll")
    for (int i = 0; i < NW; ++i) out[256 + blockIdx.x * NW + i] = ((uint32_t*)SM)[i];
  }
}
