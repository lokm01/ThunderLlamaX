// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P13 kernel A — the per-SM steady-state stream microbench (MECHANISM PROBE).
// Question: is the shipped GEMM family's 169-265 GB/s a WAVE/RAMP artifact
// (per-launch CTA cold-start, L2 warmup, tail drain) or a STEADY-STATE per-SM
// stream limit? Answer: grid = NCTA (82 = one CTA/SM, exactly one wave, no
// rotation), each CTA streams a contiguous parcel with a RINGD-deep uint4
// register ring, consuming via XOR (no smem, no syncs, no mma).
//   - steady state >=400 GB/s aggregate  -> persistence pays (wave/ramp artifact)
//   - steady state ~265                  -> KILL (per-SM stream limit)
// NCTA=272 variant reproduces the shipped 3.3-wave pure-stream reference.
// LAWS: no gridDim/blockDim reads (NCTA is a compile-time literal), ring[]
// only compile-time indexed (unrolled), sequential loops, never-true sink
// write keeps every lane's loads alive (no DCE).
// Build: -DNTHR -DRINGD -DNCTA
#include <cuda_fp16.h>
#ifndef KNAME
#define KNAME pf13_stream
#endif
#ifndef NCTA
#define NCTA 82
#endif
#ifndef RINGD
#define RINGD 8
#endif

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const uint4* __restrict__ src, unsigned int* __restrict__ sink, const int nu16)
{
  const unsigned int tid = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * (size_t)nu16;
  const int nsteps = nu16 / NTHR;   // host guarantees nu16 % NTHR == 0, nsteps % RINGD == 0
  uint4 ring[RINGD];
  unsigned int acc = 0u;
  _Pragma("unroll")
  for (int r = 0; r < RINGD; ++r)
    ring[r] = src[base + (size_t)tid + (size_t)r * NTHR];
  _Pragma("unroll 1")
  for (int s = 0; s + RINGD <= nsteps; s += RINGD) {
    _Pragma("unroll")
    for (int r = 0; r < RINGD; ++r) {
      const uint4 v = ring[r];                       // consume (WAR: load below waits this)
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
      const int nxt = s + RINGD + r;                 // prefetch RINGD steps ahead
      if (nxt < nsteps) ring[r] = src[base + (size_t)tid + (size_t)nxt * NTHR];
    }
  }
  if (acc == 0x12345678u) sink[blockIdx.x] = acc;    // never-true sink: no DCE
}
