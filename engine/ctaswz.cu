// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// CTASWZ: CTA/SM co-residency probe (CTASM investigation).
// Per CTA (tid0): records [smid, t_start, t_end, 0] as u64s; all threads spin
// a bounded compile-time cycle count so the CTA holds its residency window.
#include <cstdint>
#define T 128
#ifndef SPIN_CYCLES
#define SPIN_CYCLES 3000000
#endif
#define BODY() \
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); \
  unsigned long long t0 = clock64(); \
  unsigned long long o = (unsigned long long)blockIdx.x * 4; \
  if (threadIdx.x == 0) { out[o+0] = smid; out[o+1] = t0; } \
  unsigned long long t = t0; \
  while (t - t0 < (unsigned long long)SPIN_CYCLES) { asm volatile("" ::: "memory"); t = clock64(); } \
  if (threadIdx.x == 0) out[o+2] = t;

extern "C" __global__ void __launch_bounds__(T) ctaswz(unsigned long long* out)
{ BODY() }
extern "C" __global__ void __launch_bounds__(T) ctaswz8k(unsigned long long* out)
{ __shared__ volatile unsigned long long pad[1024]; if (threadIdx.x==0) pad[0]=blockIdx.x; BODY() }
extern "C" __global__ void __launch_bounds__(T) ctaswz32k(unsigned long long* out)
{ __shared__ volatile unsigned long long pad[4096]; if (threadIdx.x==0) pad[0]=blockIdx.x; BODY() }
