// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R7b rung-2a SMOKE: producer/consumer warp-spec handoff primitives on the dext.
// Arms (one kernel per cubin, -D guard):
//   MB_SYNC  : all-warp load+consume, __syncthreads per stage (production shape)
//   MB_NBAR  : 4 producer + 4 consumer warps, named bar.sync meets, 2-buf overlap
//   MB_MBARR : mbarrier full/empty arrive/try_wait pipeline (bounded spin)
//   MB_EMPTY : launch-floor arm
// Correctness: per-stage u64 sums (mod-2^64 associative => grouping-invariant);
// out64[cta] must be IDENTICAL across arms == host-expected. out u32 = flags.
// LAWS: flat indexing, hardcoded sizes, bounded spins (a hang reboots the rig),
// single smem array, full-warp masks, no runtime-indexed locals, 1 kernel/cubin.
#define NSTAGE 384
#define LDG_P 4     // uint4 loads per producer thread per stage (8KB/stage/CTA)
#define LDG_A 2     // uint4 loads per all-warp thread per stage (same 8KB)
#define SLOTS (128 * LDG_P)
#define GUARD (3000000ull)

__device__ __forceinline__ void mbar_init(unsigned long long* m, unsigned count) {
  unsigned a = (unsigned)__cvta_generic_to_shared(m);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* m) {
  unsigned a = (unsigned)__cvta_generic_to_shared(m);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(a));
}
// sm_86 LAW: mbarrier parity must be a COMPILE-TIME immediate (try_wait and the
// register-parity forms are sm_90+); test_wait (non-suspending poll) is the
// only sm_86-legal wait -> two immediates selected by branch.
__device__ __forceinline__ int mbar_try(unsigned long long* m, unsigned par) {
  unsigned a = (unsigned)__cvta_generic_to_shared(m);
  unsigned r;
  if (par) {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], 1; selp.b32 %0, 1, 0, p; }"
                 : "=r"(r) : "r"(a));
  } else {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], 0; selp.b32 %0, 1, 0, p; }"
                 : "=r"(r) : "r"(a));
  }
  return r;
}
__device__ __forceinline__ void nbar(int id, int cnt) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(cnt));
}

extern "C" __global__ void __launch_bounds__(256) r7b_mb(
    const uint4* __restrict__ src, unsigned* __restrict__ out,
    unsigned long long* __restrict__ out64)
{
  __shared__ __align__(16) unsigned long long mb[4];   // mbar: full[2] empty[2]
  __shared__ __align__(16) uint4 sm[2 * SLOTS];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int prod = warp < 4;
  const size_t base = (size_t)blockIdx.x * (NSTAGE * SLOTS);
  unsigned long long acc = 0;

#if MB_EMPTY
  out[blockIdx.x] = 1;
  return;
#endif

#if MB_MBARR
  if (tid == 0) {
    mbar_init(&mb[0], 128); mbar_init(&mb[1], 128);
    mbar_init(&mb[2], 128); mbar_init(&mb[3], 128);
  }
  __syncthreads();
#endif

#if MB_SYNC
  for (int s = 0; s < NSTAGE; ++s) {
    _Pragma("unroll")
    for (int j = 0; j < LDG_A; ++j) sm[tid + j * 256] = src[base + (size_t)s * SLOTS + tid + j * 256];
    __syncthreads();
    _Pragma("unroll")
    for (int j = 0; j < LDG_A; ++j) {
      const uint4 v = sm[tid + j * 256];
      acc += (unsigned long long)v.x + v.y + v.z + v.w;
    }
    __syncthreads();
  }
  out[blockIdx.x] = 1;
#elif MB_NBAR
  // named-barrier pipeline: meets separate stages; produce(s+1) overlaps consume(s)
  if (prod) {
    _Pragma("unroll")
    for (int j = 0; j < LDG_P; ++j) sm[tid + j * 128] = src[base + (size_t)tid + j * 128];
    for (int s = 0; s < NSTAGE; ++s) {
      const int b = s & 1;
      nbar(1 + b, 256);
      if (s + 1 < NSTAGE) {
        uint4* dst = sm + (b ^ 1) * SLOTS;
        _Pragma("unroll")
        for (int j = 0; j < LDG_P; ++j)
          dst[tid + j * 128] = src[base + (size_t)(s + 1) * SLOTS + tid + j * 128];
      }
    }
  } else {
    const int ct = tid - 128;
    for (int s = 0; s < NSTAGE; ++s) {
      const int b = s & 1;
      nbar(1 + b, 256);
      _Pragma("unroll")
      for (int j = 0; j < LDG_P; ++j) {
        const uint4 v = sm[b * SLOTS + ct + j * 128];
        acc += (unsigned long long)v.x + v.y + v.z + v.w;
      }
    }
  }
  out[blockIdx.x] = 1;
#elif MB_MBARR
  if (prod) {
    for (int s = 0; s < NSTAGE; ++s) {
      const int b = s & 1;
      if (s >= 2) {
        const unsigned parE = (unsigned)(((s >> 1) - 1) & 1);
        int ok = 0;
        for (unsigned long long it = 0; it < GUARD; ++it) { if (mbar_try(&mb[2 + b], parE)) { ok = 1; break; } }
        if (!ok) { atomicOr(out + blockIdx.x, 0x80000000u); return; }
      }
      uint4* dst = sm + b * SLOTS;
      _Pragma("unroll")
      for (int j = 0; j < LDG_P; ++j)
        dst[tid + j * 128] = src[base + (size_t)s * SLOTS + tid + j * 128];
      __threadfence_block();
      mbar_arrive(&mb[b]);
    }
  } else {
    for (int s = 0; s < NSTAGE; ++s) {
      const int b = s & 1;
      const unsigned parF = (unsigned)((s >> 1) & 1);
      int ok = 0;
      for (unsigned long long it = 0; it < GUARD; ++it) { if (mbar_try(&mb[b], parF)) { ok = 1; break; } }
      if (!ok) { atomicOr(out + blockIdx.x, 0x40000000u); return; }
      _Pragma("unroll")
      for (int j = 0; j < LDG_P; ++j) {
        const uint4 v = sm[b * SLOTS + (tid - 128) + j * 128];
        acc += (unsigned long long)v.x + v.y + v.z + v.w;
      }
      __threadfence_block();
      mbar_arrive(&mb[2 + b]);
    }
  }
  out[blockIdx.x] = 1;
#endif
#if MB_SYNC
  atomicAdd(out64 + blockIdx.x, acc);
#else
  if (!prod) atomicAdd(out64 + blockIdx.x, acc);
#endif
}
