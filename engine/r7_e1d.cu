// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// R7 E1 v3: issue-law microbench arms. DEXT LAW (R7, bisected): QMD
// register_count_v<=8 + LDG.E.128 = "Out Of Range Register" SM fault (b5 regs=8
// faulted, b11 regs=11 with the IDENTICAL load instruction ran clean). Every
// arm therefore pins 16 registers (unsinkable via asm "" ties) — no extra
// instructions, only file slots (occupancy unaffected at 1 CTA/SM).
// Full-lane consumption law: use only v.x and nvcc DCEs the uint4 to 4B loads.
#include <cstdint>
#include <cuda_fp16.h>
#define T 256
#define SLICE_U4 196608u   // 256MB / 82 CTAs
#ifndef ITER_A
#define ITER_A 192
#endif
#ifndef ITER_B
#define ITER_B 213
#endif
#ifndef ITER_C
#define ITER_C 320
#endif
#ifndef ITER_D
#define ITER_D 256
#endif
#ifndef ITER_E
#define ITER_E 192
#endif
#ifndef ITER_F
#define ITER_F 96
#endif

__device__ __forceinline__ unsigned u4acc(const uint4 v, unsigned a) {
  return a ^ v.x ^ v.y ^ v.z ^ v.w;
}

__device__ __forceinline__ void hmma16816(float &c0, float &c1, float &c2, float &c3,
                                          const unsigned a0, const unsigned a1,
                                          const unsigned a2, const unsigned a3,
                                          const unsigned b0, const unsigned b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------- ARM A: independent 16B loads, ALL lanes consumed ----------------


#ifdef E1A
extern "C" __global__ void __launch_bounds__(T) e1a_ldg_nw8(const uint4* __restrict__ src, unsigned* out)
{
  const uint4* p = src + (size_t)blockIdx.x * SLICE_U4 + (threadIdx.x << 3);
  unsigned acc = threadIdx.x;
  #pragma unroll 4
  for (int it = 0; it < ITER_A; ++it) {
    acc = u4acc(p[0], acc); acc = u4acc(p[1], acc);
    acc = u4acc(p[2], acc); acc = u4acc(p[3], acc);
    acc = u4acc(p[4], acc); acc = u4acc(p[5], acc);
    acc = u4acc(p[6], acc); acc = u4acc(p[7], acc);
    p += (T << 3);
  }
  if ((acc) == 0xdeadbeefu) out[blockIdx.x] = acc;
}
#endif // E1A

#ifdef E1A0
extern "C" __global__ void __launch_bounds__(T) e1a0_ldg_nw8(const uint4* __restrict__ src, unsigned* out)
{
  const unsigned long long o = (unsigned long long)blockIdx.x * 4;
  const unsigned t0 = (unsigned)clock64();
  if (threadIdx.x == 0) { out[o+0] = blockIdx.x; out[o+1] = t0; }
  const uint4* p = src + (size_t)blockIdx.x * SLICE_U4 + (threadIdx.x << 3);
  unsigned acc = threadIdx.x;
  #pragma unroll 4
  for (int it = 0; it < ITER_A; ++it) {
    acc = u4acc(p[0], acc); acc = u4acc(p[1], acc);
    acc = u4acc(p[2], acc); acc = u4acc(p[3], acc);
    acc = u4acc(p[4], acc); acc = u4acc(p[5], acc);
    acc = u4acc(p[6], acc); acc = u4acc(p[7], acc);
    p += (T << 3);
  }
  if (threadIdx.x == 0) { out[o+2] = (unsigned)clock64(); out[o+3] = acc; }
}
#endif // E1A0

#ifdef E1B
extern "C" __global__ void __launch_bounds__(T) e1b_fma_nw8(const uint4* __restrict__ src, unsigned* out)
{
  const unsigned s0 = threadIdx.x * 3 + 1, s1 = threadIdx.x * 5 + 2;
  float a0 = (float)s0, a1 = (float)s1, a2 = (float)(s0 ^ s1), a3 = 1.0f;
  const float w = 1.0000001f, c = 0.5f;
  #pragma unroll 4
  for (int it = 0; it < ITER_B; ++it) {
    a0 = fmaf(a0, w, c); a1 = fmaf(a1, w, c); a2 = fmaf(a2, w, c); a3 = fmaf(a3, w, c);
    a0 = fmaf(a0, w, c); a1 = fmaf(a1, w, c); a2 = fmaf(a2, w, c); a3 = fmaf(a3, w, c);
    a0 = fmaf(a0, w, c); a1 = fmaf(a1, w, c); a2 = fmaf(a2, w, c); a3 = fmaf(a3, w, c);
    a0 = fmaf(a0, w, c); a1 = fmaf(a1, w, c); a2 = fmaf(a2, w, c); a3 = fmaf(a3, w, c);
  }
  const unsigned u = (unsigned)(a0 + a1 + a2 + a3);
  if (u == 0xdeadbeefu) out[blockIdx.x] = u;
}
#endif // E1B

#ifdef E1C
extern "C" __global__ void __launch_bounds__(T) e1c_hmma_nw8(const uint4* __restrict__ src, unsigned* out)
{
  const unsigned ah = threadIdx.x * 3 + 1, bh = threadIdx.x * 7 + 3;
  const unsigned a0 = (ah << 16) | (ah & 0xffffu), a1 = ~a0;
  const unsigned a2 = (bh << 16) | (bh & 0xffffu), a3 = ~a2;
  const unsigned b0 = (ah << 16) | (bh & 0xffffu), b1 = ~b0;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  float d0 = 1.f, d1 = 2.f, d2 = 3.f, d3 = 4.f;
  #pragma unroll 4
  for (int it = 0; it < ITER_C; ++it) {
    hmma16816(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    hmma16816(d0, d1, d2, d3, a1, a0, a3, a2, b1, b0);
  }
  const unsigned u = (unsigned)(c0 + c1 + c2 + c3 + d0 + d1 + d2 + d3);
  if (u == 0xdeadbeefu) out[blockIdx.x] = u;
}
#endif // E1C

#ifdef E1D
extern "C" __global__ void __launch_bounds__(T) e1d_mix_nw8(const uint4* __restrict__ src, unsigned* out)
{
  const uint4* p = src + (size_t)blockIdx.x * SLICE_U4 + (threadIdx.x << 1);
  const unsigned lane = threadIdx.x & 31;
  const unsigned ah = threadIdx.x * 3 + 1, bh = threadIdx.x * 7 + 3;
  const unsigned a0 = (ah << 16) | (ah & 0xffffu), a1 = ~a0;
  const unsigned a2 = (bh << 16) | (bh & 0xffffu), a3 = ~a2;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  unsigned acc = threadIdx.x;
  #pragma unroll 4
  for (int it = 0; it < ITER_D; ++it) {
    const uint4 u0 = p[0], u1 = p[1];
    acc = u4acc(u0, acc); acc = u4acc(u1, acc);
    const unsigned q0 = u0.x & 0xFFFFu, q1 = u0.x >> 16;
    const unsigned sw = u0.z, d = u0.w & 0xFFFFu;
    unsigned sidx = (sw >> (7u * (lane & 3))) & 0x7Fu;
    sidx ^= (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6);
    const __half h0 = __float2half((float)((int)(q0 & 0xff) - 128) * (int)(sidx & 1));
    const __half h1 = __float2half((float)((int)(q1 & 0xff) - 128) * (int)((sidx>>1) & 1));
    const unsigned b0f = (__half_as_ushort(h1) << 16) | __half_as_ushort(h0);
    const __half h2 = __float2half((float)((int)(u1.x & 0xff) - 128) * (int)((sidx>>2) & 1));
    const __half h3 = __float2half((float)((int)(u1.x >> 24) - 128) * (int)((sidx>>3) & 1));
    const unsigned b1f = (__half_as_ushort(h3) << 16) | __half_as_ushort(h2);
    hmma16816(c0, c1, c2, c3, a0, a1, a2, a3, b0f, b1f);
    c0 += (float)(d & 1u) + (float)acc * 0.f;
    p += (T << 1);
  }
  const unsigned u = ((unsigned)(c0 + c1 + c2 + c3) + acc);
  if (u == 0xdeadbeefu) out[blockIdx.x] = u;
}
#endif // E1D

#ifdef E1E
extern "C" __global__ void __launch_bounds__(T) e1e_copy_nw8(const uint4* __restrict__ src, uint4* __restrict__ dst, unsigned* out)
{
  const uint4* p = src + (size_t)blockIdx.x * SLICE_U4 + (threadIdx.x << 2);
  uint4* q = dst + (size_t)blockIdx.x * SLICE_U4 + (threadIdx.x << 2);
  unsigned g = threadIdx.x;
  #pragma unroll 4
  for (int it = 0; it < ITER_E; ++it) {
    const uint4 v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3];
    q[0] = v0; q[1] = v1; q[2] = v2; q[3] = v3;
    g ^= v0.x ^ v1.y ^ v2.z ^ v3.w;
    p += (T << 2); q += (T << 2);
  }
  if ((g) == 0xdeadbeefu) out[blockIdx.x] = g;
}
#endif // E1E

#ifdef E1F
extern "C" __global__ void __launch_bounds__(T) e1f_dldg_nw8(const uint4* __restrict__ src, unsigned* out)
{
  const uint4* base = src + (size_t)blockIdx.x * SLICE_U4;
  uint4 v = base[threadIdx.x];
  unsigned acc = threadIdx.x;
  #pragma unroll 4
  for (int it = 0; it < ITER_F; ++it) {
    v = base[((v.x ^ v.y ^ acc) & 0x2FFFFu)];
    v = base[((v.z ^ v.w ^ acc) & 0x2FFFFu)];
    acc ^= v.x + v.y + v.z + v.w;
  }
  if ((acc) == 0xdeadbeefu) out[blockIdx.x] = acc;
}
#endif // E1F

// ---------------- ARM G: empty kernel (launch+wait floor) ----------------
#ifdef E1G
extern "C" __global__ void __launch_bounds__(T) e1g_empty_nw8(const uint4* __restrict__ src, unsigned* out)
{
  if ((threadIdx.x ^ blockIdx.x) == 0xdeadbeefu) out[blockIdx.x] = threadIdx.x;
}
#endif
