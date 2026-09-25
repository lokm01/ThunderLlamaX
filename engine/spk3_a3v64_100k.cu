// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// K1S v3: tile-unrolled double buffer (all register indices compile-time; the
// v2 runtime ring index demoted arrays to local memory). Same per-row op order.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define RMAX (6*ROWS)

#define PROC_ROW(DSTm, DSTs, DSTa, Q, ACT) { \
  float sc = 0.f; \
  _Pragma("unroll") for (int j = 0; j < 8; ++j) sc += Q[j] * kr0[j]; \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(FULL, sc, o); \
  if (ACT) { \
    if (sc > DSTm) { const float cor = expf(DSTm - sc); DSTs *= cor; _Pragma("unroll") for (int j=0;j<8;++j) DSTa[j] *= cor; DSTm = sc; } \
    const float p = expf(sc - DSTm); DSTs += p; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) DSTa[j] += p * vr0[j]; \
  } }

extern "C" __global__ void __launch_bounds__(256) K1S(
    const __half* __restrict__ kv, const float* __restrict__ qw, const int* __restrict__ pos_slot,
    float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
{
  const int g = blockIdx.x / S;
  const int s = blockIdx.x % S;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = pos_slot[0];
  const int l0 = s * CH;
  const int l1 = (l0 + CH) < (pos + ROWS) ? (l0 + CH) : (pos + ROWS);
  const __half* Kc = kv + (size_t)g * (CTXK*256);
  const __half* Vc = kv + (size_t)(4 + g) * (CTXK*256);
  const int rr0 = warp;
  const int nr = (rr0 < RMAX) + (rr0 + 8 < RMAX) + (rr0 + 16 < RMAX);
  #define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )
  #define LDQ8(DST, R) { const float4 qa = *(const float4*)(qw + (size_t)QROWIDX(R)*256 + lane*8); \
    const float4 qb = *(const float4*)(qw + (size_t)QROWIDX(R)*256 + lane*8 + 4); \
    DST[0]=qa.x; DST[1]=qa.y; DST[2]=qa.z; DST[3]=qa.w; DST[4]=qb.x; DST[5]=qb.y; DST[6]=qb.z; DST[7]=qb.w; }
  float qe0[8], qe1[8], qe2[8];
  if (rr0     < RMAX) LDQ8(qe0, rr0)
  if (rr0 + 8 < RMAX) LDQ8(qe1, rr0 + 8)
  if (rr0 +16 < RMAX) LDQ8(qe2, rr0 + 16)
  float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f;
  float s0 = 0.f, s1 = 0.f, s2 = 0.f;
  float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0}, a2[8] = {0,0,0,0,0,0,0,0};
  if (nr > 0 && l0 < l1) {
    float4 rk[PF], rv[PF], rk2[PF], rv2[PF];
    const int nt = (l1 - l0 + PF - 1) / PF;
    #define LOADTILE(RK, RV, BASE) _Pragma("unroll") \
      for (int u = 0; u < PF; ++u) { const int lu = (BASE) + u; if (lu < l1) { \
        RK[u] = *(const float4*)(Kc + (size_t)lu*256 + lane*8); \
        RV[u] = *(const float4*)(Vc + (size_t)lu*256 + lane*8); } }
    LOADTILE(rk, rv, l0)
    for (int t = 0; t < nt; ++t) {
      const int base = l0 + t * PF;
      if (t + 1 < nt) LOADTILE(rk2, rv2, base + PF)
      #pragma unroll
      for (int u = 0; u < PF; ++u) {
        const int l = base + u;
        if (l >= l1) break;
        const float4 kx = rk[u], vx = rv[u];
        float kr0[8], vr0[8];
        { const __half2* hk = (const __half2*)&kx; const __half2* hv = (const __half2*)&vx;
          float2 f0 = __half22float2(hk[0]), f1 = __half22float2(hk[1]), f2 = __half22float2(hk[2]), f3 = __half22float2(hk[3]);
          kr0[0]=f0.x; kr0[1]=f0.y; kr0[2]=f1.x; kr0[3]=f1.y; kr0[4]=f2.x; kr0[5]=f2.y; kr0[6]=f3.x; kr0[7]=f3.y;
          float2 g0 = __half22float2(hv[0]), g1 = __half22float2(hv[1]), g2 = __half22float2(hv[2]), g3 = __half22float2(hv[3]);
          vr0[0]=g0.x; vr0[1]=g0.y; vr0[2]=g1.x; vr0[3]=g1.y; vr0[4]=g2.x; vr0[5]=g2.y; vr0[6]=g3.x; vr0[7]=g3.y; }
        PROC_ROW(m0, s0, a0, qe0, (rr0 < RMAX && l <= pos + (rr0 % ROWS)))
        PROC_ROW(m1, s1, a1, qe1, (rr0 + 8 < RMAX && l <= pos + ((rr0+8) % ROWS)))
        PROC_ROW(m2, s2, a2, qe2, (rr0 +16 < RMAX && l <= pos + ((rr0+16) % ROWS)))
      }
      if (t + 1 < nt) {
        #pragma unroll
        for (int u = 0; u < PF; ++u) { rk[u] = rk2[u]; rv[u] = rv2[u]; }
      }
    }
  }
  const size_t pb0 = (size_t)(g*S + s)*RMAX;
  if (rr0 < RMAX) {
    const size_t b = pb0 + rr0;
    if (lane == 0) { pm[b] = m0; ps[b] = s0; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a0[j];
  }
  if (rr0 + 8 < RMAX) {
    const size_t b = pb0 + rr0 + 8;
    if (lane == 0) { pm[b] = m1; ps[b] = s1; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a1[j];
  }
  if (rr0 +16 < RMAX) {
    const size_t b = pb0 + rr0 + 16;
    if (lane == 0) { pm[b] = m2; ps[b] = s2; }
    _Pragma("unroll") for (int j = 0; j < 8; ++j) pA[b*256 + lane*8 + j] = a2[j];
  }
}
