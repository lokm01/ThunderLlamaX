// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W4 SKV-G: smem-staged split-KV attention (K1 replacement for skv.cu K1S).
// ROOT CAUSE being fixed: K1S streams the SAME KV bytes 8x (8 warps x same
// l-range) -> per-SM dext load-return pipe saturates at ~10.8 B/clk -> 190 GB/s.
// Fix: fetch each KV byte ONCE per CTA into smem (1x DRAM, coalesced 16B,
// E4 k_stream-proven class), then:
//   G1 (VARIANT=0, spk_g1*): DISCRIMINATOR. Same staging, but the CURRENT
//       per-l QK structure (lanes split d 8-each + 5-shuffle reduce per
//       (row,l), per-l online softmax, per-l PV) reading K/V from smem.
//       Per-row FP op sequence == K1S verbatim -> output should match the
//       current trio bitwise; isolates "staging works" from "shuffle poison".
//   G2 (VARIANT=1, spk_g2*): THE REDESIGN. Lane-per-l full private 256-dot
//       QK (ZERO shuffles in the hot loop; Q from qw via 16B broadcast loads
//       -- see DEVIATION note), tile-granularity FA2 online softmax (fixed
//       butterfly 16,8,4,2,1), v-split PV from smem (p via own-warp SM_P).
//       Partials + K2S combine BYTE-FOR-BYTE UNCHANGED.
// DEVIATION from spec (documented in W2B_SKVG.md): spec stages Q into
// SM_Q (RMAX*256 fp32 = 18KB at T=3) -> 53.5KB total > the 48KB STATIC
// __shared__ limit on sm_86; the fork launch path has no dynamic-smem
// (QMD size = cubin .nv.shared section) and 2 CTAs x 53.5KB > the 100KB
// sm_86 carveout anyway. Q therefore read from global qw (fp32 values
// verbatim, same address all lanes -> one 16B L1 broadcast per load).
// Numerics contracts kept: identical TILE for T=1/T=3 builds, butterfly
// 16,8,4,2,1, PV i ascending, dot d ascending 0..255, masking predicate
// (l<l1)&&(l<=pos+t) -> T=1 and T=3 builds row-bitwise-identical.
// Variants: -DPF=1 G2 register-prefetch ring (TILE/8 float4 K + TILE/8 V,
//           compile-time indexed, global loads overlap QK+PV compute);
//           -DD2=1 G2 double-buffered SM_K/V (2x tile bytes + P <= 48KB
//           -> TILE=16), ONE __syncthreads per tile.
// LAWS honored: single __shared__ array + compile-time offsets; no
// blockDim/gridDim reads; flat indexing + guards; 16B-natural alignment for
// all smem offsets and global addresses (528 = 33*16); full-mask shuffles
// only in warp-uniform code; sequential loops; pos read once to a register.
// -DKNAME -DCTXK -DROWS (3|1) -DS -DCH -DTILE (default 32) -DVARIANT -DPF -DD2
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define RMAX (6*ROWS)
#ifndef SYNCW
  #define SYNCW 1
#endif

// ---------------- smem layout (bytes, compile-time) ----------------
// G1:  [K TILE*528][V TILE*512]
// G2:  [K TILE*528][V TILE*512][P 8*3*TILE*4]
// D2:  G2 + second [K][V] pair after P.
#define SM_K0   0
#define SM_V0   (TILE*528)
#if VARIANT == 0
  #define SM_BYTES (TILE*528 + TILE*512)
#else
  #define SM_P0   (TILE*528 + TILE*512)
  #if D2
    #define SM_K1 (SM_P0 + 8*3*TILE*4)
    #define SM_V1 (SM_K1 + TILE*528)
    #define SM_BYTES (SM_V1 + TILE*512)
  #else
    #define SM_BYTES (SM_P0 + 8*3*TILE*4)
  #endif
#endif

#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )

// ---------------- STAGE: KV tile -> smem, 1x DRAM, coalesced 16B ----------------
// chunk c = i*256 + tid; row = c>>5 (l row within tile); c16 = c&31 (16B slot).
// rows are 512B (256 halves); l >= l1 -> zero fill, NO global touch.
#define STAGE(BASEL, KO, VO) _Pragma("unroll") \
  for (int i = 0; i < TILE/8; ++i) { \
    const int c = i*256 + tid; \
    const int row = c >> 5, c16 = c & 31; \
    const int l = (BASEL) + row; \
    float4 kk = make_float4(0.f, 0.f, 0.f, 0.f), vv = make_float4(0.f, 0.f, 0.f, 0.f); \
    if (l < l1) { \
      kk = *(const float4*)(Kc + (size_t)l*256 + c16*8); \
      vv = *(const float4*)(Vc + (size_t)l*256 + c16*8); \
    } \
    *(float4*)(SM + (KO) + row*528 + c16*16) = kk; \
    *(float4*)(SM + (VO) + row*512 + c16*16) = vv; \
  }

#define BUTTERFLY_MAX(V) _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) (V) = fmaxf((V), __shfl_xor_sync(FULL, (V), o));
#define BUTTERFLY_SUM(V) _Pragma("unroll") \
  for (int o = 16; o > 0; o >>= 1) (V) += __shfl_xor_sync(FULL, (V), o);

// G2 per-owned-row QK + tile-granularity softmax + SM_P (NO shuffles in the
// dot; Q 16B broadcast from global). FP order: dot d ascending 0..255
// (chunk c 0..31, j 0..7); tile max via fixed butterfly 16,8,4,2,1;
// p = valid ? expf(sc - mnew) : 0.
#define G2_QKROW(RI, R, Mv, Sv, A8) { \
  const int t_ = (R) % ROWS; \
  const bool vld = (lpl < l1) && (lpl <= pos + t_); \
  float sc = 0.f; \
  const float* qb = qw + (size_t)QROWIDX(R)*256; \
  _Pragma("unroll") \
  for (int c = 0; c < 32; ++c) { \
    const float4 kf4 = *(const float4*)(SM + kcur + lane*528 + c*16); \
    const __half2* hk = (const __half2*)&kf4; \
    const float2 k0 = __half22float2(hk[0]), k1 = __half22float2(hk[1]), k2 = __half22float2(hk[2]), k3 = __half22float2(hk[3]); \
    const float4 qa = *(const float4*)(qb + c*8); \
    const float4 qcx = *(const float4*)(qb + c*8 + 4); \
    const float qq[8] = {qa.x, qa.y, qa.z, qa.w, qcx.x, qcx.y, qcx.z, qcx.w}; \
    const float kk[8] = {k0.x, k0.y, k1.x, k1.y, k2.x, k2.y, k3.x, k3.y}; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) sc += qq[j] * kk[j]; \
  } \
  float scm = vld ? sc : -1e30f; \
  BUTTERFLY_MAX(scm) \
  const float mn = fmaxf((Mv), scm); \
  const float cor = expf((Mv) - mn); \
  (Sv) *= cor; \
  _Pragma("unroll") for (int j = 0; j < 8; ++j) (A8)[j] *= cor; \
  (Mv) = mn; \
  const float p = vld ? expf(sc - mn) : 0.f; \
  float ps = p; \
  BUTTERFLY_SUM(ps) \
  (Sv) += ps; \
  ((float*)(SM + SM_P0))[(warp*3 + (RI))*TILE + lane] = p; \
}

// G2 PV for one owned row: i ascending, p own-warp broadcast, v4 lane-private.
#define G2_PVROW(RI, A8) _Pragma("unroll") \
  for (int i2 = 0; i2 < TILE; ++i2) { \
    const float p = ((const float*)(SM + SM_P0))[(warp*3 + (RI))*TILE + i2]; \
    const float4 vf4 = *(const float4*)(SM + vcur + i2*512 + lane*16); \
    const __half2* hv = (const __half2*)&vf4; \
    const float2 g0 = __half22float2(hv[0]), g1 = __half22float2(hv[1]), g2 = __half22float2(hv[2]), g3 = __half22float2(hv[3]); \
    const float vv[8] = {g0.x, g0.y, g1.x, g1.y, g2.x, g2.y, g3.x, g3.y}; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) (A8)[j] += p * vv[j]; \
  }

// G1 per-(row,l) QK structure == K1S verbatim (8-dim dot + 5-shfl reduce +
// per-l online softmax + per-l PV), K/V values from smem.
#define G1_PROCROW(Mv, Sv, A8, Q8, K8, V8, ACT) { \
  float sc = 0.f; \
  _Pragma("unroll") for (int j = 0; j < 8; ++j) sc += (Q8)[j] * (K8)[j]; \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(FULL, sc, o); \
  if (ACT) { \
    if (sc > (Mv)) { const float cor = expf((Mv) - sc); (Sv) *= cor; _Pragma("unroll") for (int j=0;j<8;++j) (A8)[j] *= cor; (Mv) = sc; } \
    const float p = expf(sc - (Mv)); \
    (Sv) += p; \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) (A8)[j] += p * (V8)[j]; \
  } \
}

extern "C" __global__ void __launch_bounds__(256) KNAME(
    const __half* __restrict__ kv, const float* __restrict__ qw, const int* __restrict__ pos_slot,
    float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
{
  const int g = blockIdx.x / S;
  const int s = blockIdx.x % S;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int pos = pos_slot[0];                       // read ONCE
  const int l0 = s * CH;
  const int l1 = (l0 + CH) < (pos + ROWS) ? (l0 + CH) : (pos + ROWS);
  const __half* Kc = kv + (size_t)g * (CTXK*256);
  const __half* Vc = kv + (size_t)(4 + g) * (CTXK*256);
  __shared__ __align__(16) char SM[SM_BYTES];

  const int rr0 = warp;
  float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f;
  float s0 = 0.f, s1 = 0.f, s2 = 0.f;
  float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0}, a2[8] = {0,0,0,0,0,0,0,0};

#if VARIANT == 0
  // ============================== G1 (discriminator) ==============================
  // Q slices in registers (per-lane 8 dims), verbatim K1S LDQ8.
  float qe0[8], qe1[8], qe2[8];
  #define LDQ8(DST, R) { const float4 qa = *(const float4*)(qw + (size_t)QROWIDX(R)*256 + lane*8); \
    const float4 qb = *(const float4*)(qw + (size_t)QROWIDX(R)*256 + lane*8 + 4); \
    DST[0]=qa.x; DST[1]=qa.y; DST[2]=qa.z; DST[3]=qa.w; DST[4]=qb.x; DST[5]=qb.y; DST[6]=qb.z; DST[7]=qb.w; }
  if (rr0     < RMAX) LDQ8(qe0, rr0)
  if (rr0 + 8 < RMAX) LDQ8(qe1, rr0 + 8)
  if (rr0 +16 < RMAX) LDQ8(qe2, rr0 + 16)
  if (l0 < l1) {                                     // CTA-uniform (barriers inside)
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      __syncthreads();                               // prev tile readers done
      STAGE(tile, SM_K0, SM_V0)
      __syncthreads();                               // stage visible
      #pragma unroll 4
      for (int i2 = 0; i2 < TILE; ++i2) {            // per-l, l ascending
        const int l = tile + i2;
        const bool livel = l < l1;
        float kr0[8], vr0[8];
        { const float4 kf4 = *(const float4*)(SM + SM_K0 + i2*528 + lane*16);
          const float4 vf4 = *(const float4*)(SM + SM_V0 + i2*512 + lane*16);
          const __half2* hk = (const __half2*)&kf4; const __half2* hv = (const __half2*)&vf4;
          float2 k0 = __half22float2(hk[0]), k1 = __half22float2(hk[1]), k2 = __half22float2(hk[2]), k3 = __half22float2(hk[3]);
          float2 g0 = __half22float2(hv[0]), g1 = __half22float2(hv[1]), g2 = __half22float2(hv[2]), g3 = __half22float2(hv[3]);
          kr0[0]=k0.x; kr0[1]=k0.y; kr0[2]=k1.x; kr0[3]=k1.y; kr0[4]=k2.x; kr0[5]=k2.y; kr0[6]=k3.x; kr0[7]=k3.y;
          vr0[0]=g0.x; vr0[1]=g0.y; vr0[2]=g1.x; vr0[3]=g1.y; vr0[4]=g2.x; vr0[5]=g2.y; vr0[6]=g3.x; vr0[7]=g3.y; }
        if (rr0 < RMAX && livel)
          G1_PROCROW(m0, s0, a0, qe0, kr0, vr0, (l <= pos + (rr0 % ROWS)))
        if (rr0 + 8 < RMAX && livel)
          G1_PROCROW(m1, s1, a1, qe1, kr0, vr0, (l <= pos + ((rr0+8) % ROWS)))
        if (rr0 +16 < RMAX && livel)
          G1_PROCROW(m2, s2, a2, qe2, kr0, vr0, (l <= pos + ((rr0+16) % ROWS)))
      }
    }
  }
#else
  // ============================== G2 (the redesign) ==============================
  int kcur = SM_K0, vcur = SM_V0;                    // current compute buffer
  #if PF
    float4 pfk[TILE/8], pfv[TILE/8];                 // register prefetch ring (compile-time indexed)
    #define PFETCH(BASEL) _Pragma("unroll") \
      for (int i = 0; i < TILE/8; ++i) { \
        const int c = i*256 + tid; \
        const int row = c >> 5, c16 = c & 31; \
        const int l = (BASEL) + row; \
        if (l < l1) { \
          pfk[i] = *(const float4*)(Kc + (size_t)l*256 + c16*8); \
          pfv[i] = *(const float4*)(Vc + (size_t)l*256 + c16*8); \
        } else { pfk[i] = make_float4(0.f,0.f,0.f,0.f); pfv[i] = make_float4(0.f,0.f,0.f,0.f); } \
      }
    #define PSTORE _Pragma("unroll") \
      for (int i = 0; i < TILE/8; ++i) { \
        const int c = i*256 + tid; \
        const int row = c >> 5, c16 = c & 31; \
        *(float4*)(SM + SM_K0 + row*528 + c16*16) = pfk[i]; \
        *(float4*)(SM + SM_V0 + row*512 + c16*16) = pfv[i]; \
      }
  #endif
  if (l0 < l1) {                                     // CTA-uniform (barriers inside)
    const int nt = (l1 - l0 + TILE - 1) / TILE;
   #if PF
    __syncthreads();
    STAGE(l0, SM_K0, SM_V0)
    __syncthreads();
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      const int lpl = tile + lane;
      if (t + 1 < nt) PFETCH(tile + TILE)            // next-tile loads in flight during compute
   #elif D2
    __syncthreads();
    STAGE(l0, SM_K0, SM_V0)
    __syncthreads();
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      const int lpl = tile + lane;
      kcur = (t & 1) ? SM_K1 : SM_K0;                // buffer staged at iteration t
      vcur = (t & 1) ? SM_V1 : SM_V0;
      if (t + 1 < nt) {                              // stage NEXT tile into the other buffer
        if (t & 1) STAGE(tile + TILE, SM_K0, SM_V0) else STAGE(tile + TILE, SM_K1, SM_V1)
      }
   #else
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      const int lpl = tile + lane;
      __syncthreads();                               // prev tile readers done
      STAGE(tile, SM_K0, SM_V0)
      __syncthreads();                               // stage visible
   #endif
      // ---- QK + tile-granularity online softmax + SM_P (warp-uniform guards) ----
      if (rr0 < RMAX)         G2_QKROW(0, rr0,      m0, s0, a0)
      if (rr0 + 8 < RMAX)     G2_QKROW(1, rr0 + 8,  m1, s1, a1)
      if (rr0 + 16 < RMAX)    G2_QKROW(2, rr0 + 16, m2, s2, a2)
      #if SYNCW
        __syncwarp();                                 // own-warp SM_P visible before PV
      #else
        __syncthreads();                              // SYNCW=0: __syncwarp faults this dext (Multiple Warp Errors, all SMs)
      #endif
      // ---- PV from smem ----
      if (rr0 < RMAX)         G2_PVROW(0, a0)
      if (rr0 + 8 < RMAX)     G2_PVROW(1, a1)
      if (rr0 + 16 < RMAX)    G2_PVROW(2, a2)
   #if PF
      __syncthreads();                                // all readers done before overwrite
      if (t + 1 < nt) PSTORE
      __syncthreads();                                // staged tile visible
   #elif D2
      __syncthreads();                                // readers of cur done + staged nxt visible
   #endif
    }
  }
#endif

  // ---- partial writes: IDENTICAL to K1S (empty split -> identity partials) ----
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
