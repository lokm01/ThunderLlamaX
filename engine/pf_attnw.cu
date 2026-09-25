// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P17: pfa-W — the wide-M attention (ROWS=32/64 tokens per window; the 400 lever).
// VERBATIM port of the PROVEN pfa32c PFA_T32 skeleton (pf_attn32c.cu) with the
// ROWS/HRP/NW parameterization that was hardcoded (P15 Stage-0 law: grid-doubling
// reads OOB because ROWS was baked). Per-row math + online-softmax TILE grouping
// + S-split/combine structure UNCHANGED -> per-row outputs bit-identical to the
// shipped t32 (rows beyond the t32 window's 16 are masked by the SAME
// ka <= pos + t_ causal law the t32 applies to its own tail rows).
// Changes vs t32 (all data-path, zero arithmetic):
//   1. ROWS/HRP/NW are -D params (RMAX = HRP*ROWS rows/CTA; NHP = 6/HRP).
//   2. QK phase LOOPS over QK_TILES (warp stride NW) — at RMAX>32 there are
//      more than NW 16x8 out-tiles; each tile still computed exactly once with
//      the same mma k-order (independent outputs, order across warps moot).
//   3. PV db mapping generalized: NW<=16: db=2*warp+(ti&1); NW=32: db=warp.
//   4. corv lives in the K plane (dead after QK(t) of the same tile) — saves
//      RP*4 B so ROWS=64/RMAX=128 lands EXACTLY at the 48KB static smem cap.
//   5. 1024-thread staging variant (NTHR==1024): one int2 per thread.
// Combine (PFC_T32 block): t < ROWS loop, same epilogue/op-order as pfc16t.
// Laws kept: single 16B-aligned smem, compile-time offsets, no blockDim/gridDim
// reads, full masks, sequential tile loop, no cp.async, no runtime-indexed
// locals, l1-guarded global reads, per-kernel cubins + symbol check.
// -DCTXK -DS -DKNAME -DROWS -DHRP -DNW -DMINB(=1) [-DPFC_T32=1 for the combine]
#include <cuda_fp16.h>
#define TILE 32
#ifndef NW
  #define NW 16
#endif
#ifndef HRP
  #define HRP 2
#endif
#ifndef ROWS
  #define ROWS 16
#endif
#define NTHR (NW*32)
#define NHP (6/HRP)
#define NCTA (4*S*NHP)               // hardcoded grid size (the no-gridDim law)
#define CHW (256/NW)
#define CH (((CTXK) + S - 1) / S)    // CEIL split (100352 = 2^11*7^2)
#define FULL 0xffffffffu
#define RMAX (HRP*ROWS)
#define RP RMAX
#define MB (RP/16)
#define QK_TILES (MB*(TILE/8))
#define PV_TILES (MB*32)
#define PV_TPW (PV_TILES/NW)
#define ROWR (RP/NW)

#define SM_K0    0                       // K f16 [TILE][512B] swizzled
#define SM_V0    (TILE*512)              // V f16 [TILE][512B]
#define SM_SC0   (SM_V0 + TILE*512)      // SCP f32 [RP][TILE] -> Pm f16 in-place
#define SM_COR0  SM_K0                   // corv [RP] f32 in the K plane (dead after QK(t))
#define SM_BYTES (SM_SC0 + RP*TILE*4)
#define KSZ(R, C)  (((R) * 512) + (((C) ^ ((R) & 7)) * 16))
#define KEL(R, C, I) (*(const __half*)(SM + SM_K0 + KSZ(R, C) + 2*(I)))
#define VSZ(R, C)  (((R) * 512) + ((C) * 16))
#define VEL(R, C, I) (*(const __half*)(SM + SM_V0 + VSZ(R, C) + 2*(I)))
#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + hp*HRP + (R)/ROWS )

#if NW > 16
  #define PV_MB(TI) (TI)
  #define PV_DB(TI) (warp)
#else
  #define PV_MB(TI) ((TI) >> 1)
  #define PV_DB(TI) (2*warp + ((TI) & 1))
#endif

union F4H8 { float4 f; __half2 h[4]; };
#define DQH2(R, S2) __hmul2(__hsub2((R), (__half2)__float2half2_rn(1152.f)), (S2))
__device__ __forceinline__ float4 dq8(const int2 iv, const __half s) {
  F4H8 o;
  const __half2 s2 = __half2half2(s);
  const unsigned x = (unsigned)iv.x, y = (unsigned)iv.y;
  const unsigned p0 = __byte_perm(x, 0x64646464u, 0x5140), p1 = __byte_perm(x, 0x64646464u, 0x7362);
  const unsigned p2 = __byte_perm(y, 0x64646464u, 0x5140), p3 = __byte_perm(y, 0x64646464u, 0x7362);
  o.h[0] = DQH2(*(const __half2*)&p0, s2);
  o.h[1] = DQH2(*(const __half2*)&p1, s2);
  o.h[2] = DQH2(*(const __half2*)&p2, s2);
  o.h[3] = DQH2(*(const __half2*)&p3, s2);
  return o.f;
}
__device__ __forceinline__ unsigned h2u(const __half2 h) { return *reinterpret_cast<const unsigned*>(&h); }
__device__ __forceinline__ void hmma16816(float4 &c, const unsigned a0, const unsigned a1,
                                          const unsigned a2, const unsigned a3,
                                          const unsigned b0, const unsigned b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#ifndef PFC_T32
#if MINB > 0
extern "C" __global__ void __launch_bounds__(NTHR, MINB) KNAME(
#else
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
#endif
    const unsigned char* __restrict__ kv, const __half* __restrict__ sc, const __half* __restrict__ qw16,
    const int* __restrict__ pos_slot, float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
{
  const int hp = blockIdx.x % NHP;
  const int s  = (blockIdx.x / NHP) % S;
  const int g  = blockIdx.x / (NHP * S);
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int pos = pos_slot[0];
  const int l0 = s * CH;
  const int l1 = (l0 + CH) < (pos + ROWS) ? (l0 + CH) : (pos + ROWS);
  const unsigned char* Kc8 = kv + (size_t)g * (CTXK*256);
  const unsigned char* Vc8 = kv + (size_t)(4 + g) * (CTXK*256);
  const __half* Ksc = sc + (size_t)g * (CTXK*8);
  const __half* Vsc = sc + (size_t)(4 + g) * (CTXK*8);
  __shared__ __align__(16) char SM[SM_BYTES];

  float* corv = (float*)(SM + SM_COR0);
  __half* Pm = (__half*)(SM + SM_SC0);     // Pm f16 lives in the low half of the plane
  float* SCP = (float*)(SM + SM_SC0);
  float ms_r[ROWR], ss_r[ROWR];
  _Pragma("unroll") for (int ri = 0; ri < ROWR; ++ri) { ms_r[ri] = -1e30f; ss_r[ri] = 0.f; }
  float4 acc[PV_TPW];
  _Pragma("unroll") for (int ti = 0; ti < PV_TPW; ++ti) acc[ti] = make_float4(0.f, 0.f, 0.f, 0.f);

  // ---- linear staging ring: K(t) + V(t) ----
#if NTHR == 1024
  const int st_row = tid >> 5, st_c16 = tid & 31;      // 32 rows x 32 x8B slots (1x int2)
  int2 kreg0, vreg0; __half ksc0, vsc0;
  #define STG_LOAD(BASE) do { \
    const int lr0 = (BASE) + st_row; \
    kreg0 = make_int2(0, 0); ksc0 = __float2half(1.f); \
    vreg0 = make_int2(0, 0); vsc0 = __float2half(1.f); \
    if (lr0 < l1) { \
      kreg0 = *(const int2*)(Kc8 + (size_t)lr0*256 + st_c16*8);      ksc0 = Ksc[(size_t)lr0*8 + (st_c16 >> 2)]; \
      vreg0 = *(const int2*)(Vc8 + (size_t)lr0*256 + st_c16*8);      vsc0 = Vsc[(size_t)lr0*8 + (st_c16 >> 2)]; \
    } \
  } while (0)
  #define STG_COMMIT() do { \
    const int lr0c = tile + st_row; \
    if (lr0c < l1) { \
      *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16)) = dq8(kreg0, ksc0); \
      *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16)) = dq8(vreg0, vsc0); \
    } else { \
      *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16)) = make_float4(0.f,0.f,0.f,0.f); \
      *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16)) = make_float4(0.f,0.f,0.f,0.f); \
    } \
  } while (0)
#else
  const int st_row = tid >> 4, st_c16 = (tid & 15) << 1; // 32 rows x 32 c16 slots (2x int2)
  int2 kreg0, kreg1, vreg0, vreg1; __half ksc0, ksc1, vsc0, vsc1;
  // P18 CACHE-POLICY: the K/V stream is read-ONCE -> __ldcs (evict-first) keeps
  // the Q panels L1-resident across tile iterations (the A(Q) loads re-touch the
  // same 64KB every iteration; with default policy the stream evicts them -> the
  // DRAM-latency wall). Hints only -> values identical -> BIT-IDENTICAL.
  #define STG_LOAD(BASE) do { \
    const int lr0 = (BASE) + st_row; \
    kreg0 = make_int2(0, 0); kreg1 = make_int2(0, 0); ksc0 = __float2half(1.f); ksc1 = ksc0; \
    vreg0 = make_int2(0, 0); vreg1 = make_int2(0, 0); vsc0 = __float2half(1.f); vsc1 = vsc0; \
    if (lr0 < l1) { \
      kreg0 = __ldcs((const int2*)(Kc8 + (size_t)lr0*256 + st_c16*8));      ksc0 = __ldcs(&Ksc[(size_t)lr0*8 + (st_c16 >> 2)]); \
      vreg0 = __ldcs((const int2*)(Vc8 + (size_t)lr0*256 + st_c16*8));      vsc0 = __ldcs(&Vsc[(size_t)lr0*8 + (st_c16 >> 2)]); \
    } \
    if (lr0 < l1) { \
      kreg1 = __ldcs((const int2*)(Kc8 + (size_t)lr0*256 + st_c16*8 + 8));  ksc1 = __ldcs(&Ksc[(size_t)lr0*8 + ((st_c16+1) >> 2)]); \
      vreg1 = __ldcs((const int2*)(Vc8 + (size_t)lr0*256 + st_c16*8 + 8));  vsc1 = __ldcs(&Vsc[(size_t)lr0*8 + ((st_c16+1) >> 2)]); \
    } \
  } while (0)
  #define STG_COMMIT() do { \
    const int lr0c = tile + st_row; \
    if (lr0c < l1) { \
      *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16))       = dq8(kreg0, ksc0); \
      *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16))       = dq8(vreg0, vsc0); \
    } else { \
      *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16))       = make_float4(0.f,0.f,0.f,0.f); \
      *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16))       = make_float4(0.f,0.f,0.f,0.f); \
    } \
    if (lr0c < l1) { \
      *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16 + 1)) = dq8(kreg1, ksc1); \
      *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16 + 1)) = dq8(vreg1, vsc1); \
    } else { \
      *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16 + 1)) = make_float4(0.f,0.f,0.f,0.f); \
      *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16 + 1)) = make_float4(0.f,0.f,0.f,0.f); \
    } \
  } while (0)
#endif
  #define STG_ZERO() do { \
    kreg0 = make_int2(0, 0); ksc0 = __float2half(1.f); \
    vreg0 = make_int2(0, 0); vsc0 = __float2half(1.f); \
  } while (0)

  if (l0 < l1) {
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    STG_LOAD(l0);
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      __syncthreads();   // (1) PV(t-1) done -> V region safe to rewrite
      STG_COMMIT();      // stage K(t) + V(t)
      __syncthreads();   // (2) K,V visible
      // reload K(t+1)+V(t+1) raw (K region dead after QK(t); V dead after PV(t))
      if (t + 1 < nt) STG_LOAD(tile + TILE); else STG_ZERO();

      // ---- QK(t): P18 A-SHARE (permutation form) — tile mapping permuted to
      // mb = wt %% MB, nb = wt / MB so a warp's two wt tiles (wt, wt+NW; needs
      // MB | NW — true for every shipped shape) own the SAME mb row-block: the
      // second pass's A(Q) loads hit L1 (same warp, same addresses, back-to-back)
      // and the address regs are shared. THE P18 GROWTH-POOL FIX: the old mapping
      // re-read Q from DRAM every tile iteration (241 x 64KB/CTA = 2.4GB/launch
      // at 100k — the KV stream thrashes L2). Zero new live registers (the c[2]
      // explicit-share variant spills at the 128-reg/512-thread cap = the w64q
      // nondet class — banked). Pure work permutation: each (mb,nb) tile computed
      // exactly once, same mma k-order, fresh c -> BIT-IDENTICAL.
      for (int wt = warp; wt < QK_TILES; wt += NW) {
        const int nb8 = TILE >> 3;
        const int mb = wt % MB, nb = wt / MB;   // P18: was wt / nb8, wt % nb8
        const int r0 = mb*16, n0 = nb*8;
        const int m1 = r0 + (lane >> 2);
        const __half* QB  = qw16 + (size_t)QROWIDX(m1)*256;
        const __half* QB2 = qw16 + (size_t)QROWIDX(m1+8)*256;
        float4 c = make_float4(0.f, 0.f, 0.f, 0.f);
        _Pragma("unroll")
        for (int kb = 0; kb < 16; ++kb) {
          const int kbase = kb*16;
          const unsigned a0 = __ldg((const unsigned*)(QB  + kbase + (lane & 3)*2));
          const unsigned a1 = __ldg((const unsigned*)(QB2 + kbase + (lane & 3)*2));
          const unsigned a2 = __ldg((const unsigned*)(QB  + kbase + (lane & 3)*2 + 8));
          const unsigned a3 = __ldg((const unsigned*)(QB2 + kbase + (lane & 3)*2 + 8));
          const int nn = n0 + (lane >> 2);
          const int chc = kbase >> 3, dpair = (lane & 3)*2;
          const __half k00 = KEL(nn, chc,     dpair);
          const __half k01 = KEL(nn, chc,     dpair + 1);
          const __half k10 = KEL(nn, chc + 1, dpair);
          const __half k11 = KEL(nn, chc + 1, dpair + 1);
          const unsigned b0 = h2u(__halves2half2(k00, k01));
          const unsigned b1 = h2u(__halves2half2(k10, k11));
          hmma16816(c, a0, a1, a2, a3, b0, b1);
        }
        const int r1 = r0 + (lane >> 2), n1 = n0 + (lane & 3)*2;
        SCP[(size_t)r1*TILE + n1]       = c.x;
        SCP[(size_t)r1*TILE + n1 + 1]   = c.y;
        SCP[(size_t)(r1+8)*TILE + n1]   = c.z;
        SCP[(size_t)(r1+8)*TILE + n1 + 1] = c.w;
      }
      __syncthreads();   // (3) SCP(t) visible

      // ---- row owners: single-pass TILE-wide online softmax; Pm f16 in-place ----
      _Pragma("unroll")
      for (int ri = 0; ri < ROWR; ++ri) {
        const int r = warp + ri*NW;
        if (r < RP) {
          const int t_ = r % ROWS;
#if TILE > 32
          const int ka0 = tile + lane, ka1 = tile + lane + 32;
          float scv0 = -1e30f, scv1 = -1e30f;
          if (ka0 < l1 && ka0 <= pos + t_) scv0 = SCP[(size_t)r*TILE + lane];
          if (ka1 < l1 && ka1 <= pos + t_) scv1 = SCP[(size_t)r*TILE + lane + 32];
          float scm = fmaxf(scv0, scv1);
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) scm = fmaxf(scm, __shfl_xor_sync(FULL, scm, o));
          const float mold = ms_r[ri];
          const float mn = fmaxf(mold, scm);
          const float cor = expf(mold - mn);
          const float p0 = (scv0 > -1e29f) ? expf(scv0 - mn) : 0.f;
          const float p1 = (scv1 > -1e29f) ? expf(scv1 - mn) : 0.f;
          Pm[(size_t)r*TILE + lane] = (__half)p0;
          Pm[(size_t)r*TILE + lane + 32] = (__half)p1;
          float ps_ = p0 + p1;
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) ps_ += __shfl_xor_sync(FULL, ps_, o);
          ss_r[ri] = ss_r[ri]*cor + ps_;
          ms_r[ri] = mn;
          if (lane == 0) corv[r] = cor;
#else
          const int ka = tile + lane;
          float scv = -1e30f;
          if (ka < l1 && ka <= pos + t_) scv = SCP[(size_t)r*TILE + lane];
          float scm = scv;
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) scm = fmaxf(scm, __shfl_xor_sync(FULL, scm, o));
          const float mold = ms_r[ri];
          const float mn = fmaxf(mold, scm);
          const float cor = expf(mold - mn);
          const float p = (scv > -1e29f) ? expf(scv - mn) : 0.f;
          Pm[(size_t)r*TILE + lane] = (__half)p;
          float ps_ = p;
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) ps_ += __shfl_xor_sync(FULL, ps_, o);
          ss_r[ri] = ss_r[ri]*cor + ps_;
          ms_r[ri] = mn;
          if (lane == 0) corv[r] = cor;
#endif
        }
      }
      __syncthreads();   // (4) Pm + corv visible

      // ---- PV(t): warp-private channels; TILE/16 k-steps of m16n8k16 ----
      _Pragma("unroll")
      for (int ti = 0; ti < PV_TPW; ++ti) {
        const int mb = PV_MB(ti), db = PV_DB(ti);
        const int r0 = mb*16, d0 = db*8;
        const float cor0 = corv[r0 + (lane >> 2)];
        const float cor1 = corv[r0 + (lane >> 2) + 8];
        acc[ti].x *= cor0; acc[ti].y *= cor0;
        acc[ti].z *= cor1; acc[ti].w *= cor1;
        _Pragma("unroll")
        for (int kh = 0; kh < (TILE >> 4); ++kh) {
          const int m1 = r0 + (lane >> 2);
          const __half* PB = Pm;
          const unsigned a0 = *(const unsigned*)(PB + (size_t)m1*TILE + kh*16 + (lane & 3)*2);
          const unsigned a1 = *(const unsigned*)(PB + (size_t)(m1+8)*TILE + kh*16 + (lane & 3)*2);
          const unsigned a2 = *(const unsigned*)(PB + (size_t)m1*TILE + kh*16 + (lane & 3)*2 + 8);
          const unsigned a3 = *(const unsigned*)(PB + (size_t)(m1+8)*TILE + kh*16 + (lane & 3)*2 + 8);
          const int kr0 = kh*16 + (lane & 3)*2;
          const int bd = d0 + (lane >> 2);
          const __half v00 = VEL(kr0,     bd >> 3, bd & 7);
          const __half v01 = VEL(kr0 + 1, bd >> 3, bd & 7);
          const __half v10 = VEL(kr0 + 8, bd >> 3, bd & 7);
          const __half v11 = VEL(kr0 + 9, bd >> 3, bd & 7);
          const unsigned b0 = h2u(__halves2half2(v00, v01));
          const unsigned b1 = h2u(__halves2half2(v10, v11));
          hmma16816(acc[ti], a0, a1, a2, a3, b0, b1);
        }
      }
    }
  }

  // ---- final partial writes ----
  const size_t pb0 = (size_t)blockIdx.x * RP;
  _Pragma("unroll")
  for (int ri = 0; ri < ROWR; ++ri) {
    const int r = warp + ri*NW;
    if (r < RP && lane == 0) {
      pm[pb0 + r] = ms_r[ri];
      ps[pb0 + r] = ss_r[ri];
    }
  }
  _Pragma("unroll")
  for (int ti = 0; ti < PV_TPW; ++ti) {
    const int mb = PV_MB(ti), db = PV_DB(ti);
    const int r0 = mb*16, d0 = db*8;
    const int m1 = r0 + (lane >> 2), dd = d0 + (lane & 3)*2;
    pA[(pb0 + m1)*256 + dd] = acc[ti].x;
    pA[(pb0 + m1)*256 + dd + 1] = acc[ti].y;
    pA[(pb0 + m1 + 8)*256 + dd] = acc[ti].z;
    pA[(pb0 + m1 + 8)*256 + dd + 1] = acc[ti].w;
  }
}
#endif  // !PFC_T32

#ifdef PFC_T32
// ---- K2 combine for the wide layouts: partial slot
// (g*(NHP*S) + s*NHP + hp)*RMAX + (hli*ROWS + t); fixed-order combine +
// sigmoid gate, same math/op-order as the shipped pfc16t epilogue.
// grid 24 heads, 256 thr (one per channel).
extern "C" __global__ void __launch_bounds__(256) KNAME(
    const float* __restrict__ pm, const float* __restrict__ ps, const float* __restrict__ pA,
    const __half* __restrict__ qrow16, __half* __restrict__ ao16)
{
  const int h = blockIdx.x;
  const int g = h / 6, hl = h % 6;
  const int hp = hl / HRP, hli = hl % HRP;
  const int d = threadIdx.x;
  for (int t = 0; t < ROWS; ++t) {
    const int r = hli*ROWS + t;
    float M = -1e30f;
    for (int s2 = 0; s2 < S; ++s2) {
      const size_t pb = (size_t)(g*(NHP*S) + s2*NHP + hp)*RMAX + r;
      M = fmaxf(M, pm[pb]);
    }
    float out = 0.f, Ssum = 0.f;
    for (int s2 = 0; s2 < S; ++s2) {
      const size_t pb = (size_t)(g*(NHP*S) + s2*NHP + hp)*RMAX + r;
      const float ex = expf(pm[pb] - M);
      Ssum += ps[pb] * ex;
      out += pA[pb*256 + d] * ex;
    }
    out /= Ssum;
    const float gf = __half2float(qrow16[(size_t)t*12288 + h*512 + 256 + d]);
    const float sg = 1.0f / (1.0f + expf(-gf));
    ao16[(size_t)t*6144 + h*256 + d] = __float2half(out * sg);
  }
}
#endif  // PFC_T32
