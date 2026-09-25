// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P2 pATTN-M: causal M=16 x L chunk attention on mma.m16n8k16 (W2G/W2H
// probe-verified fragment map; fp32 acc; 0.0625 already folded into qw16 by
// pfk_pre16 — exact-in-fp16 fold). TWO kernels:
//   pfa16 (K1): grid (4 kv-groups x S splits), 1024thr/32warps. All 6 q-heads
//     x 16 rows = 96 q-rows of the group per CTA (RMAX=96, RP=96: no padding).
//     Per L-tile of 32 keys: stage K -> QK mma (24 out-tiles = 6 m-blocks x 4
//     key-blocks; warp<TILE count guard; single SCP plane) -> row owners
//     (each warp owns rows w, w+32, w+64; online softmax in registers, fixed
//     xor butterflies — same op order as the decode kernel per row) -> stage V
//     OVER the dead K region (smem diet: K+V share one 16KB region) -> PV mma
//     (192 out-tiles = 6 m-blocks x 32 dim-blocks; 6 tiles/warp = 24 acc regs).
//     Causal bound l <= pos + (r & 15). Empty split -> identity partials.
//   pfc16 (K2): grid 24 heads, 256thr. Fixed-order combine over S partials +
//     sigmoid gate from qrow16 gate cols -> ao16[16][6144] fp16.
// SMEM: KV 16384 | SCP [96][32] f32 12288 | Pm [96][32] f16 6144 | corv 384
//   = 35200B <= 36864 (the in-graph law). All offsets 16B-aligned.
// LAWS: single 16B-aligned smem array, compile-time offsets, no blockDim/gridDim
//   reads, full masks, sequential tile loop, no cp.async, per-kernel cubins,
//   name carries the warp token (nw32 -> 1024 threads).
// -DCTXK -DS (splits) -DTILE(32) -DNW(32) -DKNAME; KSEL selects the kernel.
#include <cuda_fp16.h>
#ifndef TILE
  #define TILE 32
#endif
#ifndef NW
  #define NW 32
#endif
#define NTHR (NW*32)
#define CH (CTXK / S)
#define FULL 0xffffffffu
#define ROWS 16
#define RMAX (6*ROWS)          // 96
#define RP RMAX                // 96 (multiple of 16: no padding rows)
#define MB (RP / 16)           // 6
#define QK_TILES (MB * 4)      // 24
#define QK_WPT 1
#define QK_KS 16
#define PV_TILES (MB * 32)     // 192
#define PV_TPW (PV_TILES / NW) // 6
#define ROWR (RP / NW)         // 3 rows per warp

#define SM_KV0   0
#define SM_SC0   (TILE*512)
#define SM_P0    (SM_SC0 + RMAX*TILE*4)
#define SM_MS0   (SM_P0 + RP*TILE*2)
#define SM_BYTES (SM_MS0 + RP*4)
#define KSZ(R, C)  (((R) * 512) + (((C) ^ ((R) & 7)) * 16))
#define VSZ(R, C)  (((R) * 512) + ((C) * 16))
#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )
#define KEL(R, C, I) (*(const __half*)(SM + SM_KV0 + KSZ(R, C) + 2*(I)))
#define VEL(R, C, I) (*(const __half*)(SM + SM_KV0 + VSZ(R, C) + 2*(I)))

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

#if KSEL == 1
// ============================== K1: pfa16 ==============================
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ kv, const __half* __restrict__ sc, const __half* __restrict__ qw16,
    const int* __restrict__ pos_slot, float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
{
  const int g = blockIdx.x / S;
  const int s = blockIdx.x % S;
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

  float* corv = (float*)(SM + SM_MS0);          // [RP] per-row correction
  __half* Pm = (__half*)(SM + SM_P0);           // [RP][TILE]
  float* SCP = (float*)(SM + SM_SC0);           // [RMAX][TILE]
  float ms_r[ROWR], ss_r[ROWR];                 // row-owner online-softmax state
  _Pragma("unroll") for (int ri = 0; ri < ROWR; ++ri) { ms_r[ri] = -1e30f; ss_r[ri] = 0.f; }

  // PV accumulators (this warp's 6 tiles), fp32 c-frags
  float4 acc[PV_TPW];
  _Pragma("unroll") for (int ti = 0; ti < PV_TPW; ++ti) acc[ti] = make_float4(0.f, 0.f, 0.f, 0.f);

  if (l0 < l1) {
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    // ---- P5 DBUF tile ring: K(t+1)/V(t) raw words prefetched to registers
    // during the PV/QK phases; commit at the next boundary. 4 syncs/tile (was
    // 6); no arithmetic order change -> BIT-IDENTICAL outputs.
    const int st_row = tid >> 5, st_c16 = tid & 31;   // this thread's staging slot
    int2 kreg; __half ksc_reg; int2 vreg; __half vsc_reg;
    { // prologue: issue K(tile 0) loads
      const int l = l0 + st_row;
      kreg = make_int2(0, 0); ksc_reg = __float2half(1.f);
      if (l < l1) {
        kreg = *(const int2*)(Kc8 + (size_t)l*256 + st_c16*8);
        ksc_reg = Ksc[(size_t)l*8 + (st_c16 >> 2)];
      }
    }
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      // ---- commit K(t) (K raw words were prefetched during PV of t-1) ----
      {
        *(float4*)(SM + SM_KV0 + KSZ(st_row, st_c16)) = dq8(kreg, ksc_reg);
      }
      __syncthreads();                       // (A) K(t) visible; PV(t-1) V-reads done

      // ---- issue V(t) + K(t+1) raw-word loads -> regs (kreg/vreg both free;
      //      latency hides under QK + owners + PV; pure reg loads, no smem) ----
      {
        const int lv = tile + st_row;
        vreg = make_int2(0, 0); vsc_reg = __float2half(1.f);
        if (lv < l1) {
          vreg = *(const int2*)(Vc8 + (size_t)lv*256 + st_c16*8);
          vsc_reg = Vsc[(size_t)lv*8 + (st_c16 >> 2)];
        }
      }
      if (t + 1 < nt) {
        const int l = tile + TILE + st_row;
        kreg = make_int2(0, 0); ksc_reg = __float2half(1.f);
        if (l < l1) {
          kreg = *(const int2*)(Kc8 + (size_t)l*256 + st_c16*8);
          ksc_reg = Ksc[(size_t)l*8 + (st_c16 >> 2)];
        }
      }

      // ---- QK: one warp per out-tile (24 tiles), 16 k-steps each ----
      if (warp < QK_TILES) {
        const int mb = warp >> 2, nb = warp & 3;
        const int r0 = mb*16, n0 = nb*8;
        float4 c = make_float4(0.f, 0.f, 0.f, 0.f);
        _Pragma("unroll")
        for (int kb = 0; kb < QK_KS; ++kb) {
          const int kbase = kb*16;
          const int m1 = r0 + (lane >> 2);
          const int nn = n0 + (lane >> 2);        // key within tile
          // TRUE m16n8k16 A layout (probe-verified): a0=(m,k) a1=(m+8,k) a2=(m,k+8) a3=(m+8,k+8)
          const __half* QB = qw16 + (size_t)QROWIDX(m1)*256;
          const __half* QB2 = qw16 + (size_t)QROWIDX(m1+8)*256;
          const unsigned a0 = *(const unsigned*)(QB + kbase + (lane & 3)*2);
          const unsigned a1 = *(const unsigned*)(QB2 + kbase + (lane & 3)*2);
          const unsigned a2 = *(const unsigned*)(QB + kbase + (lane & 3)*2 + 8);
          const unsigned a3 = *(const unsigned*)(QB2 + kbase + (lane & 3)*2 + 8);
          const int dpair = (lane & 3)*2;
          const int chc = kbase >> 3;
          const __half k00 = KEL(nn, chc,     dpair);
          const __half k01 = KEL(nn, chc,     dpair + 1);
          const __half k10 = KEL(nn, chc + 1, dpair);
          const __half k11 = KEL(nn, chc + 1, dpair + 1);
          const unsigned b0 = h2u(__halves2half2(k00, k01));
          const unsigned b1 = h2u(__halves2half2(k10, k11));
          hmma16816(c, a0, a1, a2, a3, b0, b1);
        }
        const int r1 = r0 + (lane >> 2), n1 = n0 + (lane & 3)*2;
        SCP[(size_t)r1*TILE + n1] = c.x;
        SCP[(size_t)r1*TILE + n1 + 1] = c.y;
        SCP[(size_t)(r1+8)*TILE + n1] = c.z;
        SCP[(size_t)(r1+8)*TILE + n1 + 1] = c.w;
      }
      __syncthreads();                       // (B) SCP visible; K region dead

      // ---- row owners (warp w: rows w, w+32, w+64): reduce + online softmax ----
      _Pragma("unroll")
      for (int ri = 0; ri < ROWR; ++ri) {
        const int r = warp + ri*NW;
        if (r < RMAX) {
          const int t_ = r % ROWS;
          const int k = lane, ka = tile + k;
          float sc = -1e30f;
          if (ka < l1 && ka <= pos + t_) sc = SCP[(size_t)r*TILE + k];
          float scm = sc;
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) scm = fmaxf(scm, __shfl_xor_sync(FULL, scm, o));
          const float mold = ms_r[ri];
          const float mn = fmaxf(mold, scm);
          const float cor = expf(mold - mn);
          const float p = (sc > -1e29f) ? expf(sc - mn) : 0.f;
          Pm[(size_t)r*TILE + k] = (__half)p;
          float ps_ = p;
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) ps_ += __shfl_xor_sync(FULL, ps_, o);
          ss_r[ri] = ss_r[ri]*cor + ps_;
          ms_r[ri] = mn;
          if (lane == 0) corv[r] = cor;
        }
      }
      // ---- commit V(t) OVER the dead K region (concurrent with owners: the
      //      KV region has no readers between (B) and PV) ----
      {
        *(float4*)(SM + SM_KV0 + VSZ(st_row, st_c16)) = dq8(vreg, vsc_reg);
      }
      __syncthreads();                       // (C) Pm + V(t) visible

      // ---- PV: acc rescale + 2 k-steps of mma (A = Pm, B = V) ----
      _Pragma("unroll")
      for (int ti = 0; ti < PV_TPW; ++ti) {
        const int pt = warp + ti*NW;
        const int mb = pt >> 5, db = pt & 31;
        const int r0 = mb*16, d0 = db*8;
        const float cor0 = corv[r0 + (lane >> 2)];
        const float cor1 = corv[r0 + (lane >> 2) + 8];
        acc[ti].x *= cor0; acc[ti].y *= cor0;
        acc[ti].z *= cor1; acc[ti].w *= cor1;
        _Pragma("unroll")
        for (int kh = 0; kh < 2; ++kh) {
          const int m1 = r0 + (lane >> 2);
          const __half* PB = Pm;
          const unsigned a0 = *(const unsigned*)(PB + (size_t)m1*TILE + kh*16 + (lane & 3)*2);
          const unsigned a1 = *(const unsigned*)(PB + (size_t)(m1+8)*TILE + kh*16 + (lane & 3)*2);
          const unsigned a2 = *(const unsigned*)(PB + (size_t)m1*TILE + kh*16 + (lane & 3)*2 + 8);
          const unsigned a3 = *(const unsigned*)(PB + (size_t)(m1+8)*TILE + kh*16 + (lane & 3)*2 + 8);
          const int kr0 = kh*16 + (lane & 3)*2;
          const int bd = d0 + (lane >> 2);          // B-frag n = groupID (probe-verified)
          const __half v00 = VEL(kr0,     bd >> 3, bd & 7);
          const __half v01 = VEL(kr0 + 1, bd >> 3, bd & 7);
          const __half v10 = VEL(kr0 + 8, bd >> 3, bd & 7);
          const __half v11 = VEL(kr0 + 9, bd >> 3, bd & 7);
          const unsigned b0 = h2u(__halves2half2(v00, v01));
          const unsigned b1 = h2u(__halves2half2(v10, v11));
          hmma16816(acc[ti], a0, a1, a2, a3, b0, b1);
        }
      }
      __syncthreads();                       // (D) PV(t) done -> KV region rewritable
    }
  }

  // ---- final partial writes (all 96 rows; row owners write pm/ps) ----
  const size_t pb0 = (size_t)(g*S + s)*RMAX;
  _Pragma("unroll")
  for (int ri = 0; ri < ROWR; ++ri) {
    const int r = warp + ri*NW;
    if (r < RMAX && lane == 0) {
      pm[pb0 + r] = ms_r[ri];
      ps[pb0 + r] = ss_r[ri];
    }
  }
  _Pragma("unroll")
  for (int ti = 0; ti < PV_TPW; ++ti) {
    const int pt = warp + ti*NW;
    const int mb = pt >> 5, db = pt & 31;
    const int r0 = mb*16, d0 = db*8;
    const int m1 = r0 + (lane >> 2), dd = d0 + (lane & 3)*2;
    pA[(pb0 + m1)*256 + dd] = acc[ti].x;
    pA[(pb0 + m1)*256 + dd + 1] = acc[ti].y;
    pA[(pb0 + m1 + 8)*256 + dd] = acc[ti].z;
    pA[(pb0 + m1 + 8)*256 + dd + 1] = acc[ti].w;
  }
}

#elif KSEL == 2
// ============================== K2: pfc16 ==============================
extern "C" __global__ void __launch_bounds__(256) KNAME(
    const float* __restrict__ pm, const float* __restrict__ ps, const float* __restrict__ pA,
    const __half* __restrict__ qrow16, __half* __restrict__ ao16)
{
  const int h = blockIdx.x;
  const int g = h / 6, hl = h % 6;
  const int d = threadIdx.x;
  for (int t = 0; t < ROWS; ++t) {
    const int r = hl*ROWS + t;
    const size_t pb = (size_t)(g*S)*RMAX + r;
    float M = -1e30f;
    for (int s2 = 0; s2 < S; ++s2) M = fmaxf(M, pm[pb + (size_t)s2*RMAX]);
    float out = 0.f, Ssum = 0.f;
    for (int s2 = 0; s2 < S; ++s2) {
      const float ex = expf(pm[pb + (size_t)s2*RMAX] - M);
      Ssum += ps[pb + (size_t)s2*RMAX] * ex;
      out += pA[(pb + (size_t)s2*RMAX)*256 + d] * ex;
    }
    out /= Ssum;
    const float gf = __half2float(qrow16[(size_t)t*12288 + h*512 + 256 + d]);
    const float sg = 1.0f / (1.0f + expf(-gf));
    ao16[(size_t)t*6144 + h*256 + d] = __float2half(out * sg);
  }
}
#endif
