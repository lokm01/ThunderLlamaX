// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// W2G: HMMA (mma.sync.m16n8k16) rewrite of the K1 dots — QK and PV phases on
// tensor cores (2x the HFMA2 MAC ceiling on sm_86; the scalar build measures
// 88% of the fp16-vector math floor, so no scalar restructure can win).
// Structure per CTA (NW=32 warps, 1024 threads, grid 4*S unchanged):
//  - STAGE: canonical int2 dequant stage (unchanged math/layout -> same smem
//    bytes as spk_g4qh2).
//  - Q staged ONCE to smem [RP][256] fp16 (RP = RMAX rounded to 16; padding
//    rows zero). Extra smem is free: 1024-thread CTAs => 1 CTA/SM regardless.
//  - QK: out-tiles (RP/16 m-blocks) x (4 key-blocks of 8); QK_WPT warps per
//    tile k-split the 256 dims (4 k-steps of 16 each); partial c-frags go to
//    sc[QK_WPT][RMAX][TILE] fp32 planes (padding rows never stored; W2H diet).
//  - row owners (warp = row): sum planes in fixed ks order, then the SAME
//    online-softmax op sequence as the scalar kernel (max -> cor -> rescale
//    -> p -> sum via fixed xor butterflies), writing P[R][TILE] fp16 and
//    per-row cor to smem.
//  - PV: out-tiles (RP/16) x (32 dim-blocks of 8); each warp owns its tiles;
//    per tile: acc frags scaled by cor[row], 2 k-steps of mma with A = P
//    (16 rows x 16 keys), B = V (16 keys x 8 dims).
//  - final: pm/ps from the row-owner state; pA from PV acc frags (STG.64
//    consecutive-dim pairs); rows >= RMAX never written.
// NUMERICS: fp16 inputs, fp32 accumulate (BETTER precision class than the
// scalar fp16-chunk accumulators); reduction orders differ from the scalar
// build -> Tier-2 vs W2F canonical (T1 ref must be regenerated; sequence
// overlap reported). LAWS: single smem array, compile-time 16B-aligned
// offsets, no blockDim/gridDim reads, full masks, sequential tile loop.
// -DKNAME -DCTXK -DROWS (3|1) -DS -DCH -DTILE (32) -DNW (32)
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define RMAX (6*ROWS)
#ifndef NW
  #define NW 8
#endif
#define NTHR (NW*32)
#define LBOUNDS __launch_bounds__(NTHR)
#define RP (((RMAX + 15) / 16) * 16)
#define MB (RP / 16)            /* m-blocks */
#define QK_TILES (MB * 4)       /* x 4 key-blocks of 8 */
#define QK_WPT 1  /* single SCP plane: dext in-graph smem works to the ~36.4KB class only */
#if QK_WPT != 1
#error "W2H SCP plane packing ([RMAX][TILE] planes) assumes QK_WPT == 1"
#endif
#define QK_KS (16 / QK_WPT)     /* 16-dim k-steps per split warp */
#define PV_TILES (MB * 32)      /* x 32 dim-blocks of 8 */
#define PV_TPW ((PV_TILES) / NW)

#define SM_K0    0
#define SM_V0    (TILE*512)
#define SM_SC0   (SM_V0 + TILE*512)              /* 32KB: single SCP plane, 2.25KB (a3) */
#define SM_P0    SM_K0                            /* P ALIASES the K region: K is dead after QK+sync */
/* W2H SMEM DIET (in-graph-zero fix, hyp.1: dext in-graph backing ~36,864B):
 * (1) SCP plane is [RMAX][TILE] — the RP-RMAX padding rows were never read
 *     (rowvld) and are now also never written (guards on the c-frag stores);
 * (2) msv/ssv online-softmax state moved to row-owner REGISTERS (each row is
 *     owned by exactly one warp; only corv is read cross-warp by PV -> smem).
 * a3: 32768 + 18*32*4 + 32*4 = 35,200B (kept: free, leaner smem).
 * W2H POSTMORTEM: the smem-size class was NEVER the in-graph bug (a 35,840B
 * padded SCALAR kernel ran 60/60 in-graph); the real cause was the kernel
 * NAME lacking "nw32" -> the ParityGraph name-heuristic (gcycle.py) launched
 * it with 256 threads instead of 1024 in-graph. Cubins renamed spk_g4nw32hm*. */
#define SM_MS0   (SM_SC0 + QK_WPT*RMAX*TILE*4)
#define SM_BYTES (SM_MS0 + RP*4)
#define KSZ(R, C)  (((R) * 512) + (((C) ^ ((R) & 7)) * 16))
#define VSZ(R, C)  (((R) * 512) + ((C) * 16))
#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )
#define KEL(R, C, I) (*(const __half*)(SM + SM_K0 + KSZ(R, C) + 2*(I)))
#define VEL(R, C, I) (*(const __half*)(SM + SM_V0 + VSZ(R, C) + 2*(I)))

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

// canonical stage (int2 loads, all threads, K+V; identical to spk_g4qh2)
#define STAGE(BASEL) _Pragma("unroll") \
  for (int i = 0; i < TILE/NW; ++i) { \
    const int c = i*NTHR + tid; \
    const int row = c >> 5, c16 = c & 31; \
    const int l = (BASEL) + row; \
    float4 kk = make_float4(0.f, 0.f, 0.f, 0.f), vv = make_float4(0.f, 0.f, 0.f, 0.f); \
    if (l < l1) { \
      const int2 ki = *(const int2*)(Kc8 + (size_t)l*256 + c16*8); \
      const int2 vi = *(const int2*)(Vc8 + (size_t)l*256 + c16*8); \
      kk = dq8(ki, Ksc[(size_t)l*8 + (c16 >> 2)]); \
      vv = dq8(vi, Vsc[(size_t)l*8 + (c16 >> 2)]); \
    } \
    *(float4*)(SM + SM_K0 + KSZ(row, c16)) = kk; \
    *(float4*)(SM + SM_V0 + VSZ(row, c16)) = vv; \
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

extern "C" __global__ void LBOUNDS KNAME(
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

  float* corv = (float*)(SM + SM_MS0);          // [RP] per-row correction (read cross-warp by PV)
  __half* Pm = (__half*)(SM + SM_P0);           // [RP][TILE]
  float* SCP = (float*)(SM + SM_SC0);           // [QK_WPT][RMAX][TILE]
  // R5 LAW FIX: when RP > NW (ROWS=6: RP=48 vs 32 warps) a warp must OWN
  // MULTIPLE rows — single-row owners left rows NW..RP-1 (h%%6==5 heads, t>=2)
  // UNWRITTEN in pm/ps -> combine 0/0 = NaN. MAXOWN=1 for ROWS<=5 (RP<=NW):
  // codegen-identical to the proven single-row path.
  #define MAXOWN ((RP + NW - 1) / NW)
  float ms_r[MAXOWN > 0 ? MAXOWN : 1], ss_r[MAXOWN > 0 ? MAXOWN : 1];
  _Pragma("unroll") for (int oi = 0; oi < MAXOWN; ++oi) { ms_r[oi] = -1e30f; ss_r[oi] = 0.f; }

  // PV accumulators (this warp's tiles), fp32 c-frags
  float4 acc[PV_TPW > 0 ? PV_TPW : 1];
  _Pragma("unroll") for (int ti = 0; ti < PV_TPW; ++ti) acc[ti] = make_float4(0.f, 0.f, 0.f, 0.f);

  if (l0 < l1) {
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      __syncthreads();
      STAGE(tile)
      __syncthreads();

      // ---- QK: QK_WPT warps per out-tile, k-split ----
      {
        const int tileid = warp % QK_TILES;
        const int ks = warp / QK_TILES;           // 0..QK_WPT-1
        if (ks < QK_WPT) {
        const int mb = tileid >> 2, nb = tileid & 3; // m-block, key-block(8)
        const int r0 = mb*16, n0 = nb*8;          // key index within tile
        float4 c = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll 2
        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {
          const int kb = ks*QK_KS + ks4;     // 16-dim k-step within 256
          const int kbase = kb*16;
          const int m1 = r0 + (lane >> 2);
          const int nn = n0 + (lane >> 2);        // absolute dim (key col)
          // TRUE m16n8k16 A layout (probe-verified): a0=(m,k), a1=(m+8,k), a2=(m,k+8), a3=(m+8,k+8)
          unsigned a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
          if (m1 < RMAX) {
            const __half* QB = qw16 + (size_t)QROWIDX(m1)*256;
            a0 = *(const unsigned*)(QB + kbase + (lane & 3)*2);
            a2 = *(const unsigned*)(QB + kbase + (lane & 3)*2 + 8);
          }
          if (m1 + 8 < RMAX) {
            const __half* QB2 = qw16 + (size_t)QROWIDX(m1+8)*256;
            a1 = *(const unsigned*)(QB2 + kbase + (lane & 3)*2);
            a3 = *(const unsigned*)(QB2 + kbase + (lane & 3)*2 + 8);
          }
          // B[k][n] = Ksmem[key=n][dim=k]: row = nn (key), chunk/element = dim
          const int dpair = (lane & 3)*2;   // dim pair within this 16-k window
          const int chc = kbase >> 3;       // dim chunk (16B) of kbase
          const __half k00 = KEL(nn, chc,     dpair);
          const __half k01 = KEL(nn, chc,     dpair + 1);
          const __half k10 = KEL(nn, chc + 1, dpair);
          const __half k11 = KEL(nn, chc + 1, dpair + 1);
          const unsigned b0 = h2u(__halves2half2(k00, k01));
          const unsigned b1 = h2u(__halves2half2(k10, k11));
          hmma16816(c, a0, a1, a2, a3, b0, b1);
        }
        // partial c-frags -> sc plane (fixed lane mapping, deterministic);
        // rows >= RMAX never stored NOR read (padding rows; plane is [RMAX][TILE])
        const int r1 = r0 + (lane >> 2), n1 = n0 + (lane & 3)*2;
        if (r1 < RMAX) {
          SCP[((size_t)ks*RMAX + r1)*TILE + n1] = c.x;
          SCP[((size_t)ks*RMAX + r1)*TILE + n1 + 1] = c.y;
        }
        if (r1 + 8 < RMAX) {
          SCP[((size_t)ks*RMAX + r1 + 8)*TILE + n1] = c.z;
          SCP[((size_t)ks*RMAX + r1 + 8)*TILE + n1 + 1] = c.w;
        }
        }
      }
      __syncthreads();

      // ---- row owners: reduce planes + online softmax (same op order as scalar:
      // ---- lane = key; fixed xor butterflies for max and sum) ----
      _Pragma("unroll")
      for (int oi = 0; oi < MAXOWN; ++oi) {
        const int r = warp + oi*NW;
        if (r >= RP) break;
        const int t_ = r % ROWS;
        const bool rowvld = (r < RMAX);
        const int k = lane, ka = tile + k;
        float sc = -1e30f;
        if (rowvld && ka < l1 && ka <= pos + t_) {
          sc = 0.f;
          _Pragma("unroll") for (int ksi = 0; ksi < QK_WPT; ++ksi)
            sc += SCP[((size_t)ksi*RMAX + r)*TILE + k];
        }
        float scm = sc;
        _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) scm = fmaxf(scm, __shfl_xor_sync(FULL, scm, o));
        const float mold = ms_r[oi];
        const float mn = fmaxf(mold, scm);
        const float cor = expf(mold - mn);
        const float p = (sc > -1e29f) ? expf(sc - mn) : 0.f;
        Pm[(size_t)r*TILE + k] = (__half)p;
        float ps_ = p;
        _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) ps_ += __shfl_xor_sync(FULL, ps_, o);
        ss_r[oi] = ss_r[oi]*cor + ps_;
        ms_r[oi] = mn;
        corv[r] = cor;
      }
      __syncthreads();

      // ---- PV: acc-frags rescale + 2 k-steps of mma (A = P, B = V) ----
      _Pragma("unroll")
      for (int ti = 0; ti < PV_TPW; ++ti) {
        const int pt = warp + ti*NW;
        const int mb = pt >> 5, db = pt & 31;      // m-block, dim-block(8)
        const int r0 = mb*16, d0 = db*8;
        const float cor0 = corv[r0 + (lane >> 2)];
        const float cor1 = corv[r0 + (lane >> 2) + 8];
        acc[ti].x *= cor0; acc[ti].y *= cor0;
        acc[ti].z *= cor1; acc[ti].w *= cor1;
        _Pragma("unroll")
        for (int kh = 0; kh < 2; ++kh) {           // 2 x 16-key steps
          const int m1 = r0 + (lane >> 2);
          const int dd = d0 + (lane & 3)*2;
          const __half* PB = (const __half*)(SM + SM_P0);
          unsigned a0 = *(const unsigned*)(PB + (size_t)m1*TILE + kh*16 + (lane & 3)*2);
          unsigned a1 = *(const unsigned*)(PB + (size_t)(m1+8)*TILE + kh*16 + (lane & 3)*2);
          unsigned a2 = *(const unsigned*)(PB + (size_t)m1*TILE + kh*16 + (lane & 3)*2 + 8);
          unsigned a3 = *(const unsigned*)(PB + (size_t)(m1+8)*TILE + kh*16 + (lane & 3)*2 + 8);
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
      __syncthreads();
    }
  }

  // ---- final partial writes (rows < RMAX only) ----
  const size_t pb0 = (size_t)(g*S + s)*RMAX;
  _Pragma("unroll")
  for (int oi = 0; oi < MAXOWN; ++oi) {
    const int r = warp + oi*NW;
    if (r < RP && r < RMAX && lane == 0) {
      pm[pb0 + r] = ms_r[oi];
      ps[pb0 + r] = ss_r[oi];
    }
  }
  _Pragma("unroll")
  for (int ti = 0; ti < PV_TPW; ++ti) {
    const int pt = warp + ti*NW;
    const int mb = pt >> 5, db = pt & 31;
    const int r0 = mb*16, d0 = db*8;
    const int m1 = r0 + (lane >> 2), dd = d0 + (lane & 3)*2;
    if (m1 < RMAX) {
      pA[(pb0 + m1)*256 + dd] = acc[ti].x;
      pA[(pb0 + m1)*256 + dd + 1] = acc[ti].y;
    }
    if (m1 + 8 < RMAX) {
      pA[(pb0 + m1 + 8)*256 + dd] = acc[ti].z;
      pA[(pb0 + m1 + 8)*256 + dd + 1] = acc[ti].w;
    }
  }
}
