// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P9 probe: pfa32c — co-resident attention (512thr x 64regs x 2 CTAs/SM target).
// Kimi's design; grok kill-switch protocol (see P9_PROBE.md). Ported from the
// PROVEN pfa8t64 skeleton (pf_attn8.cu KSEL=3) + pfa16 (pf_attn.cu KSEL=1):
//   - R=32 q-rows/CTA (2 GQA head-rows x 16 t-rows; 16c: R=16, 1 head-row)
//   - grid (g,s,hp) ordered hp-fastest: the NHP CTAs re-reading the same KV
//     split are adjacent -> L2-adjacent re-reads (the theory under test)
//   - fp16 HMMA QK (m16n8k16); K B-frags dequantized DIRECT from global kv8 u8
//     (byte_perm trick; same per-element dq8 values as the staged path)
//   - V dq8-staged f16 into WARP-PRIVATE channel slices (each warp stages only
//     the channels its own PV tiles read -> the PV/V-stage race is structurally
//     impossible -> 3 barriers/tile without the 4th "PV done" barrier)
//   - SCP f32 -> Pm f16 IN-PLACE plane (pfa8t64 trick)
//   - owners single-pass TILE-wide online softmax (P7F2 law 5)
//   - PV tile remap: warp w owns channels [w*CHW, w*CHW+CHW) (one 32-ch scale
//     group) -> 16B int2 staging + single scale per key
// Variants: -DPFA_T32=1 (TILE=32, K+V smem-staged f16, 4 barriers) and
// -DPFA_R16=1 (R=16, 256thr, grid 4xSx6).
// NOTE: CH = ceil(CTXK/S) — S=13 does not divide 100352 (2^11*7^2); floor-div
// would drop the last 5 key positions at pos=100336 (l1 clamp covers the rest).
// Laws kept: single 16B-aligned smem, compile-time offsets, no blockDim/gridDim
// reads, full masks, sequential tile loop, no cp.async, no runtime-indexed
// locals, l1-guarded global K reads, per-kernel cubins + symbol check.
// -DCTXK -DS -DKNAME -DMINB(0=none,1,2,3)
#include <cuda_fp16.h>
#ifdef PFA_T32
  #define TILE 32
  #define NW 16
  #define HRP 2
#else
  #ifndef TILE
    #define TILE 64
  #endif
  #ifdef PFA_R16
    #define NW 8
    #define HRP 1
  #else
    #define NW 16
    #define HRP 2
  #endif
#endif
#define NTHR (NW*32)
#define NHP (6/HRP)                  // 3 (t64/t32) or 6 (16c)
#define NCTA (4*S*NHP)               // hardcoded grid size (the no-gridDim law; A4 ticket)
#define CHW (256/NW)                 // warp-private staging width: 16 or 32 ch
#define CH (((CTXK) + S - 1) / S)    // CEIL split
#define FULL 0xffffffffu
#define ROWS 16
#define RMAX (HRP*ROWS)              // 32 (t64/t32) or 16 (16c)
#define RP RMAX
#define MB (RP/16)
#define QK_TILES (MB*(TILE/8))
#define PV_TILES (MB*32)
#define PV_TPW (PV_TILES/NW)
#define ROWR (RP/NW)

#ifdef PFA_T32
  #define SM_K0    0                       // K f16 [TILE][512B] swizzled
  #define SM_V0    (TILE*512)              // V f16 [TILE][512B]
  #define SM_SC0   (SM_V0 + TILE*512)      // SCP f32 [RP][TILE] -> Pm f16 in-place
  #define SM_MS0   (SM_SC0 + RP*TILE*4)
  #define SM_BYTES (SM_MS0 + RP*4)
  #define KSZ(R, C)  (((R) * 512) + (((C) ^ ((R) & 7)) * 16))
  #define KEL(R, C, I) (*(const __half*)(SM + SM_K0 + KSZ(R, C) + 2*(I)))
#else
  #define SM_V0    0                       // V f16 [64][512B] = 32KB
  #define SM_SC0   (TILE*512)              // SCP f32 [RP][TILE] -> Pm in-place
  #define SM_SK0   (SM_SC0 + RP*TILE*4)    // sk f16 [TILE][8]
  #define SM_MS0   (SM_SK0 + TILE*16)
  #define SM_BYTES (SM_MS0 + RP*4)
#endif
#define VSZ(R, C)  (((R) * 512) + ((C) * 16))
#define VEL(R, C, I) (*(const __half*)(SM + SM_V0 + VSZ(R, C) + 2*(I)))
#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + hp*HRP + (R)/ROWS )

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
// dequant a 2-byte u8 pair (2 channels, one scale) -> half2 as u32 B-frag reg
__device__ __forceinline__ unsigned dqu16(const unsigned x, const __half s) {
  const unsigned p = __byte_perm(x, 0x64646464u, 0x5140);
  return h2u(DQH2(*(const __half2*)&p, __half2half2(s)));
}
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
#ifdef PFA_LC
    // P10-A4: + qrow16/ao16 (the combine inputs) + ctr (self-resetting ticket)
    const unsigned char* __restrict__ kv, const __half* __restrict__ sc, const __half* __restrict__ qw16,
    const int* __restrict__ pos_slot, float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA,
    const __half* __restrict__ qrow16, __half* __restrict__ ao16, unsigned int* __restrict__ ctr)
#else
    const unsigned char* __restrict__ kv, const __half* __restrict__ sc, const __half* __restrict__ qw16,
    const int* __restrict__ pos_slot, float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
#endif
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

  float* corv = (float*)(SM + SM_MS0);
  __half* Pm = (__half*)(SM + SM_SC0);     // Pm f16 lives in the low half of the plane
  float* SCP = (float*)(SM + SM_SC0);
#ifndef PFA_T32
  __half* SK = (__half*)(SM + SM_SK0);     // [TILE][8] f16
#endif
  float ms_r[ROWR], ss_r[ROWR];
  _Pragma("unroll") for (int ri = 0; ri < ROWR; ++ri) { ms_r[ri] = -1e30f; ss_r[ri] = 0.f; }
  float4 acc[PV_TPW];
  _Pragma("unroll") for (int ti = 0; ti < PV_TPW; ++ti) acc[ti] = make_float4(0.f, 0.f, 0.f, 0.f);

#ifdef PFA_T32
  // ---- linear staging ring: K(t) + V(t), 2 int2 each per thread (12 regs) ----
  const int st_row = tid >> 4, st_c16 = (tid & 15) << 1;   // 32 rows x 32 c16 slots
  int2 kreg0, kreg1, vreg0, vreg1; __half ksc0, ksc1, vsc0, vsc1;
  {
    const int lr0 = l0 + st_row;
    kreg0 = make_int2(0, 0); kreg1 = make_int2(0, 0); ksc0 = __float2half(1.f); ksc1 = ksc0;
    vreg0 = make_int2(0, 0); vreg1 = make_int2(0, 0); vsc0 = __float2half(1.f); vsc1 = vsc0;
    if (lr0 < l1) {
      kreg0 = *(const int2*)(Kc8 + (size_t)lr0*256 + st_c16*8);      ksc0 = Ksc[(size_t)lr0*8 + (st_c16 >> 2)];
      vreg0 = *(const int2*)(Vc8 + (size_t)lr0*256 + st_c16*8);      vsc0 = Vsc[(size_t)lr0*8 + (st_c16 >> 2)];
    }
    if (lr0 < l1) {
      kreg1 = *(const int2*)(Kc8 + (size_t)lr0*256 + st_c16*8 + 8);  ksc1 = Ksc[(size_t)lr0*8 + ((st_c16+1) >> 2)];
      vreg1 = *(const int2*)(Vc8 + (size_t)lr0*256 + st_c16*8 + 8);  vsc1 = Vsc[(size_t)lr0*8 + ((st_c16+1) >> 2)];
    }
  }
#else
  // ---- warp-private staging ring: warp w owns channels [w*CHW, w*CHW+CHW)
  //      (exactly its PV read set); lane stages keys {lane, lane+32} ----
  const int my_key0 = lane, my_key1 = lane + 32;
  const int my_c16 = warp * (CHW >> 3);                    // first c16 slot of the slice
  const int my_gp = (warp * CHW) >> 5;                     // the ONE 32-ch scale group
  int2 vr[2][CHW >> 3]; __half vsr[2];
  #define VLOAD(T, K2I) do { \
    const int lv = (T) + ((K2I) ? my_key1 : my_key0); \
    _Pragma("unroll") for (int q = 0; q < (CHW >> 3); ++q) vr[K2I][q] = make_int2(0, 0); \
    vsr[K2I] = __float2half(1.f); \
    if (lv < l1) { \
      _Pragma("unroll") for (int q = 0; q < (CHW >> 3); ++q) \
        vr[K2I][q] = *(const int2*)(Vc8 + (size_t)lv*256 + warp*CHW + q*8); \
      vsr[K2I] = Vsc[(size_t)lv*8 + my_gp]; \
    } \
  } while (0)
#endif

  if (l0 < l1) {
    const int nt = (l1 - l0 + TILE - 1) / TILE;
#ifndef PFA_T32
    VLOAD(l0, 0); VLOAD(l0, 1);   // prologue: V(0) ring
#endif
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
#ifndef PFA_T32
      // ---- commit V(t) (warp-private slots; PV(t-1) reads only own slots) + sk(t) ----
      _Pragma("unroll") for (int k2i = 0; k2i < 2; ++k2i) {
        const int kk = k2i ? my_key1 : my_key0;
        _Pragma("unroll") for (int q = 0; q < (CHW >> 3); ++q)
          *(float4*)(SM + SM_V0 + VSZ(kk, my_c16 + q)) = dq8(vr[k2i][q], vsr[k2i]);
      }
      if (tid < TILE) {
        uint4 skr = make_uint4(0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u);
        if (tile + tid < l1) skr = *(const uint4*)(Ksc + (size_t)(tile + tid)*8);
        *(uint4*)(SM + SM_SK0 + tid*16) = skr;
      }
#endif
#ifdef PFA_T32
      __syncthreads();   // (1) PV(t-1) done -> V region safe to rewrite
      // ---- stage K(t) + V(t) ----
      {
        const int lr0 = tile + st_row;
        if (lr0 < l1) {
          *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16))       = dq8(kreg0, ksc0);
          *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16))       = dq8(vreg0, vsc0);
        } else {
          *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16))       = make_float4(0.f,0.f,0.f,0.f);
          *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16))       = make_float4(0.f,0.f,0.f,0.f);
        }
        if (lr0 < l1) {
          *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16 + 1)) = dq8(kreg1, ksc1);
          *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16 + 1)) = dq8(vreg1, vsc1);
        } else {
          *(float4*)(SM + SM_K0 + KSZ(st_row, st_c16 + 1)) = make_float4(0.f,0.f,0.f,0.f);
          *(float4*)(SM + SM_V0 + VSZ(st_row, st_c16 + 1)) = make_float4(0.f,0.f,0.f,0.f);
        }
      }
#endif
      __syncthreads();   // (t64/16c: S1) V(t)+sk(t) visible | (t32: S2) K,V visible

#ifndef PFA_T32
      // reload V(t+1) ring (latency hides under QK+owners+PV)
      if (t + 1 < nt) { VLOAD(tile + TILE, 0); VLOAD(tile + TILE, 1); }
#endif
#ifdef PFA_T32
      // reload K(t+1)+V(t+1) raw (K region dead after QK(t); V dead after PV(t))
      {
        const int lr0 = tile + TILE + st_row;
        kreg0 = make_int2(0, 0); kreg1 = make_int2(0, 0); ksc0 = __float2half(1.f); ksc1 = ksc0;
        vreg0 = make_int2(0, 0); vreg1 = make_int2(0, 0); vsc0 = __float2half(1.f); vsc1 = vsc0;
        if (lr0 < l1) {
          kreg0 = *(const int2*)(Kc8 + (size_t)lr0*256 + st_c16*8);      ksc0 = Ksc[(size_t)lr0*8 + (st_c16 >> 2)];
          vreg0 = *(const int2*)(Vc8 + (size_t)lr0*256 + st_c16*8);      vsc0 = Vsc[(size_t)lr0*8 + (st_c16 >> 2)];
        }
        if (lr0 < l1) {
          kreg1 = *(const int2*)(Kc8 + (size_t)lr0*256 + st_c16*8 + 8);  ksc1 = Ksc[(size_t)lr0*8 + ((st_c16+1) >> 2)];
          vreg1 = *(const int2*)(Vc8 + (size_t)lr0*256 + st_c16*8 + 8);  vsc1 = Vsc[(size_t)lr0*8 + ((st_c16+1) >> 2)];
        }
      }
#endif

      // ---- QK(t): one warp per 16x8 out-tile; A=Q global f16; B=K ----
      if (warp < QK_TILES) {
        const int nb8 = TILE >> 3;
        const int mb = warp / nb8, nb = warp % nb8;
        const int r0 = mb*16, n0 = nb*8;
        const int m1 = r0 + (lane >> 2);
        const __half* QB  = qw16 + (size_t)QROWIDX(m1)*256;
        const __half* QB2 = qw16 + (size_t)QROWIDX(m1+8)*256;
        float4 c = make_float4(0.f, 0.f, 0.f, 0.f);
        _Pragma("unroll")
        for (int kb = 0; kb < 16; ++kb) {
          const int kbase = kb*16;
          const unsigned a0 = *(const unsigned*)(QB  + kbase + (lane & 3)*2);
          const unsigned a1 = *(const unsigned*)(QB2 + kbase + (lane & 3)*2);
          const unsigned a2 = *(const unsigned*)(QB  + kbase + (lane & 3)*2 + 8);
          const unsigned a3 = *(const unsigned*)(QB2 + kbase + (lane & 3)*2 + 8);
#ifdef PFA_T32
          const int nn = n0 + (lane >> 2);
          const int chc = kbase >> 3, dpair = (lane & 3)*2;
          const __half k00 = KEL(nn, chc,     dpair);
          const __half k01 = KEL(nn, chc,     dpair + 1);
          const __half k10 = KEL(nn, chc + 1, dpair);
          const __half k11 = KEL(nn, chc + 1, dpair + 1);
          const unsigned b0 = h2u(__halves2half2(k00, k01));
          const unsigned b1 = h2u(__halves2half2(k10, k11));
#else
          const int nl = n0 + (lane >> 2);             // key within tile
          const __half sk = SK[nl*8 + (kb >> 1)];      // the one 32-ch group of this k-step
          unsigned b0 = 0u, b1 = 0u;
          if (tile + nl < l1) {                        // l1 guard (law: global B-frags)
            const size_t koff = (size_t)(tile + nl)*256 + kbase + (lane & 3)*2;
            b0 = dqu16(*(const unsigned short*)(Kc8 + koff),     sk);
            b1 = dqu16(*(const unsigned short*)(Kc8 + koff + 8), sk);
          }
#endif
          hmma16816(c, a0, a1, a2, a3, b0, b1);
        }
        const int r1 = r0 + (lane >> 2), n1 = n0 + (lane & 3)*2;
        SCP[(size_t)r1*TILE + n1]       = c.x;
        SCP[(size_t)r1*TILE + n1 + 1]   = c.y;
        SCP[(size_t)(r1+8)*TILE + n1]   = c.z;
        SCP[(size_t)(r1+8)*TILE + n1 + 1] = c.w;
      }
      __syncthreads();   // (t64: S2 / t32: S3) SCP(t) visible

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
      __syncthreads();   // (t64: S3 / t32: S4) Pm + corv visible

      // ---- PV(t): warp-private channels; TILE/16 k-steps of m16n8k16 ----
      _Pragma("unroll")
      for (int ti = 0; ti < PV_TPW; ++ti) {
#ifdef PFA_R16
        const int mb = 0, db = 4*warp + ti;            // channels [32w, 32w+32)
#else
        const int mb = ti >> 1, db = 2*warp + (ti & 1); // channels [16w, 16w+16)
#endif
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
#ifdef PFA_R16
    const int mb = 0, db = 4*warp + ti;
#else
    const int mb = ti >> 1, db = 2*warp + (ti & 1);
#endif
    const int r0 = mb*16, d0 = db*8;
    const int m1 = r0 + (lane >> 2), dd = d0 + (lane & 3)*2;
    pA[(pb0 + m1)*256 + dd] = acc[ti].x;
    pA[(pb0 + m1)*256 + dd + 1] = acc[ti].y;
    pA[(pb0 + m1 + 8)*256 + dd] = acc[ti].z;
    pA[(pb0 + m1 + 8)*256 + dd + 1] = acc[ti].w;
  }

#ifdef PFA_LC
  // ---- P10-A4: self-resetting last-CTA combine (kills the pfc16 launch).
  // Every CTA publishes partials, takes a device-scope ticket; the LAST CTA of
  // the hardcoded NCTA combines ALL 24 heads x 16 rows x 256 ch (same s-order
  // and epilogue math as pfc16t) and resets the counter for the next launch
  // (launches are QMD-ordered, so the reset lands before any re-read).
  // NCTA/NTHR are compile-time macros (the no-gridDim/blockDim law).
  __threadfence();                      // partials visible before the ticket
  __shared__ int is_last_s;
  if (tid == 0) is_last_s = (atomicAdd(ctr, 1u) == (unsigned)(NCTA - 1));
  __syncthreads();
  if (is_last_s) {
    __threadfence();                    // acquire side: other CTAs' partials
    const int NTOT = 24 * ROWS * 256;   // 98304 outputs
    for (int idx = tid; idx < NTOT; idx += NTHR) {
      const int d = idx % 256;
      const int h = (idx / 256) % 24;   // 24*256 = 6144 per t (NOT a power of 2)
      const int t = idx / 6144;
      const int g = h / 6, hl = h % 6;
      const int hp = hl / HRP, hli = hl % HRP;
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
    if (tid == 0) *ctr = 0;             // self-reset (next launch is ordered)
  }
#endif
}
#endif  // !PFC_T32

#ifdef PFC_T32
// ---- P10 K2 for the pfa32ct layout (S=13, NHP=3, HRP=2, RMAX=32): partial
// slot (g*(NHP*S) + s*NHP + hp)*RMAX + (hli*ROWS + t); fixed-order combine +
// sigmoid gate, same math/op-order as the shipped pfc16 epilogue (Tier-2 vs
// pfa16 by TILE/S regrouping only). grid 24 heads, 256 thr (one per channel).
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
