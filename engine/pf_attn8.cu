// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P7F-2: IMMA int8 QK attention (m16n8k32.s8, P7A probe-validated fragments).
// Derived from pf_attn.cu (pfa16 skeleton) with THREE changes:
//   (1) QK dot engine: Q per-(row,32ch-group) s8 (pfk_q8 quantizer) x K raw
//       biased-u8 XOR 0x80 -> s8 staged (NO dq8 dequant for K); one
//       m16n8k32 per 32-ch scale group; s32->f32 epilogue rescale by
//       sq[row][g]*sk[pos][g]; fp32 score accum -> same SCP plane.
//   (2) K (s8, 8KB) and V (f16, 16KB) in SEPARATE smem regions -> the tile
//       loop drops from 4 to 2 __syncthreads per key tile.
//   (3) PV unchanged (fp16 HMMA on dq8-staged V; the sv-per-(key,chan-group)
//       scale cannot be factored out of an int8 PV epilogue -> PV stays fp16).
// Numerics: Tier-2 (Q int8 relerr ~1e-3 class). Laws: single 16B-aligned
// smem, compile-time offsets, no blockDim/gridDim reads, full masks,
// sequential tile loop, no cp.async, per-kernel cubins, nw token in name.
// -DCTXK -DS -DTILE(32) -DNW(32) -DKNAME -DKSEL(1=pfa8, 2=pfk_q8)
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
#define RP RMAX
#define MB (RP / 16)           // 6
#define QK_TILES (MB * 4)      // 24
#define PV_TILES (MB * 32)     // 192
#define PV_TPW (PV_TILES / NW) // 6
#define ROWR (RP / NW)         // 3

#define SM_K0    0                        // K s8  [TILE][256]      8192
#define SM_V0    (TILE*256)               // V f16 [TILE][512B]    16384
#define SM_SK0   (SM_V0 + TILE*512)       // sk f16 [TILE][8]        512
#define SM_SC0   (SM_SK0 + TILE*16)       // SCP f32 [RMAX][TILE]  12288
#define SM_P0    (SM_SC0 + RMAX*TILE*4)   // Pm f16 [RP][TILE]      6144
#define SM_MS0   (SM_P0 + RP*TILE*2)      // corv f32 [RP]           384
#define SM_BYTES (SM_MS0 + RP*4)
#define VSZ(R, C)  (SM_V0 + ((R) * 512) + ((C) * 16))
#define QROWIDX(R) ( ((R) % ROWS)*24 + g*6 + (R)/ROWS )
#define VEL(R, C, I) (*(const __half*)(SM + VSZ(R, C) + 2*(I)))

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
// P7A-validated m16n8k32.s8 (dext bit-exact, 293 TOPS probe):
//   A: a0=(gid,tig*4) a1=(gid+8,tig*4) a2=(gid,16+tig*4) a3=(gid+8,16+tig*4)
//   B: b0=(tig*4,n=gid) b1=(16+tig*4,n=gid)   C: rows gid/gid+8, cols 2*tig{,+1}
__device__ __forceinline__ void imma16832(int &c0, int &c1, int &c2, int &c3,
    const unsigned a0, const unsigned a1, const unsigned a2, const unsigned a3,
    const unsigned b0, const unsigned b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#if KSEL == 1
// ============================== pfa8 ==============================
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ kv, const __half* __restrict__ sc,
    const signed char* __restrict__ qs8, const float* __restrict__ qsc,
    const int* __restrict__ pos_slot,
    float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
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

  float* corv = (float*)(SM + SM_MS0);
  __half* Pm = (__half*)(SM + SM_P0);
  float* SCP = (float*)(SM + SM_SC0);
  __half* SK = (__half*)(SM + SM_SK0);
  float ms_r[ROWR], ss_r[ROWR];
  _Pragma("unroll") for (int ri = 0; ri < ROWR; ++ri) { ms_r[ri] = -1e30f; ss_r[ri] = 0.f; }
  float4 acc[PV_TPW];
  _Pragma("unroll") for (int ti = 0; ti < PV_TPW; ++ti) acc[ti] = make_float4(0.f, 0.f, 0.f, 0.f);

  const int st_row = tid >> 5, st_c16 = tid & 31;
  int2 kreg; uint4 skreg; int2 vreg; __half vsc_reg;

  if (l0 < l1) {
    const int nt = (l1 - l0 + TILE - 1) / TILE;
    { // prologue: commit K(0) + sk(0)
      const int l = l0 + st_row;
      kreg = make_int2(0, 0);
      if (l < l1) kreg = *(const int2*)(Kc8 + (size_t)l*256 + st_c16*8);
      *(int2*)(SM + SM_K0 + st_row*256 + st_c16*8) = make_int2(kreg.x ^ (int)0x80808080, kreg.y ^ (int)0x80808080);
      if (st_c16 == 0) {
        skreg = make_uint4(0, 0, 0, 0);
        if (l < l1) skreg = *(const uint4*)(Ksc + (size_t)l*8);
        *(uint4*)(SM + SM_SK0 + st_row*16) = skreg;
      }
    }
    __syncthreads();   // K(0) + sk(0) visible

    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*TILE;
      // ---- issue V(t) + K(t+1) raw loads (pure reg loads; latency hides
      //      under QK + owners) ----
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
        kreg = make_int2(0, 0);
        if (l < l1) kreg = *(const int2*)(Kc8 + (size_t)l*256 + st_c16*8);
        if (st_c16 == 0) {
          skreg = make_uint4(0, 0, 0, 0);
          if (l < l1) skreg = *(const uint4*)(Ksc + (size_t)l*8);
        }
      }

      // ---- QK(t): 24 warps, m16n8k32 per 32-ch scale group ----
#if ATTR != 2 && ATTR != 3 && ATTR != 4
      if (warp < QK_TILES) {
        const int mb = warp >> 2, nb = warp & 3;
        const int r0 = mb*16, n0 = nb*8;
        const int m1 = r0 + (lane >> 2);
        const signed char* QB  = qs8 + (size_t)QROWIDX(m1)*256;
        const signed char* QB2 = qs8 + (size_t)QROWIDX(m1+8)*256;
        const float* SC1 = qsc + (size_t)QROWIDX(m1)*8;
        const float* SC2 = qsc + (size_t)QROWIDX(m1+8)*8;
        const int kbn = n0 + (lane >> 2);
        const int c0l = n0 + (lane & 3)*2, c1l = c0l + 1;
        float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
        _Pragma("unroll")
        for (int gp = 0; gp < 8; ++gp) {
          const int kq = gp*32 + (lane & 3)*4;
          const unsigned a0 = *(const unsigned*)(QB + kq);
          const unsigned a1 = *(const unsigned*)(QB2 + kq);
          const unsigned a2 = *(const unsigned*)(QB + kq + 16);
          const unsigned a3 = *(const unsigned*)(QB2 + kq + 16);
          const unsigned b0 = *(const unsigned*)(SM + SM_K0 + kbn*256 + kq);
          const unsigned b1 = *(const unsigned*)(SM + SM_K0 + kbn*256 + kq + 16);
          int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
          imma16832(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
          const float sk0 = __half2float(SK[c0l*8 + gp]);
          const float sk1 = __half2float(SK[c1l*8 + gp]);
          f0 += (float)c0 * (SC1[gp] * sk0);
          f1 += (float)c1 * (SC1[gp] * sk1);
          f2 += (float)c2 * (SC2[gp] * sk0);
          f3 += (float)c3 * (SC2[gp] * sk1);
        }
        SCP[(size_t)m1*TILE + c0l] = f0;
        SCP[(size_t)m1*TILE + c1l] = f1;
        SCP[(size_t)(m1+8)*TILE + c0l] = f2;
        SCP[(size_t)(m1+8)*TILE + c1l] = f3;
      }
#endif
      __syncthreads();   // S1: SCP(t) visible; K region dead; PV(t-1) done

      // ---- row owners (warp w: rows w, w+32, w+64) ----
#if ATTR != 1 && ATTR != 3 && ATTR != 4
      _Pragma("unroll")
      for (int ri = 0; ri < ROWR; ++ri) {
        const int r = warp + ri*NW;
        if (r < RMAX) {
          const int t_ = r % ROWS;
          const int k = lane, ka = tile + k;
          float scv = -1e30f;
          if (ka < l1 && ka <= pos + t_) scv = SCP[(size_t)r*TILE + k];
          float scm = scv;
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) scm = fmaxf(scm, __shfl_xor_sync(FULL, scm, o));
          const float mold = ms_r[ri];
          const float mn = fmaxf(mold, scm);
          const float cor = expf(mold - mn);
          const float p = (scv > -1e29f) ? expf(scv - mn) : 0.f;
          Pm[(size_t)r*TILE + k] = (__half)p;
          float ps_ = p;
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) ps_ += __shfl_xor_sync(FULL, ps_, o);
          ss_r[ri] = ss_r[ri]*cor + ps_;
          ms_r[ri] = mn;
          if (lane == 0) corv[r] = cor;
        }
      }
#endif
      // ---- commit V(t) + K(t+1) + sk(t+1) ----
      {
        *(float4*)(SM + VSZ(st_row, st_c16)) = dq8(vreg, vsc_reg);
        if (t + 1 < nt) {
          *(int2*)(SM + SM_K0 + st_row*256 + st_c16*8) = make_int2(kreg.x ^ (int)0x80808080, kreg.y ^ (int)0x80808080);
          if (st_c16 == 0) *(uint4*)(SM + SM_SK0 + st_row*16) = skreg;
        }
      }
      __syncthreads();   // S2: Pm(t) + V(t) + K(t+1) visible

      // ---- PV(t): acc rescale + 2 k-steps of fp16 mma (A = Pm, B = V) ----
#if ATTR != 1 && ATTR != 2 && ATTR != 4
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
#endif
      // QK(t+1) starts next iteration; K(t+1) visible since S2; V region is
      // rewritten only after S1 of t+1 (PV(t) done there).
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
// ============================== pfk_q8 ==============================
// Q quantizer: qw f16 [384 rows per half = 16 t x 24 head-rows][256] ->
// qs8 s8 + qsc f32 per (row, 32-ch group). Warp per row; grid 48 x 256thr.
extern "C" __global__ void __launch_bounds__(256) KNAME(
    const __half* __restrict__ qw, signed char* __restrict__ qs8, float* __restrict__ qsc)
{
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) + (blockIdx.x << 3);   // 8 warps/CTA
  if (warp >= 384) return;
  const __half* QR = qw + (size_t)warp*256;
  float v[8];
  _Pragma("unroll") for (int i = 0; i < 8; ++i) v[i] = __half2float(QR[lane*8 + i]);
  float am = 0.f;
  _Pragma("unroll") for (int i = 0; i < 8; ++i) am = fmaxf(am, fabsf(v[i]));
  am = fmaxf(am, __shfl_xor_sync(FULL, am, 1));
  am = fmaxf(am, __shfl_xor_sync(FULL, am, 2));
  if ((lane & 3) == 0) qsc[(size_t)warp*8 + (lane >> 2)] = am / 127.f;
  unsigned lo = 0u, hi = 0u;
  if (am > 0.f) {
    const float inv = 127.f / am;
    _Pragma("unroll") for (int i = 0; i < 4; ++i) lo |= (unsigned)(__float2int_rn(v[i]*inv) & 0xff) << (8*i);
    _Pragma("unroll") for (int i = 0; i < 4; ++i) hi |= (unsigned)(__float2int_rn(v[i+4]*inv) & 0xff) << (8*(i&3));
  }
  *(uint2*)(qs8 + (size_t)warp*256 + lane*8) = make_uint2(lo, hi);
}
#endif
// ============================== pfa8t64 (v3) ==============================
// TILE=64 keys/iteration; K B-frags DIRECT from global kv (u32 + XOR 0x80);
// V dq8-dequant staged f16 [64][512B]; SCP+Pm f16; 2 barriers per 64 keys.
// smem: V 32768 | SCP f16 12288 | Pm f16 12288 | skf 2048 | corv 384 = 59.8KB
// (64KB carveout class — decode spk_g4hm precedent). Q per-(row,32ch) s8.
#if KSEL == 3
#define V64(r, c)  (SM + SM_V0 + ((r) * 512) + ((c) * 16))
#define SM_V0   0
#define SM_SC0  (64*512)
#define SM_P0   SM_SC0            // in-place: owners read SCP then write Pm
#define SM_SK0  (SM_SC0 + RMAX*64*2)
#define SM_MS0  (SM_SK0 + 64*8*4)
#define SM_BYTES_V3 (SM_MS0 + RP*4)
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ kv, const __half* __restrict__ sc,
    const signed char* __restrict__ qs8, const float* __restrict__ qsc,
    const int* __restrict__ pos_slot,
    float* __restrict__ pm, float* __restrict__ ps, float* __restrict__ pA)
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
  __shared__ __align__(16) char SM[SM_BYTES_V3];

  float* corv = (float*)(SM + SM_MS0);
  __half* Pm = (__half*)(SM + SM_P0);
  __half* SCP = (__half*)(SM + SM_SC0);
  float* SK = (float*)(SM + SM_SK0);            // [64][8] f32
  float ms_r[ROWR], ss_r[ROWR];
  _Pragma("unroll") for (int ri = 0; ri < ROWR; ++ri) { ms_r[ri] = -1e30f; ss_r[ri] = 0.f; }
  float4 acc[PV_TPW];
  _Pragma("unroll") for (int ti = 0; ti < PV_TPW; ++ti) acc[ti] = make_float4(0.f, 0.f, 0.f, 0.f);

  const int st_row = tid >> 5, st_c16 = tid & 31;   // 64 rows x 64 slots? V rows=64

  if (l0 < l1) {
    const int nt = (l1 - l0 + 63) / 64;
    for (int t = 0; t < nt; ++t) {
      const int tile = l0 + t*64;
      // ---- stage V(t) (dq8 dequant) + sk(t) f32 ----
      {
        _Pragma("unroll")
        for (int h2i = 0; h2i < 2; ++h2i) {
          const int lv = tile + h2i*32 + st_row;
          int2 vr = make_int2(0, 0); __half vs = __float2half(1.f);
          if (lv < l1) { vr = *(const int2*)(Vc8 + (size_t)lv*256 + st_c16*8); vs = Vsc[(size_t)lv*8 + (st_c16 >> 2)]; }
          *(float4*)(V64(h2i*32 + st_row, st_c16)) = dq8(vr, vs);
        }
        if (st_c16 < 16) {
          const int lr0 = tile + st_c16*4;         // 64 rows x 4/thread
          _Pragma("unroll")
          for (int u = 0; u < 4; ++u) {
            float s0 = 1.f, s1 = 1.f, s2 = 1.f, s3 = 1.f;
            if (lr0 + u < l1) {
              const __half* kr = Ksc + (size_t)(lr0 + u)*8;
              s0 = __half2float(kr[0]); s1 = __half2float(kr[1]); s2 = __half2float(kr[2]); s3 = __half2float(kr[3]);
            }
            *(float4*)(SM + SM_SK0 + (st_c16*4 + u)*32) = make_float4(s0, s1, s2, s3);
          }
        }
      }
      __syncthreads();   // S1: V(t) + sk(t) visible; PV(t-1) done

      // ---- QK(t): 24 warps, K B-frags direct from global (XOR trick) ----
      if (warp < QK_TILES) {
        const int mb = warp >> 2, nb = warp & 3;
        const int r0 = mb*16, n0 = nb*16;         // n0 = 16 keys per warp (64/4)
        const int m1 = r0 + (lane >> 2);
        const int qoff1 = QROWIDX(m1)*256, qoff2 = QROWIDX(m1+8)*256;
        const int so1 = QROWIDX(m1)*8, so2 = QROWIDX(m1+8)*8;
        const int kbn = (n0 + (lane >> 2)) * 256;
        const int c0l = n0 + (lane & 3)*2, c1l = c0l + 1;
        float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
        _Pragma("unroll")
        for (int gp = 0; gp < 8; ++gp) {
          const int kq = gp*32 + (lane & 3)*4;
          const unsigned a0 = *(const unsigned*)(qs8 + qoff1 + kq);
          const unsigned a1 = *(const unsigned*)(qs8 + qoff2 + kq);
          const unsigned a2 = *(const unsigned*)(qs8 + qoff1 + kq + 16);
          const unsigned a3 = *(const unsigned*)(qs8 + qoff2 + kq + 16);
          const size_t koff = (size_t)(tile + n0 + (lane >> 2))*256;
          const unsigned b0 = ((tile + n0 + (lane >> 2)) < l1) ? ((*(const unsigned*)(Kc8 + koff + kq)) ^ 0x80808080u) : 0u;
          const unsigned b1 = ((tile + n0 + (lane >> 2)) < l1) ? ((*(const unsigned*)(Kc8 + koff + kq + 16)) ^ 0x80808080u) : 0u;
          int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
          imma16832(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
          const float sk0 = SK[c0l*8 + gp];
          const float sk1 = SK[c1l*8 + gp];
          f0 += (float)c0 * (qsc[so1 + gp] * sk0);
          f1 += (float)c1 * (qsc[so1 + gp] * sk1);
          f2 += (float)c2 * (qsc[so2 + gp] * sk0);
          f3 += (float)c3 * (qsc[so2 + gp] * sk1);
        }
        SCP[(size_t)m1*64 + c0l] = (__half)f0;
        SCP[(size_t)m1*64 + c1l] = (__half)f1;
        SCP[(size_t)(m1+8)*64 + c0l] = (__half)f2;
        SCP[(size_t)(m1+8)*64 + c1l] = (__half)f3;
      }
      __syncthreads();   // S2: SCP(t) visible

      // ---- row owners: online softmax over 64 keys; Pm f16 (single pass) ----
      _Pragma("unroll")
      for (int ri = 0; ri < ROWR; ++ri) {
        const int r = warp + ri*NW;
        if (r < RMAX) {
          const int t_ = r % ROWS;
          const int ka0 = tile + lane, ka1 = tile + lane + 32;
          float scv0 = -1e30f, scv1 = -1e30f;
          if (ka0 < l1 && ka0 <= pos + t_) scv0 = __half2float(SCP[(size_t)r*64 + lane]);
          if (ka1 < l1 && ka1 <= pos + t_) scv1 = __half2float(SCP[(size_t)r*64 + lane + 32]);
          float scm = fmaxf(scv0, scv1);
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) scm = fmaxf(scm, __shfl_xor_sync(FULL, scm, o));
          const float mold = ms_r[ri];
          const float mn = fmaxf(mold, scm);
          const float cor = expf(mold - mn);
          const float p0 = (scv0 > -1e29f) ? expf(scv0 - mn) : 0.f;
          const float p1 = (scv1 > -1e29f) ? expf(scv1 - mn) : 0.f;
          Pm[(size_t)r*64 + lane] = (__half)p0;
          Pm[(size_t)r*64 + lane + 32] = (__half)p1;
          float ps_ = p0 + p1;
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) ps_ += __shfl_xor_sync(FULL, ps_, o);
          ss_r[ri] = ss_r[ri]*cor + ps_;
          ms_r[ri] = mn;
          if (lane == 0) corv[r] = cor;
        }
      }
      __syncthreads();   // S3: Pm(t) visible

      // ---- PV(t): 4 k-steps of fp16 mma per out-tile (k=64 keys) ----
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
        for (int kh = 0; kh < 4; ++kh) {
          const int m1 = r0 + (lane >> 2);
          const __half* PB = Pm;
          const unsigned a0 = *(const unsigned*)(PB + (size_t)m1*64 + kh*16 + (lane & 3)*2);
          const unsigned a1 = *(const unsigned*)(PB + (size_t)(m1+8)*64 + kh*16 + (lane & 3)*2);
          const unsigned a2 = *(const unsigned*)(PB + (size_t)m1*64 + kh*16 + (lane & 3)*2 + 8);
          const unsigned a3 = *(const unsigned*)(PB + (size_t)(m1+8)*64 + kh*16 + (lane & 3)*2 + 8);
          const int kr0 = kh*16 + (lane & 3)*2;
          const int bd = d0 + (lane >> 2);
          const __half v00 = *(__half*)(V64(kr0,     bd >> 3) + 2*(bd & 7));
          const __half v01 = *(__half*)(V64(kr0 + 1, bd >> 3) + 2*(bd & 7));
          const __half v10 = *(__half*)(V64(kr0 + 8, bd >> 3) + 2*(bd & 7));
          const __half v11 = *(__half*)(V64(kr0 + 9, bd >> 3) + 2*(bd & 7));
          const unsigned b0 = h2u(__halves2half2(v00, v01));
          const unsigned b1 = h2u(__halves2half2(v10, v11));
          hmma16816(acc[ti], a0, a1, a2, a3, b0, b1);
        }
      }
      // next iteration stages V over the dead region only after PV done -> S1
    }
  }

  // ---- final partial writes ----
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
#endif
