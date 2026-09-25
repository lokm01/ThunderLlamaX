// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P1 PREFILL pGEMM: batched M=16 dequant-GEMM family over the engine's packed
// quant weights. out[16][N] = x[16][K] . W[N][K]^T with W in GGUF/packed quant
// layouts (decode addressing COPIED VERBATIM from the bit-proven T=1 kernels in
// w1c.cu — only the lane->(row, chunk) assignment changes; per-chunk decode is
// identical math, fp32 -> half at the same rounding point as the references).
//
// Structure per CTA (NTHR threads, NTILE W-rows, KCH k per chunk):
//   stage X: x[16][KCH] -> smem fp16 (rows padded +8 halfs for bank spread)
//   stage W: each warp decodes NTILE/8 rows; lane = (row r = lane>>2, quarter
//            c = lane&3) decodes KCH/32 consecutive 8-k lane-chunks of block
//            (kc&255)>>3 + ... — the reference's lane-chunk math parameterized
//            by chunk index lc (same 0..31 domain as the reference's lane).
//   compute: HMMA mma.sync.m16n8k16.row.col.f32.f16.f16.f32 (W2G/W2H probe-
//            verified fragment map: a0=(m,k) a1=(m+8,k) a2=(m,k+8) a3=(m+8,k+8),
//            b-n = lane>>2, c-n = (lane&3)*2) OR HFMA2 fallback (fp16 products,
//            fp32 acc — same per-element rounding class as the GEMV refs).
//   FFN=1: fused gate+up (two W pointers), gact = silu(half(aG)) * half(aU)
//            — epilogue identical to ffn8 (hsilu_h verbatim).
//
// LAWS honored: flat indexing only; no gridDim/blockDim reads; hardcoded block
// sizes; sequential/unrolled loops; full-warp masks (none needed — no shuffles);
// single-array smem with 16B-aligned regions; per-kernel cubin + unique KNAME
// carrying the warp token (nw8/nw16/nw32 for 256/512/1024 threads) for the
// gcycle name-encoded launch config; static smem kept <= ~32KB per build combo.
//
// Build: -DKNAME -DQCLASS(1..7) -DKDIM -DNDIM -DNTHR -DNTILE -DKCH -DHMMA -DFFN
//   QCLASS 1=iq3xxs(packed) 2=q5k(raw) 3=q6k(packed) 4=q4k(raw)
//          5=iq3s(raw) 6=q8_0(raw) 7=q4_0(draft packed)
//   P2: -DRES=1 = fp32 residual-add output mode (down-projection class):
//   out32[m][n] = res16[m][n] + fp32 acc — matches down8's y = hh + acc (the
//   P1 law-4 dtype contract; no fp16 rounding of the acc before the residual).
//   RES+FFN is a build error. RES=0/default path byte-identical to P1.
#include <cuda_fp16.h>
#if RES && FFN
#error "RES+FFN not supported"
#endif
#if KS && FFN
#error "KS+FFN not supported"
#endif
#if KS && RES && !M32
#error "KS+RES not supported (classic; M32 KS2 partials keep RES for the combine)"
#endif
#define FULL 0xffffffffu
// P6: M32=1 builds the 32-x-row kernel (two m16 mma fragments share every
// staged W tile — the quant-word load stream amortizes 2x). See the P6
// section at the bottom of this file.
#ifndef M32
#define M32 0
#endif
#if M32 && !HMMA
#error "M32: HMMA only"
#endif
#if M32 && DBG != 0
#error "M32: DBG 0 only"
#endif
#if M32 && (DBUF || SYNCW)
#error "M32: classic path only (no DBUF/SYNCW)"
#endif
#if M32 && KS && FFN
#error "M32 KS: FFN not supported (G1 = the iq3d/iq3o/iq3s classes only)"
#endif

#if QCLASS == 1
#define ROWBYTES ((size_t)98 * (KDIM >> 8))
#elif QCLASS == 2
#define ROWBYTES ((size_t)176 * (KDIM >> 8))
#elif QCLASS == 3
#define ROWBYTES ((size_t)212 * (KDIM >> 8))
#elif QCLASS == 4
#define ROWBYTES ((size_t)144 * (KDIM >> 8))
#elif QCLASS == 5
#define ROWBYTES ((size_t)110 * (KDIM >> 8))
#elif QCLASS == 6
#define ROWBYTES ((size_t)34 * (KDIM >> 5))
#else
#define ROWBYTES ((size_t)(KDIM / 2 + KDIM / 16))
#endif

#define XS_LD (KCH + 8)      // x smem row stride (halfs)
#define WS_LD (KCH + 8)      // w smem row stride (halfs)
#define NWARP (NTHR / 32)
#define RPT (NTILE / NWARP)  // W rows per warp
#define NCL (KCH / 32)       // 8k lane-chunks per (row, quarter-lane)
#define KSTEPS (KCH / 16)

__device__ __forceinline__ __half hsilu_h(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}
__device__ __forceinline__ unsigned h2u(const __half2 h) { return *reinterpret_cast<const unsigned*>(&h); }

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


// ---- P5 H2: IQ3 decode via half2-grid LUT + sign-XOR LUT (see SYNCW section) ----
__device__ __forceinline__ void dq1_h2(const unsigned char* rowp, const unsigned char* lut,
                                       int kc, int lc, __half* wsr, int kk) {
  const int b_ = kc >> 8;
  const float d = __half2float(__ushort_as_half(((const unsigned short*)(rowp + 96*(KDIM >> 8)))[b_]));
  const unsigned int sw = ((const unsigned int*)(rowp + 64*(KDIM >> 8)))[8*b_ + (lc >> 2)];
  const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
  const unsigned int sidx = (sw >> (7u * (unsigned int)(lc & 3))) & 0x7Fu;
  const unsigned int q = ((const unsigned short*)rowp)[32*b_ + lc];
  const __half2 db2 = __float2half2_rn(db);
  const __half2* gh = (const __half2*)lut;
  const __half2 ga0 = gh[((q & 0xFFu) << 1) + 0];
  const __half2 ga1 = gh[((q & 0xFFu) << 1) + 1];
  const __half2 gb0 = gh[((q >> 8) << 1) + 0];
  const __half2 gb1 = gh[((q >> 8) << 1) + 1];
  const uint4 smv = *(const uint4*)(lut + 2048 + (sidx << 4));
  *(uint2*)(wsr + kk)     = make_uint2(h2u(__hmul2(db2, ga0)) ^ smv.x,
                                       h2u(__hmul2(db2, ga1)) ^ smv.y);
  *(uint2*)(wsr + kk + 4) = make_uint2(h2u(__hmul2(db2, gb0)) ^ smv.z,
                                       h2u(__hmul2(db2, gb1)) ^ smv.w);
}

// ---- decode one 8k lane-chunk lc of 256k-block b of W row r -> 8 halfs ----
// (addressing VERBATIM from w1c.cu references, lane -> lc)
__device__ __forceinline__ void dq_chunk_iq3(const unsigned char* rowp, const float* gridf,
                                             int b, int lc, __half* w8) {
  const unsigned short* qsp = (const unsigned short*)(rowp);
  const unsigned int* scp = (const unsigned int*)(rowp + 64*(KDIM >> 8));
  const unsigned short* dpp = (const unsigned short*)(rowp + 96*(KDIM >> 8));
  const float d = __half2float(__ushort_as_half(dpp[b]));
  const unsigned int sw = scp[8*b + (lc>>2)];
  const float db = d * (((float)(sw >> 28)) + 0.5f) * 0.5f;
  const unsigned int sidx = (sw >> (7u * (unsigned int)(lc & 3))) & 0x7Fu;
  const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
  const unsigned int q = qsp[32*b + lc];
  const float4 g0 = *((const float4*)(gridf + ((q & 0xFFu) << 2)));
  const float4 g1 = *((const float4*)(gridf + ((q >> 8) << 2)));
  const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
  const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
  const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
  const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
  w8[0] = __float2half(db*g0.x*sg0); w8[1] = __float2half(db*g0.y*sg1);
  w8[2] = __float2half(db*g0.z*sg2); w8[3] = __float2half(db*g0.w*sg3);
  w8[4] = __float2half(db*g1.x*sg4); w8[5] = __float2half(db*g1.y*sg5);
  w8[6] = __float2half(db*g1.z*sg6); w8[7] = __float2half(db*g1.w*sg7);
}

__device__ __forceinline__ void dq_chunk_q5(const unsigned char* rowp, int b, int lc, __half* w8) {
  const unsigned char* blk = rowp + b*176;
  const float d = __half2float(*((const __half*)blk));
  const float dm = __half2float(*((const __half*)(blk+2)));
  const int s = lc >> 2;
  float sc, mn;
  if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
  else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
         mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
  const unsigned char* lo = blk + 48 + ((lc>>3)<<5) + ((lc&3)<<3);
  const unsigned char* qh = blk + 16 + ((lc&3)<<3);
  const int nsh = ((lc>>2)&1) << 2;
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) {
    int qv = (lo[j] >> nsh) & 0xF;
    qv += (int)(((qh[j] >> s) & 1u) << 4);
    w8[j] = __float2half(d*sc*(float)qv - dm*mn);
  }
}

__device__ __forceinline__ void dq_chunk_q6(const unsigned char* rowp, int b, int lc, __half* w8) {
  const unsigned char* blk = rowp + b*212;   // PACKED: [pad2][d2][sc16][lo128][qh64]
  const float d = __half2float(*((const __half*)(blk+2)));
  const bool nib_hi = ((lc&15) >= 8);
  const int c2 = (lc>>2)&3;
  const unsigned char* lo = blk + 20 + ((lc>>4)<<6) + ((lc&7)<<3);
  const unsigned char* qh = blk + 148 + ((lc>>4)<<5) + ((lc&3)<<3);
  const int sc8 = (signed char)blk[4 + (lc>>1)];
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) {
    const int xl = nib_hi ? (lo[j] >> 4) & 0xF : lo[j] & 0xF;
    const int xh2 = ((qh[j] >> (c2<<1)) & 3) << 4;
    w8[j] = __float2half(d * (float)sc8 * (float)((signed char)((xl | xh2) - 32)));
  }
}

__device__ __forceinline__ void dq_chunk_q4k(const unsigned char* rowp, int b, int lc, __half* w8) {
  const unsigned char* blk = rowp + b*144;
  const float d = __half2float(*((const __half*)blk));
  const float dm = __half2float(*((const __half*)(blk+2)));
  const int s = lc >> 2;
  float sc, mn;
  if (s < 4) { sc = (float)(blk[4+s] & 63); mn = (float)(blk[8+s] & 63); }
  else { sc = (float)((blk[8+s] & 0xF) | ((blk[s] >> 6) << 4));
         mn = (float)((blk[8+s] >> 4) | ((blk[s+4] >> 6) << 4)); }
  const unsigned char* lo = blk + 16 + ((lc>>3)<<5) + ((lc&3)<<3);
  const int nsh = ((lc>>2)&1) << 2;
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) {
    const float qv = (float)((lo[j] >> nsh) & 0xF);
    w8[j] = __float2half(d*sc*qv - dm*mn);
  }
}

__device__ __forceinline__ void dq_chunk_iq3s(const unsigned char* rowp, const float* grid512,
                                              int b, int lc, __half* w8) {
  const unsigned char* blk = rowp + (size_t)b*110;
  const float d = __half2float(*((const __half*)blk));
  const int g0 = lc*2, g1 = lc*2 + 1;
  const int sraw = lc >> 2;
  const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
  const unsigned char* sgnb = blk + 74 + lc;
  const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lc);
  const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0>>3)] >> (g0&7)) & 1u) << 8));
  const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1>>3)] >> (g1&7)) & 1u) << 8));
  const float4 gv0 = *((const float4*)(grid512 + (qb0 << 2)));
  const float4 gv1 = *((const float4*)(grid512 + (qb1 << 2)));
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) {
    const float gv = (j < 4) ? ((&gv0.x)[j]) : ((&gv1.x)[j-4]);
    const float sgn = ((*sgnb >> j) & 1) ? -1.f : 1.f;
    w8[j] = __float2half(d * sc * gv * sgn);
  }
}

// Q8_0: 32k blocks of [d f16][32 i8] — decode KCH/4 k's starting at kc + c*(KCH/4)
__device__ __forceinline__ void dq_run_q8(const unsigned char* rowp, int kc, int c, __half* w16) {
  _Pragma("unroll")
  for (int i = 0; i < KCH/4; ++i) {
    const int k = kc + c*(KCH/4) + i;
    const unsigned char* blk = rowp + (size_t)(k >> 5)*34;
    const float d = __half2float(*((const __half*)blk));
    w16[i] = __float2half(d * (float)((signed char)blk[2 + (k & 31)]));
  }
}

// Q4_0 draft packed: [qs K/2][d K/16] — GGUF nibble interleave (dfgu-verified):
// weight k (32k sub-block b32, offset o) = byte b32*16 + (o>=16 ? 8 : 0) + (o&7),
// high nibble iff o >= 16; d = f16 at qs_end + b32*2. w = d*(qv-8).
__device__ __forceinline__ void dq_run_q4z(const unsigned char* rowp, int kc, int c, __half* w16) {
  _Pragma("unroll")
  for (int i = 0; i < KCH/4; ++i) {
    const int k = kc + c*(KCH/4) + i;
    const int b32 = k >> 5, o = k & 31;
    const unsigned char qb = rowp[(b32 << 4) + ((o >> 4) << 3) + (o & 7)];
    const int qv = (o >= 16) ? (qb >> 4) & 0xF : qb & 0xF;
    const float d = __half2float(*((const __half*)(rowp + KDIM/2 + (b32 << 1))));
    w16[i] = __float2half(d * (float)(qv - 8));
  }
}

// generic 256k-class chunk dispatcher
__device__ __forceinline__ void dq_chunk(const unsigned char* rowp, const float* gridf,
                                         int kc, int lc, __half* w8) {
#if QCLASS == 1
  dq_chunk_iq3(rowp, gridf, (kc >> 8), lc, w8);
#elif QCLASS == 2
  dq_chunk_q5(rowp, (kc >> 8), lc, w8);
#elif QCLASS == 3
  dq_chunk_q6(rowp, (kc >> 8), lc, w8);
#elif QCLASS == 4
  dq_chunk_q4k(rowp, (kc >> 8), lc, w8);
#elif QCLASS == 5
  dq_chunk_iq3s(rowp, gridf, (kc >> 8), lc, w8);
#endif
}

// ---- stage W rows of one chunk into smem ----
__device__ __forceinline__ void stage_w(__half* ws, const unsigned char* w,
                                        const float* gridf,
#if H2
                                        const unsigned char* lut,
#endif
                                        int kc, int tid, int lane, int warp, int n0) {
  const int r = lane >> 2;            // row within warp's RPT rows (RPT == 8 builds)
  const int c = lane & 3;             // quarter
  const unsigned char* rowp0 = w + (size_t)(n0 + warp*RPT) * ROWBYTES;
  const int lc0 = (kc & 255) >> 3;    // first lane-chunk of this chunk in its block
#if QCLASS == 6
  {
    const unsigned char* rowp = rowp0 + (size_t)r * ROWBYTES;
    __half w16[KCH/4];
    dq_run_q8(rowp, kc, c, w16);
    _Pragma("unroll")
    for (int i = 0; i < KCH/4; ++i) ws[(size_t)(warp*RPT + r)*WS_LD + c*(KCH/4) + i] = w16[i];
  }
#elif QCLASS == 7
  {
    const unsigned char* rowp = rowp0 + (size_t)r * ROWBYTES;
    __half w16[KCH/4];
    dq_run_q4z(rowp, kc, c, w16);
    _Pragma("unroll")
    for (int i = 0; i < KCH/4; ++i) ws[(size_t)(warp*RPT + r)*WS_LD + c*(KCH/4) + i] = w16[i];
  }
#else
  // 32 lanes = 8 rows (r = lane>>2) x 4 quarters (c = lane&3); each lane decodes
  // NCL consecutive 8k lane-chunks of its row (the reference's lane-chunk math,
  // parameterized by chunk index lc in the same 0..31 domain)
  {
    const unsigned char* rowp = rowp0 + (size_t)r * ROWBYTES;
    _Pragma("unroll")
    for (int cc = 0; cc < NCL; ++cc) {
      const int lc = lc0 + c*NCL + cc;
      const int kk = (c*NCL + cc)*8;   // smem column within the chunk
#if H2
      dq1_h2(rowp, lut, kc, lc, ws + (size_t)(warp*RPT + r)*WS_LD + kk, 0);
#else
      __half w8[8];
      dq_chunk(rowp, gridf, kc, lc, w8);
      _Pragma("unroll")
      for (int j = 0; j < 8; ++j) ws[(size_t)(warp*RPT + r)*WS_LD + kk + j] = w8[j];
#endif
    }
  }
#endif
}

// ---- stage X rows of one chunk ----
__device__ __forceinline__ void stage_x(__half* xs, const __half* x16, int kc, int tid) {
  _Pragma("unroll")
  for (int i = 0; i < (16*KCH + NTHR*4 - 1)/(NTHR*4); ++i) {
    const int lin = (tid + i*NTHR) * 4;
    if (lin < 16*KCH) {
      const int m = lin / KCH, kk = lin % KCH;
      *(uint2*)(xs + (size_t)m*XS_LD + kk) = *(const uint2*)(x16 + (size_t)m*KDIM + kc + kk);
    }
  }
}

#if !DBUF && !SYNCW && !M32
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
#if FFN
    const unsigned char* __restrict__ w2,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
#if RES || KS
#if RES
    const float* __restrict__ res16,
#endif
    float* __restrict__ out32
#else
    __half* __restrict__ out16
#endif
#if H2
    , const unsigned char* __restrict__ lut
#endif
)
{
  // SINGLE-ARRAY SMEM LAW: xs at 0 (16*XS_LD halfs), ws at +16*XS_LD (16B-aligned:
  // 16*XS_LD*2 % 16 == 0 since XS_LD is a multiple of 8). FFN: gate plane then up plane.
#if FFN
  __shared__ __align__(16) __half sm[16 * XS_LD + 2 * NTILE * WS_LD];
#else
  __shared__ __align__(16) __half sm[16 * XS_LD + NTILE * WS_LD];
#endif
  __half* xs = sm;
  __half* ws = sm + 16 * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
#if KS
  const int tile_ = blockIdx.x / KS, ks_ = blockIdx.x % KS;
  const int n0 = tile_ * NTILE;
  const int kc0_ = ks_ * (KDIM / KS), kc1_ = kc0_ + (KDIM / KS);
#else
  const int n0 = blockIdx.x * NTILE;
  const int kc0_ = 0, kc1_ = KDIM;
#endif
#if FFN
  float cG0 = 0.f, cG1 = 0.f, cG2 = 0.f, cG3 = 0.f;
  float cU0 = 0.f, cU1 = 0.f, cU2 = 0.f, cU3 = 0.f;
#else
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#endif

  _Pragma("unroll 2")
  for (int kc = kc0_; kc < kc1_; kc += KCH) {
#if DBG != 2
    stage_x(xs, x16, kc, tid);
#endif
#if DBG == 0 || DBG == 2 || DBG == 3
#if FFN
    stage_w(ws,         w1, gridf,
#if H2
            lut,
#endif
            kc, tid, lane, warp, n0);
    stage_w(ws + NTILE*WS_LD, w2, gridf,
#if H2
            lut,
#endif
            kc, tid, lane, warp, n0);
#else
    stage_w(ws, w1, gridf,
#if H2
            lut,
#endif
            kc, tid, lane, warp, n0);
#endif
#endif
    __syncthreads();
#if DBG == 1
    if (tid == 0) out16[(blockIdx.x & 15)*NDIM + n0] = xs[(kc / KCH) & 7];
    __syncthreads();
    continue;
#elif DBG == 2
    if (tid == 0) out16[(blockIdx.x & 15)*NDIM + n0] = ws[(warp*RPT)*WS_LD];
    __syncthreads();
    continue;
#elif DBG == 3
    // decode dump: first chunk only, first CTA: ws[row][col] -> out16[row][col]
    if (blockIdx.x == 0 && kc == 0) {
      _Pragma("unroll")
      for (int i = 0; i < 8; ++i) {
        const int idx = tid*8 + i;              // 2048 = 16 rows x 128 cols
        out16[(idx / KCH)*NDIM + (idx % KCH)] = ws[(idx / KCH)*WS_LD + (idx % KCH)];
      }
    }
    __syncthreads();
    continue;
#endif
#if HMMA
    // thread (g=lane>>2, t=lane&3); warp owns out cols [n0 + warp*8, +8)
    _Pragma("unroll")
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s*16;
      const int g = lane >> 2, tp = (lane & 3) * 2;
      const unsigned a0 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp);
      const unsigned a1 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp);
      const unsigned a2 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp + 8);
      const unsigned a3 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp + 8);
      {
        const __half* wr = ws + (size_t)(warp*8 + g)*WS_LD + kb + tp;
        const unsigned b0 = *(const unsigned*)(wr);
        const unsigned b1 = *(const unsigned*)(wr + 8);
#if FFN
        hmma16816(cG0, cG1, cG2, cG3, a0, a1, a2, a3, b0, b1);
#else
        hmma16816(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
      }
#if FFN
      {
        const __half* wr = ws + NTILE*WS_LD + (size_t)(warp*8 + g)*WS_LD + kb + tp;
        const unsigned b0 = *(const unsigned*)(wr);
        const unsigned b1 = *(const unsigned*)(wr + 8);
        hmma16816(cU0, cU1, cU2, cU3, a0, a1, a2, a3, b0, b1);
      }
#endif
    }
#else
    // HFMA2: thread (g, t) owns out rows {g, g+8} x cols {2t, 2t+1} of the warp tile
    const int g = lane >> 2, tp = (lane & 3) * 2;
    _Pragma("unroll 4")
    for (int kk = 0; kk < KCH; kk += 2) {
      const __half2 xa = *(const __half2*)(xs + (size_t)g*XS_LD + kk);
      const __half2 xb = *(const __half2*)(xs + (size_t)(g+8)*XS_LD + kk);
      {
        const __half2 b0 = *(const __half2*)(ws + (size_t)(warp*8 + tp)*WS_LD + kk);
        const __half2 b1 = *(const __half2*)(ws + (size_t)(warp*8 + tp + 1)*WS_LD + kk);
        const __half2 p0 = __hmul2(xa, b0), p1 = __hmul2(xb, b0);
        const __half2 p2 = __hmul2(xa, b1), p3 = __hmul2(xb, b1);
        float2 f0 = __half22float2(p0), f1 = __half22float2(p1);
        float2 f2 = __half22float2(p2), f3 = __half22float2(p3);
#if FFN
        cG0 += f0.x + f0.y; cG1 += f2.x + f2.y; cG2 += f1.x + f1.y; cG3 += f3.x + f3.y;
#else
        c0 += f0.x + f0.y; c1 += f2.x + f2.y; c2 += f1.x + f1.y; c3 += f3.x + f3.y;
#endif
      }
#if FFN
      {
        const __half2 b0 = *(const __half2*)(ws + NTILE*WS_LD + (size_t)(warp*8 + tp)*WS_LD + kk);
        const __half2 b1 = *(const __half2*)(ws + NTILE*WS_LD + (size_t)(warp*8 + tp + 1)*WS_LD + kk);
        const __half2 p0 = __hmul2(xa, b0), p1 = __hmul2(xb, b0);
        const __half2 p2 = __hmul2(xa, b1), p3 = __hmul2(xb, b1);
        float2 f0 = __half22float2(p0), f1 = __half22float2(p1);
        float2 f2 = __half22float2(p2), f3 = __half22float2(p3);
        cU0 += f0.x + f0.y; cU1 += f2.x + f2.y; cU2 += f1.x + f1.y; cU3 += f3.x + f3.y;
      }
#endif
    }
#endif
    __syncthreads();
  }

  // c-frag -> out[m][n]: c0=(g, n), c1=(g, n+1), c2=(g+8, n), c3=(g+8, n+1); n = 2t
  const int g = lane >> 2, tp = (lane & 3) * 2;
  const int n = n0 + warp*8 + tp;
#if DBG == 3
  return;   // decode-dump mode: keep the dumped ws values
#endif
#if KS
  float* op = out32 + (size_t)ks_*16*NDIM;
  op[(size_t)g*NDIM + n]           = c0;
  op[(size_t)g*NDIM + n + 1]       = c1;
  op[(size_t)(g+8)*NDIM + n]       = c2;
  op[(size_t)(g+8)*NDIM + n + 1]   = c3;
#elif RES
  out32[(size_t)g*NDIM + n]           = res16[(size_t)g*NDIM + n] + c0;
  out32[(size_t)g*NDIM + n + 1]       = res16[(size_t)g*NDIM + n + 1] + c1;
  out32[(size_t)(g+8)*NDIM + n]       = res16[(size_t)(g+8)*NDIM + n] + c2;
  out32[(size_t)(g+8)*NDIM + n + 1]   = res16[(size_t)(g+8)*NDIM + n + 1] + c3;
#elif FFN
  const __half hg0 = (__half)cG0, hg1 = (__half)cG1, hg2 = (__half)cG2, hg3 = (__half)cG3;
  const __half hgn0 = hsilu_h(hg0), hgn1 = hsilu_h(hg1), hgn2 = hsilu_h(hg2), hgn3 = hsilu_h(hg3);
  *(__half2*)(out16 + (size_t)g*NDIM + n)     = __halves2half2(__hmul(hgn0, (__half)cU0), __hmul(hgn1, (__half)cU1));
  *(__half2*)(out16 + (size_t)(g+8)*NDIM + n) = __halves2half2(__hmul(hgn2, (__half)cU2), __hmul(hgn3, (__half)cU3));
#else
  *(__half2*)(out16 + (size_t)g*NDIM + n)     = __halves2half2((__half)c0, (__half)c1);
  *(__half2*)(out16 + (size_t)(g+8)*NDIM + n) = __halves2half2((__half)c2, (__half)c3);
#endif
}

#endif  // !DBUF

// ================= P4 Stage 1a: DBUF software-pipelined ring =================
// The raw quant loads (q words + the lane-quarter sw word + d + the x tile)
// for chunk s+1 are issued BEFORE the mma sequence of chunk s, so the global
// latency hides under tensor-pipe work; the decode ALU + smem stores run in
// the post-mma bubble. IQ3-packed (QCLASS 1), KCH 128, HMMA, DBG 0 only.
// Ring = two NAMED register sets (E/O) driven by explicit loop bodies -- no
// runtime-indexed locals (q[cc]/x[t]/w8[j] are fully-unrolled constants).
#if DBUF
#if QCLASS != 1
#error "DBUF v1: QCLASS 1 only"
#endif
#if KCH != 128
#error "DBUF v1: KCH 128 only"
#endif
#if DBG != 0
#error "DBUF v1: DBG 0 only"
#endif
#if !HMMA
#error "DBUF v1: HMMA only"
#endif
#if KS
#error "DBUF v1: no KS"
#endif
#define NCH (KDIM / KCH)
#define XTPR (16*KCH/(NTHR*4))

__device__ __forceinline__ void dq_iq3_r(const float* gridf, const unsigned int qv,
    const unsigned int swv, const unsigned int dv, const int cc, __half* w8) {
  const float d = __half2float(__ushort_as_half((unsigned short)dv));
  const float db = d * (((float)(swv >> 28)) + 0.5f) * 0.5f;
  const unsigned int sidx = (swv >> (7u * (unsigned int)(cc & 3))) & 0x7Fu;
  const unsigned int spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1u;
  const float4 g0 = *((const float4*)(gridf + ((qv & 0xFFu) << 2)));
  const float4 g1 = *((const float4*)(gridf + ((qv >> 8) << 2)));
  const float sg0 = ((sidx>>0)&1) ? -1.f : 1.f, sg1 = ((sidx>>1)&1) ? -1.f : 1.f;
  const float sg2 = ((sidx>>2)&1) ? -1.f : 1.f, sg3 = ((sidx>>3)&1) ? -1.f : 1.f;
  const float sg4 = ((sidx>>4)&1) ? -1.f : 1.f, sg5 = ((sidx>>5)&1) ? -1.f : 1.f;
  const float sg6 = ((sidx>>6)&1) ? -1.f : 1.f, sg7 = spar ? -1.f : 1.f;
  w8[0] = __float2half(db*g0.x*sg0); w8[1] = __float2half(db*g0.y*sg1);
  w8[2] = __float2half(db*g0.z*sg2); w8[3] = __float2half(db*g0.w*sg3);
  w8[4] = __float2half(db*g1.x*sg4); w8[5] = __float2half(db*g1.y*sg5);
  w8[6] = __float2half(db*g1.z*sg6); w8[7] = __float2half(db*g1.w*sg7);
}

#define PREFR(P, R, KC) do { \
  const int b_ = (KC) >> 8; \
  const unsigned short* qsp_ = (const unsigned short*)(rowp##P##_); \
  const unsigned int* scp_ = (const unsigned int*)(rowp##P##_ + 64*(KDIM >> 8)); \
  const unsigned short* dpp_ = (const unsigned short*)(rowp##P##_ + 96*(KDIM >> 8)); \
  const int lc0_ = ((KC) & 255) >> 3; \
  _Pragma("unroll") \
  for (int cc = 0; cc < NCL; ++cc) q##P##R[cc] = (unsigned int)qsp_[32*b_ + lc0_ + qc_*NCL + cc]; \
  sw##P##R = scp_[8*b_ + ((lc0_ + qc_*NCL) >> 2)]; \
  d##P##R = (unsigned int)dpp_[b_]; \
} while (0)

#define XPREFR(R, KC) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR)*4; \
    const int m = lin / KCH, kk = lin % KCH; \
    x##R[t] = *(const uint2*)(x16 + (size_t)m*KDIM + (size_t)(KC) + kk); \
  } \
} while (0)

#define COMMWR(P, R, WSB) do { \
  __half* wsP_ = (WSB); \
  _Pragma("unroll") \
  for (int cc = 0; cc < NCL; ++cc) { \
    __half w8[8]; \
    dq_iq3_r(gridf, q##P##R[cc], sw##P##R, d##P##R, cc, w8); \
    const int kk = (qc_*NCL + cc)*8; \
    _Pragma("unroll") \
    for (int j = 0; j < 8; ++j) wsP_[(size_t)(warp*RPT + r_)*WS_LD + kk + j] = w8[j]; \
  } \
} while (0)

#define XCOMR(R) do { \
  _Pragma("unroll") \
  for (int t = 0; t < XTPR; ++t) { \
    const int lin = (tid + t*NTHR)*4; \
    const int m = lin / KCH, kk = lin % KCH; \
    *(uint2*)(xs + (size_t)m*XS_LD + kk) = x##R[t]; \
  } \
} while (0)

#define MMAR_PLAIN() do { \
  _Pragma("unroll") \
  for (int s = 0; s < KSTEPS; ++s) { \
    const int kb = s*16; \
    const int g = lane >> 2, tp = (lane & 3) * 2; \
    const unsigned a0 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp); \
    const unsigned a1 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp); \
    const unsigned a2 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp + 8); \
    const unsigned a3 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp + 8); \
    const __half* wr = ws + (size_t)(warp*8 + g)*WS_LD + kb + tp; \
    const unsigned b0 = *(const unsigned*)(wr); \
    const unsigned b1 = *(const unsigned*)(wr + 8); \
    hmma16816(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1); \
  } \
} while (0)

#define MMAR_FFN() do { \
  _Pragma("unroll") \
  for (int s = 0; s < KSTEPS; ++s) { \
    const int kb = s*16; \
    const int g = lane >> 2, tp = (lane & 3) * 2; \
    const unsigned a0 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp); \
    const unsigned a1 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp); \
    const unsigned a2 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp + 8); \
    const unsigned a3 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp + 8); \
    { \
      const __half* wr = ws + (size_t)(warp*8 + g)*WS_LD + kb + tp; \
      const unsigned b0 = *(const unsigned*)(wr); \
      const unsigned b1 = *(const unsigned*)(wr + 8); \
      hmma16816(cG0, cG1, cG2, cG3, a0, a1, a2, a3, b0, b1); \
    } \
    { \
      const __half* wr = ws + NTILE*WS_LD + (size_t)(warp*8 + g)*WS_LD + kb + tp; \
      const unsigned b0 = *(const unsigned*)(wr); \
      const unsigned b1 = *(const unsigned*)(wr + 8); \
      hmma16816(cU0, cU1, cU2, cU3, a0, a1, a2, a3, b0, b1); \
    } \
  } \
} while (0)

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
#if FFN
    const unsigned char* __restrict__ w2,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
#if RES
    const float* __restrict__ res16, float* __restrict__ out32)
#else
    __half* __restrict__ out16)
#endif
{
#if FFN
  __shared__ __align__(16) __half sm[16 * XS_LD + 2 * NTILE * WS_LD];
#else
  __shared__ __align__(16) __half sm[16 * XS_LD + NTILE * WS_LD];
#endif
  __half* xs = sm;
  __half* ws = sm + 16 * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int n0 = blockIdx.x * NTILE;
  const int r_ = lane >> 2, qc_ = lane & 3;
  const unsigned char* rowpG_ = w1 + (size_t)(n0 + warp*RPT + r_) * ROWBYTES;
#if FFN
  const unsigned char* rowpU_ = w2 + (size_t)(n0 + warp*RPT + r_) * ROWBYTES;
  float cG0 = 0.f, cG1 = 0.f, cG2 = 0.f, cG3 = 0.f;
  float cU0 = 0.f, cU1 = 0.f, cU2 = 0.f, cU3 = 0.f;
#else
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#endif
  unsigned int qGE[NCL], qGO[NCL]; unsigned int swGE, swGO, dGE, dGO;
  uint2 xE[XTPR], xO[XTPR];
#if FFN
  unsigned int qUE[NCL], qUO[NCL]; unsigned int swUE, swUO, dUE, dUO;
#endif

  PREFR(G, E, 0);
#if FFN
  PREFR(U, E, 0);
#endif
  XPREFR(E, 0);
  COMMWR(G, E, ws);
#if FFN
  COMMWR(U, E, ws + NTILE*WS_LD);
#endif
  XCOMR(E);
  __syncthreads();
  _Pragma("unroll 1")
  for (int s = 0; s + 1 < NCH; s += 2) {
    PREFR(G, O, (s+1)*KCH);
#if FFN
    PREFR(U, O, (s+1)*KCH);
#endif
    XPREFR(O, (s+1)*KCH);
#if FFN
    MMAR_FFN();
#else
    MMAR_PLAIN();
#endif
    __syncthreads();
    COMMWR(G, O, ws);
#if FFN
    COMMWR(U, O, ws + NTILE*WS_LD);
#endif
    XCOMR(O);
    __syncthreads();
    if (s + 2 >= NCH) break;
    PREFR(G, E, (s+2)*KCH);
#if FFN
    PREFR(U, E, (s+2)*KCH);
#endif
    XPREFR(E, (s+2)*KCH);
#if FFN
    MMAR_FFN();
#else
    MMAR_PLAIN();
#endif
    __syncthreads();
    COMMWR(G, E, ws);
#if FFN
    COMMWR(U, E, ws + NTILE*WS_LD);
#endif
    XCOMR(E);
    __syncthreads();
  }
#if FFN
  MMAR_FFN();
#else
  MMAR_PLAIN();
#endif
  const int g = lane >> 2, tp = (lane & 3) * 2;
  const int n = n0 + warp*8 + tp;
#if RES
  out32[(size_t)g*NDIM + n]           = res16[(size_t)g*NDIM + n] + c0;
  out32[(size_t)g*NDIM + n + 1]       = res16[(size_t)g*NDIM + n + 1] + c1;
  out32[(size_t)(g+8)*NDIM + n]       = res16[(size_t)(g+8)*NDIM + n] + c2;
  out32[(size_t)(g+8)*NDIM + n + 1]   = res16[(size_t)(g+8)*NDIM + n + 1] + c3;
#elif FFN
  const __half hg0 = (__half)cG0, hg1 = (__half)cG1, hg2 = (__half)cG2, hg3 = (__half)cG3;
  const __half hgn0 = hsilu_h(hg0), hgn1 = hsilu_h(hg1), hgn2 = hsilu_h(hg2), hgn3 = hsilu_h(hg3);
  *(__half2*)(out16 + (size_t)g*NDIM + n)     = __halves2half2(__hmul(hgn0, (__half)cU0), __hmul(hgn1, (__half)cU1));
  *(__half2*)(out16 + (size_t)(g+8)*NDIM + n) = __halves2half2(__hmul(hgn2, (__half)cU2), __hmul(hgn3, (__half)cU3));
#else
  *(__half2*)(out16 + (size_t)g*NDIM + n)     = __halves2half2((__half)c0, (__half)c1);
  *(__half2*)(out16 + (size_t)(g+8)*NDIM + n) = __halves2half2((__half)c2, (__half)c3);
#endif
}
#endif  // DBUF

// ================= P5: SYNCW + SUB + H2 FFN/IQ3 restructure =================
// SYNCW=1: xs double-buffered (+ register prefetch of the next x tile) and the
// W stage is WARP-LOCAL (each warp stages and mma's only its own rows) -> ONE
// full __syncthreads per k-chunk (the classic path needs two). SUB=n: each
// warp owns SUB sub-tiles of 8 out-rows (NTILE = NWARP*8*SUB) -> fewer, fatter
// CTAs (1-2 waves at the dext's hard 1 CTA/SM) and the a-frag smem reads
// amortize xSUB x planes. Decode math and the per-row mma k-order are VERBATIM
// the classic path -> SYNCW/SUB builds are BIT-IDENTICAL to P1/P4.
// H2=1 (QCLASS 1 only): IQ3 decode via two small LUTs in `lut` (appended LAST
// kernel arg, args-after-last-up law): [0..2048) = the 256-entry grid as
// half2 pairs (row qv = 4 halfs = 8B), [2048..4096) = 128 sign-XOR masks
// (uint4 per 7-bit sidx; element-7 mask bit = parity(sidx)). w = HMUL2(db2,
// grid_h2) XOR signmask -- 4 HMUL2 + 4 XOR + 2 STS.64 per 8 weights vs the
// classic 16 FMUL + 8 F2H + 8 STS.16 + a 7-XOR parity chain. Rounding class:
// round(db) then (exact dyadic-scaled) product vs the classic round(db*g) --
// <=1 half-ulp on a fraction of elements (F-norm-validated, NOT bit-identical).
#if SYNCW
#if !HMMA
#error "SYNCW: HMMA only"
#endif
#if KS || DBUF
#error "SYNCW: no KS/DBUF combo"
#endif
#if DBG != 0
#error "SYNCW: DBG 0 only"
#endif
#if H2 && QCLASS != 1
#error "H2: QCLASS 1 only"
#endif
#if RES && FFN
#error "RES+FFN not supported"
#endif
#if !SUB
#define SUB 1
#endif
#ifndef H2
#define H2 0
#endif
#if NTILE != (NTHR / 32) * 8 * SUB
#error "SYNCW: NTILE must be NWARP*8*SUB"
#endif
#define XTPR2 ((16 * KCH + NTHR * 4 - 1) / (NTHR * 4))

// classic decode (verbatim dq_chunk) staged to wsr+kk
__device__ __forceinline__ void dq1_cl(const unsigned char* rowp, const float* gridf,
                                       int kc, int lc, __half* wsr, int kk) {
  __half w8[8];
  dq_chunk(rowp, gridf, kc, lc, w8);
  _Pragma("unroll")
  for (int j = 0; j < 8; ++j) wsr[kk + j] = w8[j];
}

__device__ __forceinline__ void dq1(const unsigned char* rowp,
#if H2
                                    const unsigned char* lut,
#endif
                                    const float* gridf, int kc, int lc, __half* wsr, int kk) {
#if H2
  dq1_h2(rowp, lut, kc, lc, wsr, kk);
#else
  dq1_cl(rowp, gridf, kc, lc, wsr, kk);
#endif
}

// stage this warp's SUB rows (8k lane-chunks lc0..) of ONE weight plane
__device__ __forceinline__ void stage_plane(const unsigned char* w, __half* wsb,
#if H2
                                            const unsigned char* lut,
#endif
                                            const float* gridf,
                                            int kc, int n0, int lane, int warp, int r_, int qc_) {
  const int lc0_ = (kc & 255) >> 3;
  _Pragma("unroll")
  for (int ss = 0; ss < SUB; ++ss) {
    const int row_ = n0 + warp*SUB*8 + ss*8 + r_;
    const unsigned char* rowp = w + (size_t)row_ * ROWBYTES;
    __half* wsr = wsb + (size_t)(warp*SUB*8 + ss*8 + r_) * WS_LD;
    _Pragma("unroll")
    for (int cc = 0; cc < NCL; ++cc) {
      const int lc = lc0_ + qc_*NCL + cc;
      const int kk = (qc_*NCL + cc) * 8;
      dq1(rowp,
#if H2
          lut,
#endif
          gridf, kc, lc, wsr, kk);
    }
  }
}

extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
#if FFN
    const unsigned char* __restrict__ w2,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
#if RES
    const float* __restrict__ res16, float* __restrict__ out32
#else
    __half* __restrict__ out16
#endif
#if H2
    , const unsigned char* __restrict__ lut
#endif
)
{
#if FFN
  __shared__ __align__(16) __half sm[32 * XS_LD + 2 * NTILE * WS_LD];
  float acc[2][SUB][4];
#else
  __shared__ __align__(16) __half sm[32 * XS_LD + NTILE * WS_LD];
  float acc[1][SUB][4];
#endif
#define ACC(pl, ss, j) acc[pl][ss][j]
  __half* xsA = sm;
  __half* xsB = sm + 16 * XS_LD;
  __half* ws = sm + 32 * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int n0 = blockIdx.x * NTILE;
  const int r_ = lane >> 2, qc_ = lane & 3;
#if FFN
  _Pragma("unroll")
  for (int ss = 0; ss < SUB; ++ss)
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) { ACC(0, ss, j) = 0.f; ACC(1, ss, j) = 0.f; }
#else
  _Pragma("unroll")
  for (int ss = 0; ss < SUB; ++ss)
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) ACC(0, ss, j) = 0.f;
#endif
  uint2 xE[XTPR2];

  // prologue: prefetch x(0) to regs, stage ws(0), commit xsA
  _Pragma("unroll")
  for (int t = 0; t < XTPR2; ++t) {
    const int lin = (tid + t*NTHR) * 4;
    if (lin < 16*KCH) {
      const int m = lin / KCH, kk = lin % KCH;
      xE[t] = *(const uint2*)(x16 + (size_t)m*KDIM + 0 + kk);
    }
  }
  stage_plane(w1, ws,
#if H2
              lut,
#endif
              gridf, 0, n0, lane, warp, r_, qc_);
#if FFN
  stage_plane(w2, ws + NTILE*WS_LD,
#if H2
              lut,
# endif
              gridf, 0, n0, lane, warp, r_, qc_);
#endif
  _Pragma("unroll")
  for (int t = 0; t < XTPR2; ++t) {
    const int lin = (tid + t*NTHR) * 4;
    if (lin < 16*KCH) {
      const int m = lin / KCH, kk = lin % KCH;
      *(uint2*)(xsA + (size_t)m*XS_LD + kk) = xE[t];
    }
  }
  __syncthreads();

  _Pragma("unroll 1")
  for (int kc = 0; ; kc += KCH) {
    __half* xs = ((kc / KCH) & 1) ? xsB : xsA;
    // mma phase (b-frags from this warp's own staged rows; k ascending)
    {
      const int g = lane >> 2, tp = (lane & 3) * 2;
      _Pragma("unroll")
      for (int s = 0; s < KSTEPS; ++s) {
        const int kb = s*16;
        const unsigned a0 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp);
        const unsigned a1 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp);
        const unsigned a2 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp + 8);
        const unsigned a3 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp + 8);
        _Pragma("unroll")
        for (int ss = 0; ss < SUB; ++ss) {
          {
            const __half* wr = ws + (size_t)(warp*SUB*8 + ss*8 + g)*WS_LD + kb + tp;
            const unsigned b0 = *(const unsigned*)(wr);
            const unsigned b1 = *(const unsigned*)(wr + 8);
            hmma16816(ACC(0, ss, 0), ACC(0, ss, 1), ACC(0, ss, 2), ACC(0, ss, 3), a0, a1, a2, a3, b0, b1);
          }
#if FFN
          {
            const __half* wr = ws + NTILE*WS_LD + (size_t)(warp*SUB*8 + ss*8 + g)*WS_LD + kb + tp;
            const unsigned b0 = *(const unsigned*)(wr);
            const unsigned b1 = *(const unsigned*)(wr + 8);
            hmma16816(ACC(1, ss, 0), ACC(1, ss, 1), ACC(1, ss, 2), ACC(1, ss, 3), a0, a1, a2, a3, b0, b1);
          }
#endif
        }
      }
    }
    if (kc + KCH >= KDIM) break;
    // prefetch next x tile to regs (global latency hides under the ws stage)
    _Pragma("unroll")
    for (int t = 0; t < XTPR2; ++t) {
      const int lin = (tid + t*NTHR) * 4;
      if (lin < 16*KCH) {
        const int m = lin / KCH, kk = lin % KCH;
        xE[t] = *(const uint2*)(x16 + (size_t)m*KDIM + (size_t)(kc + KCH) + kk);
      }
    }
    __syncwarp(FULL);   // this warp's mma ws-reads done before re-staging its rows
    stage_plane(w1, ws,
#if H2
                lut,
#endif
                gridf, kc + KCH, n0, lane, warp, r_, qc_);
#if FFN
    stage_plane(w2, ws + NTILE*WS_LD,
# if H2
                lut,
# endif
                gridf, kc + KCH, n0, lane, warp, r_, qc_);
#endif
    {
      __half* xnext = ((kc / KCH) & 1) ? xsA : xsB;
      _Pragma("unroll")
      for (int t = 0; t < XTPR2; ++t) {
        const int lin = (tid + t*NTHR) * 4;
        if (lin < 16*KCH) {
          const int m = lin / KCH, kk = lin % KCH;
          *(uint2*)(xnext + (size_t)m*XS_LD + kk) = xE[t];
        }
      }
    }
    __syncthreads();   // xs(next) + all warps' ws(next) visible for next mma
  }

  // epilogue: EXACTLY the classic per-element op order (per sub-tile)
  {
    const int g = lane >> 2, tp = (lane & 3) * 2;
    _Pragma("unroll")
    for (int ss = 0; ss < SUB; ++ss) {
      const int n = n0 + warp*SUB*8 + ss*8 + tp;
#if RES
      out32[(size_t)g*NDIM + n]           = res16[(size_t)g*NDIM + n] + ACC(0, ss, 0);
      out32[(size_t)g*NDIM + n + 1]       = res16[(size_t)g*NDIM + n + 1] + ACC(0, ss, 1);
      out32[(size_t)(g+8)*NDIM + n]       = res16[(size_t)(g+8)*NDIM + n] + ACC(0, ss, 2);
      out32[(size_t)(g+8)*NDIM + n + 1]   = res16[(size_t)(g+8)*NDIM + n + 1] + ACC(0, ss, 3);
#elif FFN
      const __half hg0 = (__half)ACC(0, ss, 0), hg1 = (__half)ACC(0, ss, 1);
      const __half hg2 = (__half)ACC(0, ss, 2), hg3 = (__half)ACC(0, ss, 3);
      const __half hgn0 = hsilu_h(hg0), hgn1 = hsilu_h(hg1), hgn2 = hsilu_h(hg2), hgn3 = hsilu_h(hg3);
      *(__half2*)(out16 + (size_t)g*NDIM + n)     = __halves2half2(__hmul(hgn0, (__half)ACC(1, ss, 0)), __hmul(hgn1, (__half)ACC(1, ss, 1)));
      *(__half2*)(out16 + (size_t)(g+8)*NDIM + n) = __halves2half2(__hmul(hgn2, (__half)ACC(1, ss, 2)), __hmul(hgn3, (__half)ACC(1, ss, 3)));
#else
      *(__half2*)(out16 + (size_t)g*NDIM + n)     = __halves2half2((__half)ACC(0, ss, 0), (__half)ACC(0, ss, 1));
      *(__half2*)(out16 + (size_t)(g+8)*NDIM + n) = __halves2half2((__half)ACC(0, ss, 2), (__half)ACC(0, ss, 3));
#endif
    }
  }
#undef ACC
}
#endif  // SYNCW

// ================= P6: M32 — 32 x-rows per CTA (load-stream amortizer) ========
// THE WALL (P5): at M=16 the family sits at 150-167 GB/s bound by the quant-
// word LOAD STREAM vs the dext's hard 1-CTA/SM — decode/issue/sync/waves all
// falsified. M32 doubles the arithmetic per loaded weight byte: xs holds 32
// rows, each b-fragment (the staged W tile) feeds TWO m16n8k16 mmas (rows
// 0..15 and 16..31). stage_w/decode is VERBATIM the classic path; the per-row
// k-order is unchanged (same fragment map per row) -> per-row outputs are in
// the SAME rounding class (empirically bit-identical to the M=16 kernels on
// identical inputs — validated in test_p6.py).
// smem: 32*XS_LD + (FFN?2:1)*NTILE*WS_LD halfs. nw8k128: plain 26112 B, FFN
// 43520 B (< 48 KB static; the engine launches these EAGERLY, not in graphs).
// Residual/M rows: out16/out32/res16 are [32][NDIM]; x16 is [32][KDIM].
#if M32
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const unsigned char* __restrict__ w1,
#if FFN
    const unsigned char* __restrict__ w2,
#endif
    const float* __restrict__ gridf, const __half* __restrict__ x16,
#if RES || KS
    const float* __restrict__ res16, float* __restrict__ out32
#else
    __half* __restrict__ out16
#endif
    )
{
#if FFN
  __shared__ __align__(16) __half sm[32 * XS_LD + 2 * NTILE * WS_LD];
#else
  __shared__ __align__(16) __half sm[32 * XS_LD + NTILE * WS_LD];
#endif
  __half* xs = sm;
  __half* ws = sm + 32 * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
#if KS
  // P11 G1: K-split-2 -- CTA (tile_, ks_) covers half the K chunks of tile
  // tile_; each half writes EXACT fp32 partials (no epilogue) to out32 +
  // ks_*32*NDIM; the pfg_ks2c combine adds p0+p1 in a fixed order.
  const int tile_ = blockIdx.x / KS, ks_ = blockIdx.x % KS;
  const int n0 = tile_ * NTILE;
  const int kc0_ = ks_ * (KDIM / KS), kc1_ = kc0_ + (KDIM / KS);
#else
  const int n0 = blockIdx.x * NTILE;
  const int kc0_ = 0, kc1_ = KDIM;
#endif
#if FFN
  float cG0 = 0.f, cG1 = 0.f, cG2 = 0.f, cG3 = 0.f;
  float cG4 = 0.f, cG5 = 0.f, cG6 = 0.f, cG7 = 0.f;
  float cU0 = 0.f, cU1 = 0.f, cU2 = 0.f, cU3 = 0.f;
  float cU4 = 0.f, cU5 = 0.f, cU6 = 0.f, cU7 = 0.f;
#else
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  float c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
#endif

  _Pragma("unroll 2")
  for (int kc = kc0_; kc < kc1_; kc += KCH) {
    // stage X: 32 rows (the ONLY structural change on the x side)
    _Pragma("unroll")
    for (int i = 0; i < (32*KCH + NTHR*4 - 1)/(NTHR*4); ++i) {
      const int lin = (tid + i*NTHR) * 4;
      if (lin < 32*KCH) {
        const int m = lin / KCH, kk = lin % KCH;
        *(uint2*)(xs + (size_t)m*XS_LD + kk) = *(const uint2*)(x16 + (size_t)m*KDIM + kc + kk);
      }
    }
    // stage W: VERBATIM the classic path (same ws, same decode stream)
    stage_w(ws, w1, gridf,
#if H2
            lut,
#endif
            kc, tid, lane, warp, n0);
#if FFN
    stage_w(ws + NTILE*WS_LD, w2, gridf,
#if H2
            lut,
#endif
            kc, tid, lane, warp, n0);
#endif
    __syncthreads();
    _Pragma("unroll")
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s*16;
      const int g = lane >> 2, tp = (lane & 3) * 2;
      const unsigned a0 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp);
      const unsigned a1 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp);
      const unsigned a2 = *(const unsigned*)(xs + (size_t)g*XS_LD + kb + tp + 8);
      const unsigned a3 = *(const unsigned*)(xs + (size_t)(g+8)*XS_LD + kb + tp + 8);
      const unsigned a4 = *(const unsigned*)(xs + (size_t)(g+16)*XS_LD + kb + tp);
      const unsigned a5 = *(const unsigned*)(xs + (size_t)(g+24)*XS_LD + kb + tp);
      const unsigned a6 = *(const unsigned*)(xs + (size_t)(g+16)*XS_LD + kb + tp + 8);
      const unsigned a7 = *(const unsigned*)(xs + (size_t)(g+24)*XS_LD + kb + tp + 8);
      {
        const __half* wr = ws + (size_t)(warp*8 + g)*WS_LD + kb + tp;
        const unsigned b0 = *(const unsigned*)(wr);
        const unsigned b1 = *(const unsigned*)(wr + 8);
#if FFN
        hmma16816(cG0, cG1, cG2, cG3, a0, a1, a2, a3, b0, b1);
        hmma16816(cG4, cG5, cG6, cG7, a4, a5, a6, a7, b0, b1);
#else
        hmma16816(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
        hmma16816(c4, c5, c6, c7, a4, a5, a6, a7, b0, b1);
#endif
      }
#if FFN
      {
        const __half* wr = ws + NTILE*WS_LD + (size_t)(warp*8 + g)*WS_LD + kb + tp;
        const unsigned b0 = *(const unsigned*)(wr);
        const unsigned b1 = *(const unsigned*)(wr + 8);
        hmma16816(cU0, cU1, cU2, cU3, a0, a1, a2, a3, b0, b1);
        hmma16816(cU4, cU5, cU6, cU7, a4, a5, a6, a7, b0, b1);
      }
#endif
    }
    __syncthreads();
  }

  // c-frag -> out[m][n] for m in {g, g+8, g+16, g+24} (classic map per group)
  const int g = lane >> 2, tp = (lane & 3) * 2;
  const int n = n0 + warp*8 + tp;
#if KS
  float* op = out32 + (size_t)ks_*32*NDIM;
  op[(size_t)g*NDIM + n]           = c0;
  op[(size_t)g*NDIM + n + 1]       = c1;
  op[(size_t)(g+8)*NDIM + n]       = c2;
  op[(size_t)(g+8)*NDIM + n + 1]   = c3;
  op[(size_t)(g+16)*NDIM + n]      = c4;
  op[(size_t)(g+16)*NDIM + n + 1]  = c5;
  op[(size_t)(g+24)*NDIM + n]      = c6;
  op[(size_t)(g+24)*NDIM + n + 1]  = c7;
#elif RES
  out32[(size_t)g*NDIM + n]           = res16[(size_t)g*NDIM + n] + c0;
  out32[(size_t)g*NDIM + n + 1]       = res16[(size_t)g*NDIM + n + 1] + c1;
  out32[(size_t)(g+8)*NDIM + n]       = res16[(size_t)(g+8)*NDIM + n] + c2;
  out32[(size_t)(g+8)*NDIM + n + 1]   = res16[(size_t)(g+8)*NDIM + n + 1] + c3;
  out32[(size_t)(g+16)*NDIM + n]      = res16[(size_t)(g+16)*NDIM + n] + c4;
  out32[(size_t)(g+16)*NDIM + n + 1]  = res16[(size_t)(g+16)*NDIM + n + 1] + c5;
  out32[(size_t)(g+24)*NDIM + n]      = res16[(size_t)(g+24)*NDIM + n] + c6;
  out32[(size_t)(g+24)*NDIM + n + 1]  = res16[(size_t)(g+24)*NDIM + n + 1] + c7;
#elif FFN
  const __half hg0 = (__half)cG0, hg1 = (__half)cG1, hg2 = (__half)cG2, hg3 = (__half)cG3;
  const __half hg4 = (__half)cG4, hg5 = (__half)cG5, hg6 = (__half)cG6, hg7 = (__half)cG7;
  const __half hgn0 = hsilu_h(hg0), hgn1 = hsilu_h(hg1), hgn2 = hsilu_h(hg2), hgn3 = hsilu_h(hg3);
  const __half hgn4 = hsilu_h(hg4), hgn5 = hsilu_h(hg5), hgn6 = hsilu_h(hg6), hgn7 = hsilu_h(hg7);
  *(__half2*)(out16 + (size_t)g*NDIM + n)      = __halves2half2(__hmul(hgn0, (__half)cU0), __hmul(hgn1, (__half)cU1));
  *(__half2*)(out16 + (size_t)(g+8)*NDIM + n)  = __halves2half2(__hmul(hgn2, (__half)cU2), __hmul(hgn3, (__half)cU3));
  *(__half2*)(out16 + (size_t)(g+16)*NDIM + n) = __halves2half2(__hmul(hgn4, (__half)cU4), __hmul(hgn5, (__half)cU5));
  *(__half2*)(out16 + (size_t)(g+24)*NDIM + n) = __halves2half2(__hmul(hgn6, (__half)cU6), __hmul(hgn7, (__half)cU7));
#else
  *(__half2*)(out16 + (size_t)g*NDIM + n)      = __halves2half2((__half)c0, (__half)c1);
  *(__half2*)(out16 + (size_t)(g+8)*NDIM + n)  = __halves2half2((__half)c2, (__half)c3);
  *(__half2*)(out16 + (size_t)(g+16)*NDIM + n) = __halves2half2((__half)c4, (__half)c5);
  *(__half2*)(out16 + (size_t)(g+24)*NDIM + n) = __halves2half2((__half)c6, (__half)c7);
#endif
}
#endif  // M32


