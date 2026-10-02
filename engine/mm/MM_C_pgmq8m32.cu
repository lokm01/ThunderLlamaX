// MM SESSION C (L1b): pgmq8m32 -- trunk out/o Q8_0 M-GEMM mma for the
// k=4096 residual pair (THE 65% residual item). The dense pf_gemm3m M=32
// design (two m16n8k16 fragments share every staged W tile) on the MoE
// Q8_0 weights: per k-chunk KCH=128 each lane owns exactly ONE 34B Q8_0
// block (lane>>2 = tile row, lane&3 = 32-wide sub-block). fp32 x staged
// to fp16 xs; w dequant d*(float)q -> fp16 ws; mma f32.f16.f16.f32;
// epilogue y = hres + acc (fp32, same contract as gv8k4096r).
// NUMERICS-CLASS MOVE (Tier-2): fp16-rounded operands + mma reduction
// order vs the stock per-lane fp32 FMA + 5-xor tree -> the F-metric bank
// re-baseline (the dense M32 precedent: 9.4e-4 class); spec==T1 through
// the new path must hold (both decode modes never run this PF-only
// kernel; the PF chunk feeds BOTH identically).
// Grid: 1D FOLDED MxN ((P/32)*(ROWS/64)) -- the M-GRID FOLD law; P is the
// runtime int val. ROWS=2048 KDIM=4096 compile-time (hardcoded-sizes law).
// 26KB static smem -> CTASM AUTO 64KB cfg = 2 CTAs/SM (name pgmq8).
#include <cuda_fp16.h>

#define ROWS 2048
#define KDIM 4096
#define MTILE 32
#define NTILE 64
#define KCH 128
#define NTHR 256
#define NWARP (NTHR / 32)
#define XS_LD (KCH + 8)
#define WS_LD (KCH + 8)
#define NCH (KDIM / KCH)
#define KSTEPS (KCH / 16)
#define MRG (MTILE / 16)
#define RPT (NTILE / NWARP)
#define TGRID (ROWS / NTILE)
#define XTPR ((MTILE * KCH / 2 + NTHR - 1) / NTHR)

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

#define MMAR() do { \
  const int g = lane >> 2, tp = (lane & 3) * 2; \
  _Pragma("unroll") \
  for (int s = 0; s < KSTEPS; ++s) { \
    const int kb = s * 16; \
    _Pragma("unroll") \
    for (int rg = 0; rg < MRG; ++rg) { \
      const unsigned a0 = *(const unsigned*)(xs + (size_t)(rg*16 + g) * XS_LD + kb + tp); \
      const unsigned a1 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8) * XS_LD + kb + tp); \
      const unsigned a2 = *(const unsigned*)(xs + (size_t)(rg*16 + g) * XS_LD + kb + tp + 8); \
      const unsigned a3 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8) * XS_LD + kb + tp + 8); \
      const __half* wr = ws + (size_t)(warp*8 + g) * WS_LD + kb + tp; \
      const unsigned b0 = *(const unsigned*)(wr); \
      const unsigned b1 = *(const unsigned*)(wr + 8); \
      hmma16816(acc[rg][0], acc[rg][1], acc[rg][2], acc[rg][3], a0, a1, a2, a3, b0, b1); \
    } \
  } \
} while (0)

extern "C" __global__ void __launch_bounds__(NTHR) pgmq8m32(
    const unsigned char* __restrict__ w,   // [ROWS][128*34] Q8_0
    const float* __restrict__ x,           // [P][KDIM] fp32
    const float* __restrict__ hres,        // [P][ROWS] fp32
    float* __restrict__ y,                 // [P][ROWS] fp32
    const int p)
{
  __shared__ __align__(16) __half sm[MTILE * XS_LD + NTILE * WS_LD];
  __half* xs = sm;
  __half* ws = sm + MTILE * XS_LD;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int bid = blockIdx.x;
  const int mb = bid / TGRID;
  const int ns = bid - mb * TGRID;
  const int n0 = ns * NTILE;
  float acc[MRG][4];
  _Pragma("unroll")
  for (int rg = 0; rg < MRG; ++rg)
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) acc[rg][j] = 0.f;

  const int r_ = lane >> 2, qc_ = lane & 3;
  _Pragma("unroll 2")
  for (int kc = 0; kc < KDIM; kc += KCH) {
    // xs stage: fp32 pairs -> fp16 pairs (XTPR iters, always full at these shapes)
    _Pragma("unroll")
    for (int i = 0; i < XTPR; ++i) {
      const int lin2 = tid + i * NTHR;
      if (lin2 < MTILE * KCH / 2) {
        const int m = lin2 >> 6, kk = (lin2 & 63) << 1;
        const float2 f2 = *(const float2*)(x + (size_t)(mb * MTILE + m) * KDIM + kc + kk);
        *(__half2*)(xs + (size_t)m * XS_LD + kk) = __float22half2_rn(f2);
      }
    }
    // ws stage: per lane ONE Q8_0 block (row = warp*8+r_, sub-block qc_)
    {
      const unsigned char* blk = w + (size_t)(n0 + warp*8 + r_) * 4352
                                    + (size_t)((kc >> 5) + qc_) * 34;
      const float d = __half2float(*((const __half*)blk));
      __half w8[32];
      _Pragma("unroll")
      for (int j = 0; j < 32; ++j)
        w8[j] = __float2half(d * (float)((signed char)blk[2 + j]));
      _Pragma("unroll")
      for (int h = 0; h < 16; ++h)
        *(__half2*)(ws + (size_t)(warp*8 + r_) * WS_LD + qc_ * 32 + h * 2)
          = __halves2half2(w8[2 * h], w8[2 * h + 1]);
    }
    __syncthreads();
    MMAR();
    __syncthreads();
  }

  const int g = lane >> 2, tp = (lane & 3) * 2;
  _Pragma("unroll")
  for (int rg = 0; rg < MRG; ++rg) {
    const int m0 = mb * MTILE + rg * 16 + g;
    const int n = n0 + warp * 8 + tp;
    float2 hr = *(const float2*)(hres + (size_t)m0 * ROWS + n);
    *(float2*)(y + (size_t)m0 * ROWS + n) = make_float2(hr.x + acc[rg][0], hr.y + acc[rg][1]);
    hr = *(const float2*)(hres + (size_t)(m0 + 8) * ROWS + n);
    *(float2*)(y + (size_t)(m0 + 8) * ROWS + n) = make_float2(hr.x + acc[rg][2], hr.y + acc[rg][3]);
  }
}
