// MM SESSION D (port 2): pgmq8k2 -- the k=2048 trunk Q8_0 M-GEMM mma for
// the qkv/z/q/k/v seat-loop classes (the 188ms/36%@2k trunk_g residual).
// The Session-C pgmq8m32 template (two m16n8k16 fragments share every
// staged W tile) at KDIM=2048: per k-chunk KCH=128 each lane owns ONE 34B
// Q8_0 block. x staged fp32->fp16; w dequant d*(float)q->fp16; mma
// f32.f16.f16.f32; epilogue y[P][ROWS] = acc (NO residual -- unlike the
// out/o pair these projections are plain).
// NUMERICS-CLASS MOVE (Tier-2, same class as pgmq8m32): fp16 operands +
// mma order vs the stock per-lane fp32 FMA trees -> the F-bank/CE gates.
// ROWS compile-time (8192/4096/512 classes); P runtime val. 1D folded
// MxN grid ((P/32)*(ROWS/64)). 26KB static smem.
#include <cuda_fp16.h>

#ifndef ROWS
#define ROWS 8192
#endif
#define KDIM 2048
#define MTILE 32
#define NTILE 64
#define KCH 128
#define NTHR 256
#define XS_LD (KCH + 8)
#define WS_LD (KCH + 8)
#define NCH (KDIM / KCH)
#define KSTEPS (KCH / 16)
#define MRG (MTILE / 16)
#define TGRID (ROWS / NTILE)
#define XTPR ((MTILE * KCH / 2 + NTHR - 1) / NTHR)
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(pgmq8k2_r, ROWS)

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

extern "C" __global__ void __launch_bounds__(NTHR) KSYM(
    const unsigned char* __restrict__ w,   // [ROWS][64*34] Q8_0
    const float* __restrict__ x,           // [P][KDIM] fp32
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
  #pragma unroll
  for (int rg = 0; rg < MRG; ++rg)
    #pragma unroll
    for (int j = 0; j < 4; ++j) acc[rg][j] = 0.f;

  const int r_ = lane >> 2, qc_ = lane & 3;
  #pragma unroll 2
  for (int kc = 0; kc < KDIM; kc += KCH) {
    // xs stage: fp32 pairs -> fp16 pairs (always full at these shapes)
    #pragma unroll
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
      const unsigned char* blk = w + (size_t)(n0 + warp*8 + r_) * 2176
                                    + (size_t)((kc >> 5) + qc_) * 34;
      const float d = __half2float(*((const __half*)blk));
      __half w8[32];
      #pragma unroll
      for (int j = 0; j < 32; ++j)
        w8[j] = __float2half(d * (float)((signed char)blk[2 + j]));
      #pragma unroll
      for (int h = 0; h < 16; ++h)
        *(__half2*)(ws + (size_t)(warp*8 + r_) * WS_LD + qc_ * 32 + h * 2)
          = __halves2half2(w8[2 * h], w8[2 * h + 1]);
    }
    __syncthreads();
    #pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s * 16;
      const int g = lane >> 2, tp = (lane & 3) * 2;
      const __half* wr = ws + (size_t)(warp*8 + g) * WS_LD + kb + tp;
      const unsigned b0 = *(const unsigned*)(wr);
      const unsigned b1 = *(const unsigned*)(wr + 8);
      #pragma unroll
      for (int rg = 0; rg < MRG; ++rg) {
        const unsigned a0 = *(const unsigned*)(xs + (size_t)(rg*16 + g) * XS_LD + kb + tp);
        const unsigned a1 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8) * XS_LD + kb + tp);
        const unsigned a2 = *(const unsigned*)(xs + (size_t)(rg*16 + g) * XS_LD + kb + tp + 8);
        const unsigned a3 = *(const unsigned*)(xs + (size_t)(rg*16 + g + 8) * XS_LD + kb + tp + 8);
        hmma16816(acc[rg][0], acc[rg][1], acc[rg][2], acc[rg][3], a0, a1, a2, a3, b0, b1);
      }
    }
    __syncthreads();
  }

  const int g = lane >> 2, tp = (lane & 3) * 2;
  #pragma unroll
  for (int rg = 0; rg < MRG; ++rg) {
    const int m0 = mb * MTILE + rg * 16 + g;
    const int n = n0 + warp * 8 + tp;
    *(float2*)(y + (size_t)m0 * ROWS + n) = make_float2(acc[rg][0], acc[rg][1]);
    *(float2*)(y + (size_t)(m0 + 8) * ROWS + n) = make_float2(acc[rg][2], acc[rg][3]);
  }
}
