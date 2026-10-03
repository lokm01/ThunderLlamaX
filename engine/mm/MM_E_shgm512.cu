// MM SESSION E (item 2): the shared-expert M-GEMM mma pair (the family
// table: shgu32 15.6ms + shdn32 13.4ms per chunk -- the two biggest
// shared-router members; the pgmq8k2 Q8_0 mma template applied).
//   shgm512: sg/su [512][2176] Q8_0 x hnb [256][2048] -> actsh [256][512]
//            with the silu(g)*u epilogue (two acc sets, gate+up tiles).
//   sdm2048: sd [2048][2176] Q8_0 x actsh [256][512] -> shb [256][2048]
//            plain (KDIM=512).
// Tier-2 (fp16 operands + mma order vs the fp32 seat-loop FMA trees) ->
// F-bank + CE gates. Grid (P/32)x(ROWS/64); P runtime (only for the x
// row bound); det x2 (no smem pad hazards -- shapes always full).
#include <cuda_fp16.h>

#define MTILE 32
#define NTILE 64
#define KCH 128
#define NTHR 256
#define XS_LD (KCH + 8)
#define WS_LD (KCH + 8)
#define KSTEPS (KCH / 16)

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

__device__ __forceinline__ void stage_q8_64(
    const unsigned char* __restrict__ w, const int n0,
    const int kc, const int KDIM, __half* __restrict__ ws)
{
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int r_ = lane >> 2, qc_ = lane & 3;
  const unsigned char* blk = w + (size_t)(n0 + warp * 8 + r_) * (size_t)(KDIM / 32 * 34)
                                + (size_t)((kc >> 5) + qc_) * 34;
  const float d = __half2float(*((const __half*)blk));
  __half w8[32];
  #pragma unroll
  for (int j = 0; j < 32; ++j)
    w8[j] = __float2half(d * (float)((signed char)blk[2 + j]));
  #pragma unroll
  for (int h = 0; h < 16; ++h)
    *(__half2*)(ws + (size_t)(warp * 8 + r_) * WS_LD + qc_ * 32 + h * 2)
      = __halves2half2(w8[2 * h], w8[2 * h + 1]);
}

// ---- shgm512: gate+up with silu epilogue ----
extern "C" __global__ void __launch_bounds__(NTHR) shgm512(
    const unsigned char* __restrict__ wg,   // [512][2176] Q8_0
    const unsigned char* __restrict__ wu,   // [512][2176] Q8_0
    const float* __restrict__ x,            // [P][2048] fp32
    float* __restrict__ y,                  // [P][512] fp32
    const int p)
{
  __shared__ __align__(16) __half xs[MTILE * XS_LD];
  __shared__ __align__(16) __half wgs[NTILE * WS_LD];
  __shared__ __align__(16) __half wus[NTILE * WS_LD];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int bid = blockIdx.x;
  const int TGRID = 512 / NTILE;
  const int mb = bid / TGRID, ns = bid - mb * TGRID;
  const int n0 = ns * NTILE;
  float accg[2][4], accu[2][4];
  #pragma unroll
  for (int rg = 0; rg < 2; ++rg)
    #pragma unroll
    for (int j = 0; j < 4; ++j) { accg[rg][j] = 0.f; accu[rg][j] = 0.f; }

  #pragma unroll 2
  for (int kc = 0; kc < 2048; kc += KCH) {
    #pragma unroll
    for (int i = 0; i < (MTILE * KCH / 2 + NTHR - 1) / NTHR; ++i) {
      const int lin2 = tid + i * NTHR;
      if (lin2 < MTILE * KCH / 2) {
        const int m = lin2 >> 6, kk = (lin2 & 63) << 1;
        if (mb * MTILE + m < p) {
          const float2 f2 = *(const float2*)(x + (size_t)(mb * MTILE + m) * 2048 + kc + kk);
          *(__half2*)(xs + (size_t)m * XS_LD + kk) = __float22half2_rn(f2);
        } else {
          *(__half2*)(xs + (size_t)m * XS_LD + kk) = __half2half2((__half)0);
        }
      }
    }
    stage_q8_64(wg, n0, kc, 2048, wgs);
    stage_q8_64(wu, n0, kc, 2048, wus);
    __syncthreads();
    #pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s * 16;
      const int g = lane >> 2, tp = (lane & 3) * 2;
      const __half* br = wgs + (size_t)(warp * 8 + g) * WS_LD + kb + tp;
      const unsigned b0 = *(const unsigned*)(br);
      const unsigned b1 = *(const unsigned*)(br + 8);
      const __half* br2 = wus + (size_t)(warp * 8 + g) * WS_LD + kb + tp;
      const unsigned b0u = *(const unsigned*)(br2);
      const unsigned b1u = *(const unsigned*)(br2 + 8);
      #pragma unroll
      for (int rg = 0; rg < 2; ++rg) {
        const __half* ar = xs + (size_t)(rg * 16 + g) * XS_LD + kb + tp;
        hmma16816(accg[rg][0], accg[rg][1], accg[rg][2], accg[rg][3],
                  *(const unsigned*)(ar), *(const unsigned*)(ar + 8 * XS_LD),
                  *(const unsigned*)(ar + 8), *(const unsigned*)(ar + 8 * XS_LD + 8),
                  b0, b1);
        hmma16816(accu[rg][0], accu[rg][1], accu[rg][2], accu[rg][3],
                  *(const unsigned*)(ar), *(const unsigned*)(ar + 8 * XS_LD),
                  *(const unsigned*)(ar + 8), *(const unsigned*)(ar + 8 * XS_LD + 8),
                  b0u, b1u);
      }
    }
    __syncthreads();
  }
  const int g = lane >> 2, tp = (lane & 3) * 2;
  #pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
    const int m0 = mb * MTILE + rg * 16 + g;
    const int n = n0 + warp * 8 + tp;
    #pragma unroll
    for (int dn = 0; dn < 2; ++dn) {
      const float gf = accg[rg][dn], uf = accu[rg][dn];
      if (m0 < p) y[(size_t)m0 * 512 + n + dn] = (gf / (1.0f + __expf(-gf))) * uf;
      const float gf2 = accg[rg][2 + dn], uf2 = accu[rg][2 + dn];
      if (m0 + 8 < p) y[(size_t)(m0 + 8) * 512 + n + dn] = (gf2 / (1.0f + __expf(-gf2))) * uf2;
    }
  }
}

