// MM SESSION B (L2 v2): gxm_up -- GROUPED expert gate+up GEMM (IQ3_S lane).
// v1 (stage W bytes to smem, dequant per token) measured 0.86x: the
// pair-walk is LATENCY-bound on the dequant+dot chain, NOT DRAM-bound
// (stock out/o already streams at ~750 GB/s) -- traffic reduction alone
// buys nothing. v2 = the L1-winner trick on experts: each warp DECODES its
// row's 8+8 weights ONCE per (row, b) into registers (TS-times less
// dequant), the CTA stages the <=TS tokens' x block to smem (16KB), and
// the m-loop is pure FMA against xsm. Per (pair,row) the add order stays
// VERBATIM (b asc, per-lane j asc, running sums per mat, 5-xor tree,
// silu) -> bit-exact vs gx8e256up (G2-proven class).
// CTA = 256 thr = 8 warps = 8 rows; item = (expert, row-stripe, token-
// chunk) grid-stride (GN baked; gridDim.x reads 0 on this dext).
#ifndef TS
#define TS 16
#endif
#ifndef RS
#define RS 4
#endif
#ifndef GN
#define GN 1024
#endif
#define NRB 64          // 512 rows / 8-row sweeps
#include <cuda_fp16.h>
#define ROWB 880
#define GATE_B 450560
extern "C" __global__ void __launch_bounds__(256) gxm_up(
    const unsigned long long* __restrict__ ptbl,
    const unsigned int* __restrict__ items,
    const unsigned int* __restrict__ nitp,
    const int* __restrict__ eoff,
    const unsigned short* __restrict__ plist,
    const float* __restrict__ xs,          // [P][2048]
    const float* __restrict__ gridf,
    float* __restrict__ ys)                // [NPAIR][512]
{
  __shared__ float xsm[TS][256];
  __shared__ int xoff[TS];                 // (pair>>3)*2048 per token
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned int nit = *nitp;
  for (unsigned int it = blockIdx.x; it < nit; it += GN) {
    const unsigned int d = items[it];
    const int e = (int)(d >> 20), rs = (int)((d >> 16) & 15u), m0 = (int)(d & 0xFFFu);
    const int b0 = eoff[e], bin = eoff[e+1] - b0;
    if (m0 >= bin) continue;               // uniform per CTA (item-uniform)
    int mend = m0 + TS; if (mend > bin) mend = bin;
    const unsigned char* base = (const unsigned char*)(size_t)ptbl[e];
    const unsigned char* gp = base;
    const unsigned char* upp = base + GATE_B;
    // token x-row offsets staged once per item
    for (int m = threadIdx.x; m < mend - m0; m += 256)
      xoff[m] = (int)((plist[b0 + m0 + m] >> 3) * 2048);
    __syncthreads();
    const int row0 = warp;                 // row within the 8-row sweep
    for (int rb = rs; rb < NRB; rb += RS) {
      const int r0 = rb * 8;
      float accg[TS], accu[TS];
      #pragma unroll
      for (int m = 0; m < TS; ++m) { accg[m] = 0.f; accu[m] = 0.f; }
      for (int b = 0; b < 8; ++b) {
        // decode this row's gate + up weights for block b ONCE
        float wg[8], wu[8];
        {
          const unsigned char* blk = gp + (size_t)(r0+row0)*ROWB + b*110;
          const float d = __half2float(*((const __half*)blk));
          const int g0i = lane*2, g1i = lane*2 + 1;
          const int sraw = lane >> 2;
          const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
          const unsigned int sg = blk[74 + lane];
          const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lane);
          const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0i>>3)] >> (g0i&7)) & 1u) << 8));
          const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1i>>3)] >> (g1i&7)) & 1u) << 8));
          const float* gr0 = gridf + (size_t)qb0*4;
          const float* gr1 = gridf + (size_t)qb1*4;
          float t0 = d*sc*gr0[0]; if (sg & 0x01u) t0 = -t0; wg[0] = t0;
          float t1 = d*sc*gr0[1]; if (sg & 0x02u) t1 = -t1; wg[1] = t1;
          float t2 = d*sc*gr0[2]; if (sg & 0x04u) t2 = -t2; wg[2] = t2;
          float t3 = d*sc*gr0[3]; if (sg & 0x08u) t3 = -t3; wg[3] = t3;
          float t4 = d*sc*gr1[0]; if (sg & 0x10u) t4 = -t4; wg[4] = t4;
          float t5 = d*sc*gr1[1]; if (sg & 0x20u) t5 = -t5; wg[5] = t5;
          float t6 = d*sc*gr1[2]; if (sg & 0x40u) t6 = -t6; wg[6] = t6;
          float t7 = d*sc*gr1[3]; if (sg & 0x80u) t7 = -t7; wg[7] = t7;
        }
        {
          const unsigned char* blk = upp + (size_t)(r0+row0)*ROWB + b*110;
          const float d = __half2float(*((const __half*)blk));
          const int g0i = lane*2, g1i = lane*2 + 1;
          const int sraw = lane >> 2;
          const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
          const unsigned int sg = blk[74 + lane];
          const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lane);
          const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0i>>3)] >> (g0i&7)) & 1u) << 8));
          const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1i>>3)] >> (g1i&7)) & 1u) << 8));
          const float* gr0 = gridf + (size_t)qb0*4;
          const float* gr1 = gridf + (size_t)qb1*4;
          float t0 = d*sc*gr0[0]; if (sg & 0x01u) t0 = -t0; wu[0] = t0;
          float t1 = d*sc*gr0[1]; if (sg & 0x02u) t1 = -t1; wu[1] = t1;
          float t2 = d*sc*gr0[2]; if (sg & 0x04u) t2 = -t2; wu[2] = t2;
          float t3 = d*sc*gr0[3]; if (sg & 0x08u) t3 = -t3; wu[3] = t3;
          float t4 = d*sc*gr1[0]; if (sg & 0x10u) t4 = -t4; wu[4] = t4;
          float t5 = d*sc*gr1[1]; if (sg & 0x20u) t5 = -t5; wu[5] = t5;
          float t6 = d*sc*gr1[2]; if (sg & 0x40u) t6 = -t6; wu[6] = t6;
          float t7 = d*sc*gr1[3]; if (sg & 0x80u) t7 = -t7; wu[7] = t7;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < (mend - m0)*256; i += 256) {
          const int m = i >> 8, c = i & 255;
          xsm[m][c] = xs[xoff[m] + (b<<8) + c];
        }
        __syncthreads();
        const int nt = mend - m0;
        #pragma unroll
        for (int m = 0; m < TS; ++m) {
          if (m >= nt) break;
          const float x0 = xsm[m][(lane<<3)+0], x1 = xsm[m][(lane<<3)+1];
          const float x2 = xsm[m][(lane<<3)+2], x3 = xsm[m][(lane<<3)+3];
          const float x4 = xsm[m][(lane<<3)+4], x5 = xsm[m][(lane<<3)+5];
          const float x6 = xsm[m][(lane<<3)+6], x7 = xsm[m][(lane<<3)+7];
          accg[m] += wg[0]*x0; accg[m] += wg[1]*x1; accg[m] += wg[2]*x2; accg[m] += wg[3]*x3;
          accg[m] += wg[4]*x4; accg[m] += wg[5]*x5; accg[m] += wg[6]*x6; accg[m] += wg[7]*x7;
          accu[m] += wu[0]*x0; accu[m] += wu[1]*x1; accu[m] += wu[2]*x2; accu[m] += wu[3]*x3;
          accu[m] += wu[4]*x4; accu[m] += wu[5]*x5; accu[m] += wu[6]*x6; accu[m] += wu[7]*x7;
        }
      }
      const int nt = mend - m0;
      #pragma unroll
      for (int m = 0; m < TS; ++m) {
        if (m >= nt) break;
        float g = accg[m], u = accu[m];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          g += __shfl_xor_sync(0xffffffffu, g, o);
          u += __shfl_xor_sync(0xffffffffu, u, o);
        }
        if (lane == 0) {
          const unsigned short pair = plist[b0 + m0 + m];
          ys[(size_t)pair * 512 + r0 + row0] = (g / (1.0f + __expf(-g))) * u;
        }
      }
      __syncthreads();   // xsm/xoff reuse guard before the next rb
    }
  }
}
