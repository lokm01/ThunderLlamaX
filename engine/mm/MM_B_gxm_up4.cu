// MM SESSION B (L2 v2): gxm_up4 -- GROUPED expert gate+up GEMM, IQ4_XS lane
// (layer 39 only). v2 decode-once scheme (see gxm_up v2): each warp decodes
// its row's 8+8 weights per b ONCE, CTA stages the <=TS tokens' 256-elem x
// block to smem, m-loop = pure FMA (unrolled, masked -- the register-array
// law). Per (pair,row) add order VERBATIM (b asc, ib asc per mat, running
// sums, 5-xor tree, silu) -> bit-exact vs gx8e256up4.
#include <cuda_fp16.h>
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
#define ROWB4 1088
#define GATE_B4 557056
extern "C" __global__ void __launch_bounds__(256) gxm_up4(
    const unsigned long long* __restrict__ ptbl,
    const unsigned int* __restrict__ items,
    const unsigned int* __restrict__ nitp,
    const int* __restrict__ eoff,
    const unsigned short* __restrict__ plist,
    const float* __restrict__ xs,          // [P][2048]
    const float* __restrict__ iq4nlb,      // [16]
    float* __restrict__ ys)                // [NPAIR][512]
{
  __shared__ float xsm[TS][256];
  __shared__ int xoff[TS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned int nit = *nitp;
  for (unsigned int it = blockIdx.x; it < nit; it += GN) {
    const unsigned int d = items[it];
    const int e = (int)(d >> 20), rs = (int)((d >> 16) & 15u), m0 = (int)(d & 0xFFFu);
    const int b0 = eoff[e], bin = eoff[e+1] - b0;
    if (m0 >= bin) continue;
    int mend = m0 + TS; if (mend > bin) mend = bin;
    const unsigned char* gp = (const unsigned char*)(size_t)ptbl[e];
    const unsigned char* upp = gp + GATE_B4;
    const int nby = 8 + (lane & 15);
    for (int m = threadIdx.x; m < mend - m0; m += 256)
      xoff[m] = (int)((plist[b0 + m0 + m] >> 3) * 2048);
    __syncthreads();
    const int row0 = warp;
    for (int rb = rs; rb < NRB; rb += RS) {
      const int r0 = rb * 8;
      float accg[TS], accu[TS];
      #pragma unroll
      for (int m = 0; m < TS; ++m) { accg[m] = 0.f; accu[m] = 0.f; }
      for (int b = 0; b < 8; ++b) {
        float wg[8], wu[8];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          const unsigned char* blk = (mi == 0 ? gp : upp) + (size_t)(r0+row0)*ROWB4 + b*136;
          const float d = __half2float(*((const __half*)blk));
          const unsigned int sh = *(const unsigned short*)(blk + 2);
          const unsigned int sl = *(const unsigned int*)(blk + 4);
          #pragma unroll
          for (int ib = 0; ib < 8; ++ib) {
            const unsigned int nib = (lane & 16) ? (blk[16*ib + nby] >> 4) : (blk[16*ib + nby] & 0xFu);
            const unsigned int ls = ((sl >> (8*(ib>>1) + 4*(ib&1))) & 0xFu) | (((sh >> (2*ib)) & 3u) << 4);
            const float dl = d * (float)((int)ls - 32);
            const float w = dl * iq4nlb[nib];
            if (mi == 0) wg[ib] = w; else wu[ib] = w;
          }
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
          #pragma unroll
          for (int ib = 0; ib < 8; ++ib) {
            const float xv = xsm[m][(ib<<5) + lane];
            accg[m] += wg[ib] * xv;
            accu[m] += wu[ib] * xv;
          }
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
      __syncthreads();
    }
  }
}
