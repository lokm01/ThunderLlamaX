// MM SESSION B (L2 v2): gxm_dn6 -- GROUPED expert down GEMM, Q6_K lane
// (layers 34/38/39). v2 decode-once scheme (see gxm_up v2): warp decodes
// its row's 8 weights per b ONCE (h-pairs: ql0/ql32 nibbles x qh bits,
// mult order (d*sc)*q VERBATIM), CTA stages the <=TS tokens' act block to
// smem, m-loop = pure FMA. Per (pair,row) add order VERBATIM (b,h,jj asc
// running sums, 5-xor tree, fp16 store) -> bit-exact vs gx8e256dn6.
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
#define NRB 256         // 2048 rows / 8-row sweeps
extern "C" __global__ void __launch_bounds__(256) gxm_dn6(
    const unsigned long long* __restrict__ ptbl,
    const unsigned int* __restrict__ items,
    const unsigned int* __restrict__ nitp,
    const int* __restrict__ eoff,
    const unsigned short* __restrict__ plist,
    const float* __restrict__ xs,          // [NPAIR][512] moe_act
    __half* __restrict__ parts)            // [NPAIR][2048]
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
    const unsigned char* base = (const unsigned char*)(size_t)ptbl[e];
    const int is = lane >> 4;
    for (int m = threadIdx.x; m < mend - m0; m += 256)
      xoff[m] = (int)plist[b0 + m0 + m] * 512;
    __syncthreads();
    const int row0 = warp;
    for (int rb = rs; rb < NRB; rb += RS) {
      const int r0 = rb * 8;
      float acc[TS];
      #pragma unroll
      for (int m = 0; m < TS; ++m) acc[m] = 0.f;
      for (int b = 0; b < 2; ++b) {
        float wv[8];
        {
          const unsigned char* blk = base + (size_t)(r0+row0)*420 + b*210;
          const float d = __half2float(*((const __half*)(blk + 208)));
          const signed char* sc = (const signed char*)(blk + 192);
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int qb = 64*h + lane, hb = 128 + 32*h + lane, sb = 8*h + is;
            const unsigned char qhb = blk[hb];
            const unsigned int ql0 = blk[qb], ql32 = blk[qb + 32];
            float t = d * (float)sc[sb + 0];
            wv[4*h + 0] = t * (float)((int)((ql0 & 0xFu) | ((qhb & 3u) << 4)) - 32);
            t = d * (float)sc[sb + 2];
            wv[4*h + 1] = t * (float)((int)((ql32 & 0xFu) | (((qhb >> 2) & 3u) << 4)) - 32);
            t = d * (float)sc[sb + 4];
            wv[4*h + 2] = t * (float)((int)((ql0 >> 4) | (((qhb >> 4) & 3u) << 4)) - 32);
            t = d * (float)sc[sb + 6];
            wv[4*h + 3] = t * (float)((int)((ql32 >> 4) | (((qhb >> 6) & 3u) << 4)) - 32);
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
          for (int h = 0; h < 2; ++h) {
            acc[m] += wv[4*h + 0] * xsm[m][(h<<7) + lane + 0];
            acc[m] += wv[4*h + 1] * xsm[m][(h<<7) + lane + 32];
            acc[m] += wv[4*h + 2] * xsm[m][(h<<7) + lane + 64];
            acc[m] += wv[4*h + 3] * xsm[m][(h<<7) + lane + 96];
          }
        }
      }
      const int nt = mend - m0;
      #pragma unroll
      for (int m = 0; m < TS; ++m) {
        if (m >= nt) break;
        float a = acc[m];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) {
          const unsigned short pair = plist[b0 + m0 + m];
          parts[(size_t)pair * 2048 + r0 + row0] = __float2half(a);
        }
      }
      __syncthreads();
    }
  }
}
