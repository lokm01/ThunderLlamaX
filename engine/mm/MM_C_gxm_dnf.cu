// MM SESSION C: gxm_dnf -- the grouped expert down GEMM with the ACT
// RESTAGING FOLD (the ~98ms residual item). Session-B gxm_dn restages the
// 16KB act block per (row-block, b) = 128 stages/item/CTA from L2; the fold
// stages the FULL 512-wide act ONCE per item into xsm2[TS][512] (32KB smem
// -> CTASM AUTO 100KB = 3 CTAs/SM; name matches the gxm substring) and the
// rb loop becomes pure dequant+FMA with ZERO interior syncs.
// PER-(PAIR,ROW) MATH VERBATIM (rb asc; b asc; ib asc; running sums;
// 5-xor tree; fp16 store; identical xsm VALUES, only their residency
// moved) -> BIT-EXACT vs gxm_dn by construction (the fold does not touch
// any arithmetic or addressing of the compute phase).
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
extern "C" __global__ void __launch_bounds__(256) gxm_dnf(
    const unsigned long long* __restrict__ ptbl,
    const unsigned int* __restrict__ items,
    const unsigned int* __restrict__ nitp,
    const int* __restrict__ eoff,
    const unsigned short* __restrict__ plist,
    const float* __restrict__ xs,          // [NPAIR][512] moe_act
    const float* __restrict__ iq4nlb,      // [16]
    __half* __restrict__ parts)            // [NPAIR][2048]
{
  __shared__ float xsm2[TS][512];          // THE FOLD: full k=512 act resident
  __shared__ int xoff[TS];                 // pair*512 base per token
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned int nit = *nitp;
  for (unsigned int it = blockIdx.x; it < nit; it += GN) {
    const unsigned int d = items[it];
    const int e = (int)(d >> 20), rs = (int)((d >> 16) & 15u), m0 = (int)(d & 0xFFFu);
    const int b0 = eoff[e], bin = eoff[e+1] - b0;
    if (m0 >= bin) continue;               // uniform per CTA (item-uniform)
    int mend = m0 + TS; if (mend > bin) mend = bin;
    const unsigned char* base = (const unsigned char*)(size_t)ptbl[e];
    const int nby = 8 + (lane & 15);
    __syncthreads();                       // guard: prior item's FMA reads
    for (int m = threadIdx.x; m < mend - m0; m += 256)
      xoff[m] = (int)plist[b0 + m0 + m] * 512;
    __syncthreads();                       // xoff writes -> fold reads (cross-thread)
    // THE FOLD: both b-blocks staged ONCE per item (was: per rb x per b)
    for (int i = threadIdx.x; i < (mend - m0) * 512; i += 256) {
      const int m = i >> 9, c = i & 511;
      xsm2[m][c] = xs[xoff[m] + c];
    }
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
          const unsigned char* blk = base + (size_t)(r0+row0)*272 + b*136;
          const float d = __half2float(*((const __half*)blk));
          const unsigned int sh = *(const unsigned short*)(blk + 2);
          const unsigned int sl = *(const unsigned int*)(blk + 4);
          #pragma unroll
          for (int ib = 0; ib < 8; ++ib) {
            const unsigned int nib = (lane & 16) ? (blk[16*ib + nby] >> 4) : (blk[16*ib + nby] & 0xFu);
            const unsigned int ls = ((sl >> (8*(ib>>1) + 4*(ib&1))) & 0xFu) | (((sh >> (2*ib)) & 3u) << 4);
            const float dl = d * (float)((int)ls - 32);
            wv[ib] = dl * iq4nlb[nib];
          }
        }
        const int nt = mend - m0;
        #pragma unroll
        for (int m = 0; m < TS; ++m) {
          if (m >= nt) break;
          #pragma unroll
          for (int ib = 0; ib < 8; ++ib)
            acc[m] += wv[ib] * xsm2[m][(b<<8) + (ib<<5) + lane];
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
    }
    // no interior sync needed: xoff/xsm2 restaged only after the top guard
  }
}
