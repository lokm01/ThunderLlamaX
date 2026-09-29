// MM P9: gxk4e256dn -- grouped GEMV, down mat for the MTP layer's Q4_K
// experts (blk.40 routed bank; down Q4_K 2048x512). Structure VERBATIM
// gx8e256dn (one CTA per PAIR; 1024 thr = 32 warps; 2048 rows in 64 sweeps;
// per row dot k=512 = 2 Q4_K blocks of 144B; per-lane partials + xor tree;
// fp16 partial store for cmbz2048).
// Q4_K block (256 elems) = { d f16 | dmin f16 | scales[12] | qs[128] }
// (scales at [4:16], qs at [16:144]). Dequant VERBATIM the validated numpy
// port (mm_mtp_anchor.dq_q4_k_np, verbatim ggml-quants.c):
//   per group jj (0..3, 64 elems): qs bytes [32*jj .. +32)
//     elems jj*64 + 0..31   = LO nibbles of byte i (scale j0 = 2*jj)
//     elems jj*64 + 32..63  = HI nibbles of byte i (scale j1 = 2*jj+1)
//   get_scale_min_k4(j, sc): j<4 -> (sc[j]&63, sc[j+4]&63)
//     else d = (sc[j+4]&0xF) | ((sc[j-4]>>6)<<4); m = (sc[j+4]>>4) | ((sc[j]>>6)<<4)
//   w_lo = d*(s0) * q  - mn*(m0);  w_hi = d*(s1) * q - mn*(m1)
// Lane l covers per block elems {g*64 + l, g*64+32+l} g=0..3. -fmad=false.
#include <cuda_fp16.h>
#define ROWB 288                 // 2 blocks x 144B per Q4_K row (k=512)
extern "C" __global__ void __launch_bounds__(1024) gxk4e256dn(
    const unsigned long long* __restrict__ ptbl,
    const unsigned short* __restrict__ eids,   // [NPAIR]
    const float* __restrict__ xs,              // [NPAIR][512] moe_act
    __half* __restrict__ parts)                // [NPAIR][2048]
{
  const int pair = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[eids[pair]];
  const float* x = xs + (size_t)pair*512;
  __half* y = parts + (size_t)pair*2048;
  for (int r0 = 0; r0 < 2048; r0 += 32) {
    const unsigned char* rowp = base + (size_t)(r0+warp)*ROWB;
    float a = 0.f;
    #pragma unroll
    for (int b = 0; b < 2; ++b) {
      const unsigned char* blk = rowp + b*144;
      const float d = __half2float(*((const __half*)(blk + 0)));
      const float mn = __half2float(*((const __half*)(blk + 2)));
      const unsigned char* sc = blk + 4;
      const unsigned char* qs = blk + 16;
      #pragma unroll
      for (int g = 0; g < 4; ++g) {
        // scale pair (j0 = 2g lo, j1 = 2g+1 hi) -- verbatim _get_scale_min_k4
        int s0, m0, s1, m1;
        { // j0 = 2g
          const int j = 2*g;
          if (j < 4) { s0 = sc[j] & 63; m0 = sc[j+4] & 63; }
          else { s0 = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4);
                 m0 = (sc[j+4] >> 4) | ((sc[j]   >> 6) << 4); }
        }
        { // j1 = 2g+1
          const int j = 2*g + 1;
          if (j < 4) { s1 = sc[j] & 63; m1 = sc[j+4] & 63; }
          else { s1 = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4);
                 m1 = (sc[j+4] >> 4) | ((sc[j]   >> 6) << 4); }
        }
        const float d1 = d * (float)s0, mm1 = mn * (float)m0;
        const float d2 = d * (float)s1, mm2 = mn * (float)m1;
        const unsigned char qb = qs[32*g + lane];
        a += (d1 * (float)(qb & 0xFu) - mm1) * x[(b<<8) + (g<<6) + lane];
        a += (d2 * (float)(qb >> 4)   - mm2) * x[(b<<8) + (g<<6) + 32 + lane];
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) y[r0+warp] = __float2half(a);
  }
}
