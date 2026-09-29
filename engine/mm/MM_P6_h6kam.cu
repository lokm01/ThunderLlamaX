// MM P6: h6kam -- the fused head GEMV + PER-POSITION ARGMAX. Row math VERBATIM
// h6k2048 (warp/row, 8 blocks, lane elems {b*256+hh*128+l+{0,32,64,96}}, b/hh
// asc, (d*sc)*q, unsigned-subtract trap, xor tree) but each warp accumulates
// PP x-vectors (x [PP][2048]) against the ONE row read -> per-row logits for
// all probe positions at 1x HEAD traffic. Per-CTA smem reduce over 32 rows ->
// packed u64 partials PART[PP][7760]; amred36 finishes. Pack: monotone u32 of
// the logit << 18 | (262143 - row) = tie-break-lower-row, np.argmax semantics.
// -fmad=false. PP via -DPP (3 or 9).
#include <cuda_fp16.h>
#ifndef PP
#define PP 3
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(h6kam_, PP)
#define VOCAB 248320
#define NCTA 7760   // VOCAB/32
extern "C" __global__ void __launch_bounds__(1024) KSYM(
    const unsigned char* __restrict__ w,   // [VOCAB][1680] Q6_K
    const float* __restrict__ x,           // [PP][2048]
    unsigned long long* __restrict__ part) // [PP][NCTA]
{
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x << 5) + warp;
  __shared__ __align__(16) unsigned long long wbest[32*PP];
  unsigned long long mine[PP];
  #pragma unroll
  for (int p = 0; p < PP; ++p) mine[p] = 0ull;   // logit > -1e30 always beats 0-pack? NO: see init below
  if (row < VOCAB) {
    const unsigned char* rowp = w + (size_t)row*1680;
    const int is_ = lane >> 4;
    float a[PP];
    #pragma unroll
    for (int p = 0; p < PP; ++p) a[p] = 0.f;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      const unsigned char* blk = rowp + b*210;
      const float d = __half2float(*((const __half*)(blk + 208)));
      const signed char* sc = (const signed char*)(blk + 192);
      #pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int qb = 64*hh + lane, hb = 128 + 32*hh + lane, sb = 8*hh + is_;
        const unsigned char qhb = blk[hb];
        const unsigned int ql0 = blk[qb], ql32 = blk[qb + 32];
        float wv0, wv1, wv2, wv3, t;
        t = d * (float)sc[sb + 0];
        wv0 = t * (float)((int)((ql0 & 0xFu) | ((qhb & 3u) << 4)) - 32);
        t = d * (float)sc[sb + 2];
        wv1 = t * (float)((int)((ql32 & 0xFu) | (((qhb >> 2) & 3u) << 4)) - 32);
        t = d * (float)sc[sb + 4];
        wv2 = t * (float)((int)((ql0 >> 4) | (((qhb >> 4) & 3u) << 4)) - 32);
        t = d * (float)sc[sb + 6];
        wv3 = t * (float)((int)((ql32 >> 4) | (((qhb >> 6) & 3u) << 4)) - 32);
        #pragma unroll
        for (int p = 0; p < PP; ++p) {
          const float* xp = x + (size_t)p*2048;
          const float x0 = xp[(b<<8) + (hh<<7) + lane + 0];
          const float x1 = xp[(b<<8) + (hh<<7) + lane + 32];
          const float x2 = xp[(b<<8) + (hh<<7) + lane + 64];
          const float x3 = xp[(b<<8) + (hh<<7) + lane + 96];
          a[p] += wv0 * x0;
          a[p] += wv1 * x1;
          a[p] += wv2 * x2;
          a[p] += wv3 * x3;
        }
      }
    }
    #pragma unroll
    for (int p = 0; p < PP; ++p) {
      float v = a[p];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) {
        unsigned int u = __float_as_uint(v);
        u ^= ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
        mine[p] = ((unsigned long long)u << 18) | (262143ull - (unsigned)row);
      }
    }
  }
  // warps with row >= VOCAB (none at 248320 = 7760*32 exactly) keep mine=0
  if (lane == 0) {
    #pragma unroll
    for (int p = 0; p < PP; ++p) wbest[p*32 + warp] = mine[p];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int p = 0; p < PP; ++p) {
      unsigned long long b_ = wbest[p*32 + 0];
      #pragma unroll
      for (int wr = 1; wr < 32; ++wr) { const unsigned long long c = wbest[p*32 + wr]; if (c > b_) b_ = c; }
      part[(size_t)p*NCTA + blockIdx.x] = b_;
    }
  }
}
