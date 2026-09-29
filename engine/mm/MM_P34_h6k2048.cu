// MM P34: h6k2048 -- the Q6_K head GEMV (output.weight [248320 rows][2048 cols]
// Q6_K, 1680B rows = 8 x 210B blocks). One warp per row (32-row sweeps,
// 1024-thread CTAs). Per lane elems {b*256 + hh*128 + l + {0,32,64,96}},
// b asc / hh asc running sums; mult order (d*sc)*q with the int-cast-then-
// minus-32 (the unsigned-subtract trap law); xor tree. y fp32 [rows].
// x fp32 [2048]. -fmad=false.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(1024) h6k2048(
    const unsigned char* __restrict__ w,   // [rows][1680]
    const float* __restrict__ x,           // [2048]
    float* __restrict__ y,                 // [rows]
    const int rows)
{
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x << 5) + warp;
  if (row < rows) {
    const unsigned char* rowp = w + (size_t)row*1680;
    const int is_ = lane >> 4;
    float a = 0.f;
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
        const float x0 = x[(b<<8) + (hh<<7) + lane + 0];
        const float x1 = x[(b<<8) + (hh<<7) + lane + 32];
        const float x2 = x[(b<<8) + (hh<<7) + lane + 64];
        const float x3 = x[(b<<8) + (hh<<7) + lane + 96];
        float t = d * (float)sc[sb + 0];
        float wv = t * (float)((int)((ql0 & 0xFu) | ((qhb & 3u) << 4)) - 32);
        a += wv * x0;
        t = d * (float)sc[sb + 2];
        wv = t * (float)((int)((ql32 & 0xFu) | (((qhb >> 2) & 3u) << 4)) - 32);
        a += wv * x1;
        t = d * (float)sc[sb + 4];
        wv = t * (float)((int)((ql0 >> 4) | (((qhb >> 4) & 3u) << 4)) - 32);
        a += wv * x2;
        t = d * (float)sc[sb + 6];
        wv = t * (float)((int)((ql32 >> 4) | (((qhb >> 6) & 3u) << 4)) - 32);
        a += wv * x3;
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) y[row] = a;
  }
}
