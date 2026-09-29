
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define KB 8
#define ROWB (110*KB)
#define NROW 512
extern "C" __global__ void __launch_bounds__(1024) kbv(
    const unsigned long long* __restrict__ ptbl,
    const unsigned short* __restrict__ eids,
    const float* __restrict__ xs, const float* __restrict__ gridf,
    float* __restrict__ ys)
{
  const int p = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* base = (const unsigned char*)(size_t)ptbl[eids[p]];
  const float* x = xs + (size_t)p*2048;
  float* y = ys + (size_t)p*512;
  for (int r0 = 0; r0 < NROW; r0 += 32) {
    const unsigned char* rowp = base + (size_t)(r0+warp)*ROWB;
    float a = 0.f;
    #pragma unroll
    for (int b = 0; b < KB; ++b) {
      const unsigned char* blk = rowp + b*110;
      const float d = __half2float(*((const __half*)blk));
      const int g0i = lane*2, g1i = lane*2 + 1;
      const int sraw = lane >> 2;
      const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
      const unsigned int sg = blk[74 + lane];
      const int koff = (b << 8) + (lane << 3);
      const float x0 = x[koff+0], x1 = x[koff+1], x2 = x[koff+2], x3 = x[koff+3];
      const float x4 = x[koff+4], x5 = x[koff+5], x6 = x[koff+6], x7 = x[koff+7];
      const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lane);
      const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0i>>3)] >> (g0i&7)) & 1u) << 8));
      const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1i>>3)] >> (g1i&7)) & 1u) << 8));
      const float* gr0 = gridf + (size_t)qb0*4;
      const float* gr1 = gridf + (size_t)qb1*4;
      float w0 = d*sc*gr0[0]; if (sg & 0x01u) w0 = -w0;
      float w1 = d*sc*gr0[1]; if (sg & 0x02u) w1 = -w1;
      float w2 = d*sc*gr0[2]; if (sg & 0x04u) w2 = -w2;
      float w3 = d*sc*gr0[3]; if (sg & 0x08u) w3 = -w3;
      float w4 = d*sc*gr1[0]; if (sg & 0x10u) w4 = -w4;
      float w5 = d*sc*gr1[1]; if (sg & 0x20u) w5 = -w5;
      float w6 = d*sc*gr1[2]; if (sg & 0x40u) w6 = -w6;
      float w7 = d*sc*gr1[3]; if (sg & 0x80u) w7 = -w7;
      a += w0*x0; a += w1*x1; a += w2*x2; a += w3*x3;
      a += w4*x4; a += w5*x5; a += w6*x6; a += w7*x7;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(FULL, a, o);
    if (lane == 0) y[r0+warp] = a;
  }
}
