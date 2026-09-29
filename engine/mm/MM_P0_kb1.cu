
#include <cuda_fp16.h>
#define SHARDN 850
#define SLABB 1458176

extern "C" __global__ void __launch_bounds__(1024) kb1(
    const unsigned char* __restrict__ b0, const unsigned char* __restrict__ b1,
    const unsigned char* __restrict__ b2, const unsigned char* __restrict__ b3,
    const unsigned char* __restrict__ b4, const unsigned char* __restrict__ b5,
    const unsigned char* __restrict__ b6, const unsigned char* __restrict__ b7,
    const unsigned short* __restrict__ eids,
    const float* __restrict__ xs, const float* __restrict__ gridf, float* __restrict__ ys)
{
  const int p = blockIdx.x;
  float a = (float)(b0[0]+b1[0]+b2[0]+b3[0]+b4[0]+b5[0]+b6[0]+b7[0]);
  a += (float)eids[p] + xs[p*2048] + gridf[0];
  ys[p*512] = a;
}
