
#include <cuda_fp16.h>
#define SHARDN 850
#define SLABB 1458176

extern "C" __global__ void __launch_bounds__(1024) kb2(
    const unsigned char* __restrict__ b0, const unsigned char* __restrict__ b1,
    const unsigned char* __restrict__ b2, const unsigned char* __restrict__ b3,
    const unsigned char* __restrict__ b4, const unsigned char* __restrict__ b5,
    const unsigned char* __restrict__ b6, const unsigned char* __restrict__ b7,
    const unsigned short* __restrict__ eids,
    const float* __restrict__ xs, const float* __restrict__ gridf, float* __restrict__ ys)
{
  const int p = blockIdx.x;
  const unsigned int e = eids[p];
  const size_t off = (size_t)(e % SHARDN) * SLABB;
  const unsigned int s = e / SHARDN;
  const unsigned char* base = b0;
  if (s == 1) base = b1; else if (s == 2) base = b2; else if (s == 3) base = b3;
  else if (s == 4) base = b4; else if (s == 5) base = b5; else if (s == 6) base = b6;
  else if (s == 7) base = b7;
  float a = (float)(base + off)[0];
  a += xs[p*2048] + gridf[0];
  ys[p*512] = a;
}
