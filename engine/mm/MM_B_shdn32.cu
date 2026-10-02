// MM SESSION B (L3): shdn32 -- the shared-expert down seat-loop GEMM
// (phase B of shexp8, M-batched): 2048 rows x k=512 Q8_0, weights read
// once per row, SW=32 seats per group, plain dot + fp32 store to shb
// (no activation -- VERBATIM phase B; the act comes from shgu32 via a
// global fp32 buffer with exact values).
#include <cuda_fp16.h>
#ifndef SW
#define SW 32
#endif
#ifndef BC
#define BC 8
#endif
#ifndef TPB
#define TPB 256
#endif
#define NCH (16/BC)
#define RPC (TPB/32)
extern "C" __global__ void __launch_bounds__(TPB) shdn32(
    const unsigned char* __restrict__ wd,   // [2048][544] Q8_0 down
    const float* __restrict__ actsh,        // [seats][512] fp32 (from shgu32)
    float* __restrict__ y,                  // [seats][2048] fp32
    const int seats)
{
  __shared__ float xsm[SW][BC*32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * RPC + warp;           // 2048 % RPC == 0
  const unsigned char* rd = wd + (size_t)row*544;
  for (int g0 = 0; g0 < seats; g0 += SW) {
    float acc[SW];
    #pragma unroll
    for (int sw = 0; sw < SW; ++sw) acc[sw] = 0.f;
    for (int c = 0; c < NCH; ++c) {
      float wc[BC];
      #pragma unroll
      for (int b = 0; b < BC; ++b) {
        const int bi = c*BC + b;
        const float d = __half2float(*((const __half*)(rd + 34*bi)));
        const signed char q = (const signed char)rd[34*bi + 2 + lane];
        wc[b] = d * (float)q;
      }
      __syncthreads();
      for (int i = threadIdx.x; i < SW*BC*32; i += TPB) {
        const int sw = i / (BC*32), r = i % (BC*32);
        xsm[sw][r] = actsh[(size_t)(g0 + sw)*512 + c*BC*32 + r];
      }
      __syncthreads();
      #pragma unroll
      for (int b = 0; b < BC; ++b) {
        #pragma unroll
        for (int sw = 0; sw < SW; ++sw)
          acc[sw] += wc[b] * xsm[sw][b*32 + lane];
      }
    }
    #pragma unroll
    for (int sw = 0; sw < SW; ++sw) {
      float a = acc[sw];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) y[(size_t)(g0 + sw)*2048 + row] = a;
    }
  }
}
