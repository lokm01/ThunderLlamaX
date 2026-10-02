// MM SESSION B (L1): gvs32k4096r -- the seat-loop M-GEMV for the k=4096
// Q8_0 + RESIDUAL classes (GDN ssm_out / attn o via gv8k4096r today).
// Same scheme as gvs32k2048 (SW=32/BC=8/256thr, seat count runtime) with
// 128 Q8_0 blocks (16 chunks) and the stock residual epilogue
// y[seat][row] = hres[seat][row] + a  (verbatim order, -fmad=false).
// => bit-identical to gv8k4096r per seat by construction.
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
#define NCH (128/BC)
#define RPC (TPB/32)
extern "C" __global__ void __launch_bounds__(TPB) gvs32k4096r(
    const unsigned char* __restrict__ w,   // [rows][128*34]
    const float* __restrict__ x,           // [seats][4096]
    const float* __restrict__ hres,        // [seats][2048] (y may alias hres)
    float* __restrict__ y,                 // [seats][2048] fp32
    const int rows,
    const int seats)
{
  __shared__ float xsm[SW][BC*32];
  // rows % RPC == 0 asserted host-side (2048 for both classes)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x * RPC) + warp;
  const unsigned char* wr = w + (size_t)row*4352;
  for (int g0 = 0; g0 < seats; g0 += SW) {
    float acc[SW];
    #pragma unroll
    for (int sw = 0; sw < SW; ++sw) acc[sw] = 0.f;
    for (int c = 0; c < NCH; ++c) {
      float wc[BC];
      #pragma unroll
      for (int b = 0; b < BC; ++b) {
        const int bi = c*BC + b;
        const float d = __half2float(*((const __half*)(wr + 34*bi)));
        const signed char q = (const signed char)wr[34*bi + 2 + lane];
        wc[b] = d * (float)q;
      }
      __syncthreads();
      for (int i = threadIdx.x; i < SW*BC*32; i += TPB) {
        const int sw = i / (BC*32), r = i % (BC*32);
        xsm[sw][r] = x[(size_t)(g0 + sw)*4096 + c*BC*32 + r];
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
      if (lane == 0) y[(size_t)(g0 + sw)*2048 + row] = hres[(size_t)(g0 + sw)*2048 + row] + a;
    }
  }
}
