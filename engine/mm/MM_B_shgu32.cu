// MM SESSION B (L3): shgu32 -- the shared-expert gate+up seat-loop GEMM
// (phase A of shexp8, M-batched). shexp8 today launches one CTA per seat
// and re-reads the whole 512x2176 Q8_0 gate+up pair per seat (~870MB/chunk
// DRAM); this port reads the weights once per row and loops SW=32 seats
// per group (the gvs32 scheme), writing the silu(g)*u activations to a
// global fp32 buffer for the shdn32 pass (the original staged t[512] in
// smem -- same VALUES, zero extra rounding: fp32 store is exact).
// Per (seat,row): b-asc running sums ag/au (gate-add then up-add per b,
// the stock interleaving; independent accumulators), 5-xor trees, silu.
// => bit-identical to shexp8 phase A per seat by construction.
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
#define NCH (64/BC)
#define RPC (TPB/32)
extern "C" __global__ void __launch_bounds__(TPB) shgu32(
    const unsigned char* __restrict__ wg,   // [512][2176] Q8_0 gate
    const unsigned char* __restrict__ wu,   // [512][2176] Q8_0 up
    const float* __restrict__ x,            // [seats][2048]
    float* __restrict__ actsh,              // [seats][512] fp32
    const int seats)
{
  __shared__ float xsm[SW][BC*32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * RPC + warp;           // 512 % RPC == 0
  const unsigned char* rg = wg + (size_t)row*2176;
  const unsigned char* ru = wu + (size_t)row*2176;
  for (int g0 = 0; g0 < seats; g0 += SW) {
    float ag[SW], au[SW];
    #pragma unroll
    for (int sw = 0; sw < SW; ++sw) { ag[sw] = 0.f; au[sw] = 0.f; }
    for (int c = 0; c < NCH; ++c) {
      float wcg[BC], wcu[BC];
      #pragma unroll
      for (int b = 0; b < BC; ++b) {
        const int bi = c*BC + b;
        const float dg = __half2float(*((const __half*)(rg + 34*bi)));
        const signed char qg = (const signed char)rg[34*bi + 2 + lane];
        wcg[b] = dg * (float)qg;
        const float du = __half2float(*((const __half*)(ru + 34*bi)));
        const signed char qu = (const signed char)ru[34*bi + 2 + lane];
        wcu[b] = du * (float)qu;
      }
      __syncthreads();
      for (int i = threadIdx.x; i < SW*BC*32; i += TPB) {
        const int sw = i / (BC*32), r = i % (BC*32);
        xsm[sw][r] = x[(size_t)(g0 + sw)*2048 + c*BC*32 + r];
      }
      __syncthreads();
      #pragma unroll
      for (int b = 0; b < BC; ++b) {
        #pragma unroll
        for (int sw = 0; sw < SW; ++sw) {
          ag[sw] += wcg[b] * xsm[sw][b*32 + lane];
          au[sw] += wcu[b] * xsm[sw][b*32 + lane];
        }
      }
    }
    #pragma unroll
    for (int sw = 0; sw < SW; ++sw) {
      float g = ag[sw], u = au[sw];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, o);
        u += __shfl_xor_sync(0xffffffffu, u, o);
      }
      if (lane == 0) actsh[(size_t)(g0 + sw)*512 + row] = (g / (1.0f + __expf(-g))) * u;
    }
  }
}
