// MM SESSION A (L1-swap POC): gv8k4096rs -- the RASTER-SWAPPED twin of
// MM_P34_gv8k4096r (Q8_0 GEMV k=4096 + residual add). Same math as stock,
// blockIdx roles swapped (row from blockIdx.y, seat p from blockIdx.x;
// launch grid (P, rows/32) instead of (rows/32, P)). Row-block = 32x4352B
// = 139KB shared across ~82 co-resident seats. Bit-identical by
// construction. Bench-only instrument.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(1024) gv8k4096rs(
    const unsigned char* __restrict__ w,   // [rows][128*34]
    const float* __restrict__ x,           // [P][4096]
    const float* __restrict__ hres,        // [P][2048] residual
    float* __restrict__ y,                 // [P][2048] fp32
    const int rows)
{
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.y << 5) + warp;
  const int p = blockIdx.x;
  if (row < rows) {
    const unsigned char* wr = w + (size_t)row*4352;
    const float* xr = x + (size_t)p*4096;
    float a = 0.f;
    #pragma unroll 16
    for (int b = 0; b < 128; ++b) {
      const float d = __half2float(*((const __half*)(wr + 34*b)));
      const signed char q = (const signed char)wr[34*b + 2 + lane];
      const float w_ = d * (float)q;
      a += w_ * xr[(b << 5) + lane];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) y[(size_t)p*2048 + row] = hres[(size_t)p*2048 + row] + a;
  }
}
