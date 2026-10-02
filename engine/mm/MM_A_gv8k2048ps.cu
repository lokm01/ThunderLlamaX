// MM SESSION A (L1-swap POC): gv8k2048ps -- the RASTER-SWAPPED twin of
// MM_P34_gv8k2048p. Identical warp math (same lane order, per-lane elems
// {b*32+l}, b-asc running sums, 5-xor tree, -fmad=false) -- ONLY the
// blockIdx role decode changes:
//     stock gv8k2048p : row = (blockIdx.x<<5)+warp ; p = blockIdx.y ; grid (rows/32, P)
//     swap gv8k2048ps: row = (blockIdx.y<<5)+warp ; p = blockIdx.x ; grid (P, rows/32)
// x-fastest CTA walk -> ~82 co-resident CTAs = 82 consecutive seats of the
// SAME 32-row block -> the 69.6KB row-block DRAM-misses once, L2-hits the
// rest. Outputs bit-identical to stock by construction (per-seat math
// untouched). Bench-only instrument; the production kernel is unchanged.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(1024) gv8k2048ps(
    const unsigned char* __restrict__ w,   // [rows][64*34]
    const float* __restrict__ x,           // [P][2048]
    float* __restrict__ y,                 // [P][rows] fp32
    const int rows)
{
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.y << 5) + warp;
  const int p = blockIdx.x;
  if (row < rows) {
    const unsigned char* wr = w + (size_t)row*2176;
    const float* xr = x + (size_t)p*2048;
    float a = 0.f;
    #pragma unroll 8
    for (int b = 0; b < 64; ++b) {
      const float d = __half2float(*((const __half*)(wr + 34*b)));
      const signed char q = (const signed char)wr[34*b + 2 + lane];
      const float w_ = d * (float)q;
      a += w_ * xr[(b << 5) + lane];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) y[(size_t)p*rows + row] = a;
  }
}
