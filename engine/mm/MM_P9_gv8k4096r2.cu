// MM P9: gv8k4096r2 -- gv8k4096r (Q8_0 GEMV k=4096 + residual) with the
// x vector SPLIT across two [P][2048] halves (xA = elems 0..2047, xB =
// 2048..4095) -- serves the MTP eh_proj whose input is the concat
// [enorm(emb(t)) || hnorm(h)] WITHOUT a staging copy. Row math verbatim
// gv8k4096r (warp/row, 128 x 34B blocks, per-lane elems {b*32+l} b asc, xor
// tree, y = hres + dot). Grid (rows/32, P). -fmad=false.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(1024) gv8k4096r2(
    const unsigned char* __restrict__ w,   // [rows][128*34]
    const float* __restrict__ xA,          // [P][2048] (k elems 0..2047)
    const float* __restrict__ xB,          // [P][2048] (k elems 2048..4095)
    const float* __restrict__ hres,        // [P][2048] residual
    float* __restrict__ y,                 // [P][2048] fp32
    const int rows)
{
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x << 5) + warp;
  const int p = blockIdx.y;
  if (row < rows) {
    const unsigned char* wr = w + (size_t)row*4352;
    const float* xa = xA + (size_t)p*2048;
    const float* xb = xB + (size_t)p*2048;
    float a = 0.f;
    #pragma unroll 8
    for (int b = 0; b < 128; ++b) {
      const float d = __half2float(*((const __half*)(wr + 34*b)));
      const signed char q = (const signed char)wr[34*b + 2 + lane];
      const float w_ = d * (float)q;
      const float xv = (b < 64) ? xa[(b << 5) + lane] : xb[((b - 64) << 5) + lane];
      a += w_ * xv;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) y[(size_t)p*2048 + row] = hres[(size_t)p*2048 + row] + a;
  }
}
