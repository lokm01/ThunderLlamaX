// MM P34: gv8k2048p -- the multi-position variant of gv8k2048 (Q8_0 GEMV
// k=2048): grid (rows/32, P), blockIdx.y = position; x/hn strided [P][2048].
// Same lane order (per-lane elems {b*32+l}, b asc running sums, xor tree).
// Serves (per position): GDN attn_qkv [8192r] + attn_gate [4096r]; attn
// attn_q [8192r] / attn_k / attn_v [512r]. -fmad=false.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(1024) gv8k2048p(
    const unsigned char* __restrict__ w,   // [rows][64*34]
    const float* __restrict__ x,           // [P][2048]
    float* __restrict__ y,                 // [P][rows] fp32
    const int rows)
{
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x << 5) + warp;
  const int p = blockIdx.y;
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
    if (lane == 0) y[(size_t)p*rows + row] = a;   // y [P][rows]
  }
}
