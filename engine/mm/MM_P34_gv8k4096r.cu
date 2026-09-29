// MM P34: gv8k4096r -- gv8k4096 (Q8_0 GEMV k=4096) + RESIDUAL ADD (the train
// variant serving both token-mixer out_projs: GDN ssm_out [2048r] and attn
// attn_output [2048r]). y[p][row] = dot(x[p]) + hres[p][row]; fp32.
// Grid (rows/32, P): blockIdx.y = position. One warp per row,
// 128 x 34B blocks; per-lane elems {b*32+l} b asc; xor tree. -fmad=false.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(1024) gv8k4096r(
    const unsigned char* __restrict__ w,   // [rows][128*34]
    const float* __restrict__ x,           // [P][4096]
    const float* __restrict__ hres,        // [P][2048] residual (y may alias hres)
    float* __restrict__ y,                 // [P][2048] fp32
    const int rows)
{
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x << 5) + warp;
  const int p = blockIdx.y;
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
