// MM P2: shexp8 -- the shared-expert FFN, q5g8v-class GEMV retargeted to
// 2048-hidden on the GGUF-native Q8_0 shared tensors. One CTA per position,
// 1024 thr = 32 warps. Phase A: 16 sweeps x dual dots (gate row + up row,
// 64 Q8_0 blocks each, k=2048) -> silu(g)*u -> smem t[512]; Phase B: 64 sweeps
// down rows (k=512, 16 blocks) -> fp32 y[P][2048]. Per-lane elems {b*32+l},
// b asc running sums, xor trees; -fmad=false.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(1024) shexp8(
    const unsigned char* __restrict__ wg,   // [512][2176] Q8_0 gate
    const unsigned char* __restrict__ wu,   // [512][2176] Q8_0 up
    const unsigned char* __restrict__ wd,   // [2048][544] Q8_0 down
    const float* __restrict__ x,            // [P][2048]
    float* __restrict__ y)                  // [P][2048]
{
  const int p = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* xr = x + (size_t)p*2048;
  __shared__ float t[512];
  for (int r0 = 0; r0 < 512; r0 += 32) {
    const unsigned char* rg = wg + (size_t)(r0+warp)*2176;
    const unsigned char* ru = wu + (size_t)(r0+warp)*2176;
    float ag = 0.f, au = 0.f;
    #pragma unroll 8
    for (int b = 0; b < 64; ++b) {
      const float dg = __half2float(*((const __half*)(rg + 34*b)));
      const signed char qg = (const signed char)rg[34*b + 2 + lane];
      const float wg_ = dg * (float)qg;
      ag += wg_ * xr[(b<<5) + lane];
      const float du = __half2float(*((const __half*)(ru + 34*b)));
      const signed char qu = (const signed char)ru[34*b + 2 + lane];
      const float wu_ = du * (float)qu;
      au += wu_ * xr[(b<<5) + lane];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ag += __shfl_xor_sync(0xffffffffu, ag, o);
      au += __shfl_xor_sync(0xffffffffu, au, o);
    }
    if (lane == 0) {
      const float g = ag, u = au;
      t[r0+warp] = (g / (1.0f + __expf(-g))) * u;
    }
  }
  __syncthreads();
  float* yr = y + (size_t)p*2048;
  for (int r0 = 0; r0 < 2048; r0 += 32) {
    const unsigned char* rd = wd + (size_t)(r0+warp)*544;
    float a = 0.f;
    #pragma unroll
    for (int b = 0; b < 16; ++b) {
      const float d = __half2float(*((const __half*)(rd + 34*b)));
      const signed char q = (const signed char)rd[34*b + 2 + lane];
      const float w = d * (float)q;
      a += w * t[(b<<5) + lane];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) yr[r0+warp] = a;
  }
}
