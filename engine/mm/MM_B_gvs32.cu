// MM SESSION B (L1): gvs32k2048 -- the seat-loop M-GEMV for the k=2048 Q8_0
// trunk classes (GDN qkv/z + attn q/k/v), the Session-A POC winner gvsl_32_8
// (1.66x, bit-exact, regs<=102) with the seat count as a RUNTIME val so one
// cubin serves the PF-256 graph and the PF-64 tail (M-grid FOLD: the seat
// loop replaces the P grid dimension -- the P7E4 fold-not-loop law).
// CTA = 256 threads = 8 warps = 8 rows; each warp decodes its row chunk
// into wc[BC] regs exactly once per seat-GROUP pass, the CTA cooperatively
// stages x[SW seats][BC-block chunk] to smem (SW*BC*128 B = 32KB), b-asc
// mul+add per seat (STOCK per-seat order: acc += w_[b]*x[seat][b*32+lane],
// -fmad=false), then the stock 5-xor tree + lane-0 store per seat.
// => bit-identical to gv8k2048p per seat by construction (POC-proven).
// w DRAM-read once per row (vs 256x per-seat re-read); x staged per CTA.
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
extern "C" __global__ void __launch_bounds__(TPB) gvs32k2048(
    const unsigned char* __restrict__ w,   // [rows][64*34]
    const float* __restrict__ x,           // [seats][2048]
    float* __restrict__ y,                 // [seats][rows] fp32
    const int rows,
    const int seats)
{
  __shared__ float xsm[SW][BC*32];
  // rows % RPC == 0 asserted host-side for every launched class (8192/4096/
  // 512) -- NO early return: the staging __syncthreads() must be CTA-uniform.
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x * RPC) + warp;
  const unsigned char* wr = w + (size_t)row*2176;
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
        xsm[sw][r] = x[(size_t)(g0 + sw)*2048 + c*BC*32 + r];
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
      if (lane == 0) y[(size_t)(g0 + sw)*rows + row] = a;
    }
  }
}
