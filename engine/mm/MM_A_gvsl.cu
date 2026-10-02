// MM SESSION A (L1-seat-loop POC): gvsl -- seat-loop M-GEMV for the qkv
// class (Q8_0, k=2048, rows=8192, P=256 seats). THE DESIGN (qwen variant):
//   CTA = 256 threads = 8 warps = 8 rows (the 1024-thread law: reg budget).
//   Each WARP decodes its row's Q8_0 into 64 fp32 w_ registers ONCE
//   (d*(float)q per elem, exactly the stock expression), then loops all
//   256 seats in groups of SW. Per group, the CTA cooperatively stages
//   x[SW seats][BC-block chunks] into static smem (SW*BC*128 B, <= 40KB),
//   each thread keeps SW fp32 accumulators, b-ascending mul+add per seat
//   (STOCK per-seat order: acc += w_[b] * x[seat][b*32+lane], -fmad=false),
//   then the stock 5-xor tree + lane-0 store per seat.
//   => bit-identical to gv8k2048p per seat by construction.
//   w DRAM-read once per row; x staged per CTA (L2-served); the 256x w
//   re-read AND the 256x quant-decode of the stock per-seat family die.
// Build variants: -DSW={8,16,32,64} -DBC={2,4,8}; P fixed at 256 (-DPFX).
// Bench-only instrument.
#include <cuda_fp16.h>
#ifndef SW
#define SW 8
#endif
#ifndef BC
#define BC 8
#endif
#ifndef TPB
#define TPB 256
#endif
#ifndef PFX
#define PFX 256
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(CAT(CAT(gvsl_, SW), CAT(_, BC)), CAT(_t, TPB))
#define NCH (64/BC)
#define RPC (TPB/32)   // rows per CTA = warps per CTA

extern "C" __global__ void __launch_bounds__(TPB) KSYM(
    const unsigned char* __restrict__ w,   // [rows][64*34]
    const float* __restrict__ x,           // [PFX][2048]
    float* __restrict__ y,                 // [PFX][rows] fp32
    const int rows)
{
  __shared__ float xsm[SW][BC*32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = (blockIdx.x * RPC) + warp;
  if (row >= rows) return;
  const unsigned char* wr = w + (size_t)row*2176;
  // NOTE (the register-array law): a runtime-indexed w_[c*BC+b] local array
  // lands in LOCAL memory (STACK:256 spill) -- decode per-chunk into a
  // compile-time-indexed wc[BC] instead. Each w element is still decoded
  // exactly once per seat-GROUP pass (NGx total vs the stock 256x).
  for (int g = 0; g < PFX/SW; ++g) {
    float acc[SW];
    #pragma unroll
    for (int sw = 0; sw < SW; ++sw) acc[sw] = 0.f;
    for (int c = 0; c < NCH; ++c) {
      float wc[BC];
      #pragma unroll
      for (int b = 0; b < BC; ++b) {
        const int bi = c*BC + b;   // runtime index: GLOBAL address only
        const float d = __half2float(*((const __half*)(wr + 34*bi)));
        const signed char q = (const signed char)wr[34*bi + 2 + lane];
        wc[b] = d * (float)q;
      }
      __syncthreads();
      for (int i = threadIdx.x; i < SW*BC*32; i += TPB) {
        const int sw = i / (BC*32), r = i % (BC*32);
        xsm[sw][r] = x[(size_t)(g*SW + sw)*2048 + c*BC*32 + r];
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
      if (lane == 0) y[(size_t)(g*SW + sw)*rows + row] = a;
    }
  }
}
