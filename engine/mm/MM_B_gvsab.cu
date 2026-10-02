// MM SESSION B (L1): gvsab -- the seat-loop port of gvf32ab (the GDN
// ssm_alpha/beta F32 projections, [32 rows, 2048 cols] x 2 halves fused).
// grid (16 row-slots/4, seats/SW): blockIdx.x = row-slot block (4 slots per
// CTA at TPB=128), blockIdx.y = seat group; the seat split adds the
// parallelism the 64-row class lacks (the w re-read per group is 1MB x 8 =
// trivial vs the per-seat 256x re-read today).
// Per (seat, row-slot): a += wr[b*32+lane]*xr[b*32+lane], b asc 0..64,
// then the stock 5-xor tree; store ab[seat*64 + half*32 + row] -- VERBATIM
// order (the S3 half/row mapping preserved: slot<32 = Wa rows, else Wb).
#include <cuda_fp16.h>
#ifndef SW
#define SW 32
#endif
#ifndef BC
#define BC 8
#endif
#ifndef TPB
#define TPB 128
#endif
#define NCH (64/BC)
#define RPC (TPB/32)
extern "C" __global__ void __launch_bounds__(TPB) gvsab(
    const float* __restrict__ Wa,        // [32][2048]
    const float* __restrict__ Wb,        // [32][2048]
    const float* __restrict__ x,         // [seats][2048] hn
    float* __restrict__ ab,              // [seats][64]
    const int seats)
{
  __shared__ float xsm[SW][BC*32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int slot = blockIdx.x * RPC + warp;      // 0..63
  const int half = slot >> 5, row = slot & 31;
  const float* wr = (half == 0 ? Wa : Wb) + (size_t)row*2048;
  const int g0 = blockIdx.y * SW;
  float acc[SW];
  #pragma unroll
  for (int sw = 0; sw < SW; ++sw) acc[sw] = 0.f;
  for (int c = 0; c < NCH; ++c) {
    float wc[BC];
    #pragma unroll
    for (int b = 0; b < BC; ++b) wc[b] = wr[(c*BC + b)*32 + lane];
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
    if (g0 + sw >= seats) break;
    float a = acc[sw];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) ab[(size_t)(g0 + sw)*64 + half*32 + row] = a;
  }
}
