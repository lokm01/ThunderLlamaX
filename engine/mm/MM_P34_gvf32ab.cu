// MM P34: gvf32ab -- F32 GEMV for the GDN a/b projections (ssm_alpha/beta
// [32 rows, 2048 cols] F32, fused: out rows 0..31 = a, 32..63 = b).
// One CTA per token, 1024 threads = 32 warps, warp per row IN EACH HALF
// (half 0: Wa rows 0..31, half 1: Wb rows 0..31 -- row = warp, NOT
// half*32+warp which read 256KB past Wb: the S3 ab:BAD + latent S4 fault);
// per-lane elems {b*32+l} b asc running sums (64 blocks), xor tree.
// -fmad=false; out fp32 [T][64] (a | b).
extern "C" __global__ void __launch_bounds__(1024) gvf32ab(
    const float* __restrict__ Wa,        // [32][2048]
    const float* __restrict__ Wb,        // [32][2048]
    const float* __restrict__ x,         // [T][2048] hn
    float* __restrict__ ab)              // [T][64]
{
  const int t = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* xr = x + (size_t)t*2048;
  #pragma unroll
  for (int half = 0; half < 2; ++half) {
    const float* wr = (half == 0 ? Wa : Wb) + (size_t)warp*2048;
    float a = 0.f;
    #pragma unroll 8
    for (int b = 0; b < 64; ++b) {
      a += wr[b*32 + lane] * xr[b*32 + lane];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) ab[t*64 + half*32 + warp] = a;
  }
}
