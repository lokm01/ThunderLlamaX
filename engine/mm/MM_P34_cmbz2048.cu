// MM P34: cmbz2048 -- MoE combine + RESIDUAL (the train variant of
// mx8e256cmb): h_out = h_resid + (sum_{r} gates[r] * part[r][i]) + sg*shared.
// FIXED RANK ORDER r asc, then shared, then resid; fp32 store (the fp32 trunk
// contract). One CTA per position, 256 threads x 8 elems. -fmad=false.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) cmbz2048(
    const __half* __restrict__ parts,   // [P][8][2048] rank-addressed fp16
    const float* __restrict__ gates,    // [P][8]
    const float* __restrict__ sg,       // [P]
    const float* __restrict__ shared,   // [P][2048] fp32
    const float* __restrict__ hres,     // [P][2048] fp32 residual
    float* __restrict__ y)              // [P][2048] fp32
{
  const int p = blockIdx.x;
  const int i0 = threadIdx.x << 3;
  float g[8];
  #pragma unroll
  for (int r = 0; r < 8; ++r) g[r] = gates[p*8 + r];
  const float s = sg[p];
  const __half* pb = parts + (size_t)p*8*2048;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    float acc = 0.f;
    #pragma unroll
    for (int r = 0; r < 8; ++r) acc += g[r] * __half2float(pb[r*2048 + i0 + k]);
    acc += s * shared[(size_t)p*2048 + i0 + k];
    y[(size_t)p*2048 + i0 + k] = hres[(size_t)p*2048 + i0 + k] + acc;
  }
}
