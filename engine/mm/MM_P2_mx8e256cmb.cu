#include <cuda_fp16.h>
// MM P2: mx8e256cmb -- the MoE combine. One CTA per position, 256 threads x 8
// elems. FIXED RANK ORDER fp32: y = (sum_{r=0..7} gates[p][r] * part[p][r][i])
// + sg[p] * shared[p][i]; fp16 store. Gates = the renormed top-8 (rt8e256);
// sg = sigmoid shared-gate; shared = shexp8 output (fp32). -fmad=false.
extern "C" __global__ void __launch_bounds__(256) mx8e256cmb(
    const __half* __restrict__ parts,   // [P][8][2048] rank-addressed partials
    const float* __restrict__ gates,    // [P][8]
    const float* __restrict__ sg,       // [P]
    const float* __restrict__ shared,   // [P][2048] fp32
    __half* __restrict__ y)             // [P][2048]
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
    y[(size_t)p*2048 + i0 + k] = __float2half(acc);
  }
}
