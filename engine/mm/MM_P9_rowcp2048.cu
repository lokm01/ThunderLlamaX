// MM P9: rowcp2048 -- copy one row (runtime index via vals) of an [N][2048]
// f32 buffer to a [2048] dst. The MTP chain's committed-hidden carry: the
// probe's hA row m (m = the accept count, host-read from eb) seeds the chain.
// 1 CTA x 256 thr, 8 elems each; m as the runtime scalar.
extern "C" __global__ void __launch_bounds__(256) rowcp2048(
    const float* __restrict__ src,   // [N][2048]
    float* __restrict__ dst,         // [2048]
    const int row)
{
  const float* r = src + (size_t)row*2048;
  const int i0 = threadIdx.x << 3;
  #pragma unroll
  for (int k = 0; k < 8; ++k) dst[i0 + k] = r[i0 + k];
}
