// MM_P0 D4: mm_hrot — device-side hidden-state drift so router ids CHANGE each
// replay (h += eps*k: shifts logits by eps*(W . k), varies per expert).
// One CTA of 128 threads, one position per thread, sequential k loop.
#define NPAIR 88

extern "C" __global__ void __launch_bounds__(128) mm_hrot(
    float* __restrict__ h)
{
  const int i = threadIdx.x;
  if (i < NPAIR) {
    float* row = h + (size_t)i*2048;
    for (int k = 0; k < 2048; ++k) row[k] = row[k] + 0.001f*(float)(k+1);
  }
}
