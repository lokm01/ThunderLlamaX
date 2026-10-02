// MM SESSION A (expert-M histogram probe): cp4k -- u16 copy kernel.
// Appended after each rt8e256 in the PF seq (bench-only capture build) to
// snapshot eids[P*8] into EIDSCAP[layer][2048] before the next layer
// overwrites eidsb. Grid ceil(n/256), 256 threads.
extern "C" __global__ void __launch_bounds__(256) cp4k(
    const unsigned short* __restrict__ src,
    unsigned short* __restrict__ dst,
    const int n)
{
  const int i = blockIdx.x*256 + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
