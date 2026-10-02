// MM SESSION A (L2/DRAM read-BW microbench): bwread -- ONE float4 per
// thread, exact-cover grid (n4/256 CTAs), NO gridDim/blockDim reads in
// kernel (THE DEXT LAW: both read 0 in-kernel -- the stride-loop variant
// hung the QMD live). Per-thread sum stored masked (defeats DCE; content
// unchecked). -DLDG=1 uses __ldcg (L1 bypass; legal+neutral on this dext)
// for the clean L2-hit number on the small buffer.
extern "C" __global__ void __launch_bounds__(256) bwread(
    const float4* __restrict__ g,   // [n4]
    float* __restrict__ out,        // [8192]
    const int n4)
{
  const int i = blockIdx.x*256 + threadIdx.x;
  if (i < n4) {
#if LDG
    const float4 v = __ldcg(g + i);
#else
    const float4 v = g[i];
#endif
    out[i & 8191] = v.x + v.y + v.z + v.w;
  }
}
