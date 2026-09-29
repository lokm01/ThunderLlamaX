// MM_P0 D4: rt8e256poc — device-side router top-8 producer (the mini-graph's
// "router-write" node). CTA per position; 1024 threads = 256 experts x 4 parts;
// per-thread sequential float4 sums -> smem parts -> deterministic 4-way reduce
// -> ONE warp does 8 masked-argmax passes (strict >, tie->lower id = the gold
// router contract) writing eids[p*8+rank]. smem = 256*4 + 256 floats = 5KB.
#define NPAIR 88

extern "C" __global__ void __launch_bounds__(1024) rt8e256poc(
    const float* __restrict__ wrt,        // [256][2048]
    const float* __restrict__ h,          // [NPAIR][2048]
    unsigned short* __restrict__ eids)    // [NPAIR][8]
{
  const int p = blockIdx.x;
  __shared__ float lg[256];
  __shared__ float parts[256][4];
  const float* hrow = h + (size_t)p*2048;
  const int e = threadIdx.x >> 2, part = threadIdx.x & 3;
  const float* wrow = wrt + (size_t)e*2048 + part*512;
  float s = 0.f;
  for (int i = 0; i < 512; i += 4) {
    const float4 wv = *(const float4*)(wrow + i);
    const float4 hv = *(const float4*)(hrow + part*512 + i);
    s += wv.x*hv.x; s += wv.y*hv.y; s += wv.z*hv.z; s += wv.w*hv.w;
  }
  parts[e][part] = s;
  __syncthreads();
  if (part == 0) lg[e] = parts[e][0] + parts[e][1] + parts[e][2] + parts[e][3];
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int r = 0; r < 8; ++r) {
      int be = 0; float bv = lg[0];
      for (int e2 = 1; e2 < 256; ++e2) {
        if (lg[e2] > bv) { bv = lg[e2]; be = e2; }
      }
      lg[be] = -3.4e38f;
      eids[(size_t)p*8 + r] = (unsigned short)be;
    }
  }
}
