// MM PB1 (Part B, step 1): rtsh8 -- the FUSED router + shared-expert FFN
// (rt8e256 + shexp8 merged VERBATIM, one launch instead of two; the
// launch-serialization law: ~0.094ms per in-graph kernel dispatch).
// Both bodies are 1 CTA per position, 1024 threads, read the same h row,
// write DISJOINT outputs (eids/gates/sg vs shb) -> bit-exact by
// construction: every output's arithmetic + accumulation order is the
// original kernel's, only the __syncthreads schedule is shared.
// smem: lg[256] + parts[256][4] + shp[1024] + t[512] = 23KB. -fmad=false.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(1024) rtsh8(
    const float* __restrict__ wrt,        // [256][2048] F32 router
    const float* __restrict__ wsh,        // [2048] F32 shared-expert gate w
    const unsigned char* __restrict__ wg, // [512][2176] Q8_0 shared gate
    const unsigned char* __restrict__ wu, // [512][2176] Q8_0 shared up
    const unsigned char* __restrict__ wd, // [2048][544] Q8_0 shared down
    const float* __restrict__ h,          // [NP][2048]
    unsigned short* __restrict__ eids,    // [NP][8]
    float* __restrict__ gates,            // [NP][8]
    float* __restrict__ sg,               // [NP]
    float* __restrict__ shb)              // [NP][2048] fp32
{
  const int p = blockIdx.x;
  __shared__ float lg[256];
  __shared__ float parts[256][4];
  __shared__ float shp[1024];
  __shared__ float t[512];
  const float* hrow = h + (size_t)p*2048;
  const int e = threadIdx.x >> 2, part = threadIdx.x & 3;
  // ---- rt8e256 phase 1: router logits partials + shared-gate partials ----
  const float* wrow = wrt + (size_t)e*2048 + part*512;
  float s = 0.f;
  for (int i = 0; i < 512; i += 4) {
    const float4 wv = *(const float4*)(wrow + i);
    const float4 hv = *(const float4*)(hrow + part*512 + i);
    s += wv.x*hv.x; s += wv.y*hv.y; s += wv.z*hv.z; s += wv.w*hv.w;
  }
  parts[e][part] = s;
  shp[threadIdx.x] = wsh[threadIdx.x*2]*hrow[threadIdx.x*2]
                   + wsh[threadIdx.x*2+1]*hrow[threadIdx.x*2+1];
  // ---- shexp8 phase A: shared-expert gate+up dots -> t[512] ----
  {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int r0 = 0; r0 < 512; r0 += 32) {
      const unsigned char* rg = wg + (size_t)(r0+warp)*2176;
      const unsigned char* ru = wu + (size_t)(r0+warp)*2176;
      float ag = 0.f, au = 0.f;
      #pragma unroll 8
      for (int b = 0; b < 64; ++b) {
        const float dg = __half2float(*((const __half*)(rg + 34*b)));
        const signed char qg = (const signed char)rg[34*b + 2 + lane];
        const float wg_ = dg * (float)qg;
        ag += wg_ * hrow[(b<<5) + lane];
        const float du = __half2float(*((const __half*)(ru + 34*b)));
        const signed char qu = (const signed char)ru[34*b + 2 + lane];
        const float wu_ = du * (float)qu;
        au += wu_ * hrow[(b<<5) + lane];
      }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        ag += __shfl_xor_sync(0xffffffffu, ag, o);
        au += __shfl_xor_sync(0xffffffffu, au, o);
      }
      if (lane == 0) {
        const float g = ag, u = au;
        t[r0+warp] = (g / (1.0f + __expf(-g))) * u;
      }
    }
  }
  __syncthreads();
  // ---- rt8e256 phase 2: 4-way logit reduce ----
  if (part == 0) lg[e] = parts[e][0] + parts[e][1] + parts[e][2] + parts[e][3];
  // ---- shexp8 phase B: shared-expert down -> shb ----
  {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* yr = shb + (size_t)p*2048;
    for (int r0 = 0; r0 < 2048; r0 += 32) {
      const unsigned char* rd = wd + (size_t)(r0+warp)*544;
      float a = 0.f;
      #pragma unroll
      for (int b = 0; b < 16; ++b) {
        const float d = __half2float(*((const __half*)(rd + 34*b)));
        const signed char q = (const signed char)rd[34*b + 2 + lane];
        const float w = d * (float)q;
        a += w * t[(b<<5) + lane];
      }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) yr[r0+warp] = a;
    }
  }
  __syncthreads();
  // ---- rt8e256 phase 3: THE GOLD-ROUTER CONTRACT epilogue (thread 0) ----
  if (threadIdx.x == 0) {
    float a = shp[0];
    for (int i = 1; i < 1024; ++i) a += shp[i];
    sg[p] = 1.f/(1.f + __expf(-a));
    float m = lg[0];
    for (int i = 1; i < 256; ++i) if (lg[i] > m) m = lg[i];
    float s8 = 0.f;
    for (int r = 0; r < 8; ++r) {
      int be = 0; float bv = lg[0];
      for (int e2 = 1; e2 < 256; ++e2) {
        if (lg[e2] > bv) { bv = lg[e2]; be = e2; }
      }
      eids[p*8 + r] = (unsigned short)be;
      gates[p*8 + r] = __expf(bv - m);
      s8 += __expf(bv - m);
      lg[be] = -3.402823466e38f;
    }
    for (int r = 0; r < 8; ++r) gates[p*8 + r] = gates[p*8 + r] / s8;
  }
}
