// MM SESSION E (item 2): rt8e_m{2,4} -- the M-SEAT BATCHED router.
// Stock rt8e256 = 1 CTA/position, 1024 threads (e=tid>>2, part=tid&3),
// each thread a 512-float4 serial dot; the 2MB fp32 router weights are
// re-read by EVERY position's CTA. This port batches SEATS=2/4 positions
// per CTA: the weight float4 is loaded ONCE per i-step and applied to
// SEATS running sums (per-seat op order VERBATIM stock: x,y,z,w asc) --
// BIT-EXACT logits/gates/eids/sg by construction (the parts 4-way reduce
// order, the thread-0 shared-gate serial 1024-sum, and the top-8
// masked-argmax epilogue are replicated per seat).
//   smem 20KB; grid P/SEATS (static at seq build); vals=(P,).
// THE GOLD-ROUTER CONTRACT: never mma, never reordered.
#include <cuda_fp16.h>
#ifndef SEATS
#define SEATS 4
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(rt8e_m, SEATS)

extern "C" __global__ void __launch_bounds__(1024) KSYM(
    const float* __restrict__ wrt,        // [256][2048] F32
    const float* __restrict__ wsh,        // [2048] F32
    const float* __restrict__ h,          // [P][2048]
    unsigned short* __restrict__ eids,    // [P][8]
    float* __restrict__ gates,            // [P][8]
    float* __restrict__ sg,               // [P]
    const int P)
{
  const int g0 = blockIdx.x * SEATS;
  const int ns = min(SEATS, P - g0);      // seats in this group (>=1)
  __shared__ float parts[256][4][SEATS];
  __shared__ float lg[SEATS][256];
  const int e = threadIdx.x >> 2, part = threadIdx.x & 3;
  const float* wrow = wrt + (size_t)e * 2048 + part * 512;
  float acc[SEATS];
  #pragma unroll
  for (int s = 0; s < SEATS; ++s) acc[s] = 0.f;
  for (int i = 0; i < 512; i += 4) {
    const float4 wv = *(const float4*)(wrow + i);
    #pragma unroll
    for (int s = 0; s < SEATS; ++s) {
      if (s >= ns) break;
      const float4 hv = *(const float4*)(h + (size_t)(g0 + s) * 2048 + part * 512 + i);
      acc[s] += wv.x*hv.x; acc[s] += wv.y*hv.y; acc[s] += wv.z*hv.z; acc[s] += wv.w*hv.w;
    }
  }
  #pragma unroll
  for (int s = 0; s < SEATS; ++s)
    if (s < ns) parts[e][part][s] = acc[s];
  __syncthreads();
  if (part == 0) {
    #pragma unroll
    for (int s = 0; s < SEATS; ++s)
      if (s < ns)
        lg[s][e] = parts[e][0][s] + parts[e][1][s] + parts[e][2][s] + parts[e][3][s];
  }
  __syncthreads();
  // per-seat epilogue: seat s handled by thread s (the stock serial
  // shared-gate 1024-sum + top-8 masked argmax + softmax->top8->renorm)
  const int s = threadIdx.x;
  if (s < ns) {
    const float* hrow = h + (size_t)(g0 + s) * 2048;
    float a = wsh[0]*hrow[0] + wsh[1]*hrow[1];
    for (int i = 1; i < 1024; ++i) {
      const float t = wsh[2*i]*hrow[2*i] + wsh[2*i+1]*hrow[2*i+1];
      a = a + t;
    }
    sg[g0 + s] = 1.f/(1.f + __expf(-a));
    float* lgs = lg[s];
    float m = lgs[0];
    for (int i = 1; i < 256; ++i) if (lgs[i] > m) m = lgs[i];
    float s8 = 0.f;
    for (int r = 0; r < 8; ++r) {
      int be = 0; float bv = lgs[0];
      for (int e2 = 1; e2 < 256; ++e2) {
        if (lgs[e2] > bv) { bv = lgs[e2]; be = e2; }
      }
      eids[(g0 + s)*8 + r] = (unsigned short)be;
      gates[(g0 + s)*8 + r] = __expf(bv - m);
      s8 += __expf(bv - m);
      lgs[be] = -3.402823466e38f;
    }
    for (int r = 0; r < 8; ++r) gates[(g0 + s)*8 + r] = gates[(g0 + s)*8 + r] / s8;
  }
}
