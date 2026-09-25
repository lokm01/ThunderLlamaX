// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
#define NEG_INF (__int_as_float(0xff800000))
struct __align__(16) skv_u4 { unsigned x, y, z, w; };
extern "C" __global__ void __launch_bounds__(256) k1_test(float* data0_7225344, float* data1_18432, half* data2_205520896, const int data3_) {
  float* __restrict__ ws = (float*)0x1234567890ULL;
  const int s  = blockIdx.x / 12;
  const int gr = blockIdx.x % 12;
  const int g = gr / 3, r = gr % 3;
  const int start = s * 5018;
  const int cend = min(start + 5018, 100352);
  const int end_r = min(cend, data3_ + r + 1);
  const int h0 = g * 6;
  const __half* Kg = data2_205520896 + (size_t)g * (100352 * 256);
  const __half* Vg = Kg + (size_t)100352 * 1024;
  __shared__ __align__(16) __half sK[32][256];
  __shared__ __align__(16) __half sV[32][256];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nwarp = 8;
  float q[6][8];
  #pragma unroll
  for (int h6 = 0; h6 < 6; h6++)
    #pragma unroll
    for (int j = 0; j < 8; j++)
      q[h6][j] = data1_18432[(size_t)(h0 + h6) * 768 + r * 256 + lane * 8 + j];
  float m[6], l[6], o[6][8];
  #pragma unroll
  for (int h6 = 0; h6 < 6; h6++) { m[h6] = NEG_INF; l[h6] = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) o[h6][j] = 0.0f; }
  for (int t0 = start; t0 < cend; t0 += 32) {
    {
      const int pr = tid >> 3, dc = (tid & 7) << 5;
      const int pbase = t0 + pr;
      if (pbase < cend) {
        const skv_u4* ksrc = (const skv_u4*)(Kg + (size_t)pbase * 256 + dc);
        const skv_u4* vsrc = (const skv_u4*)(Vg + (size_t)pbase * 256 + dc);
        skv_u4* kd = (skv_u4*)&sK[pr][dc];
        skv_u4* vd = (skv_u4*)&sV[pr][dc];
        kd[0]=ksrc[0]; kd[1]=ksrc[1]; kd[2]=ksrc[2]; kd[3]=ksrc[3];
        vd[0]=vsrc[0]; vd[1]=vsrc[1]; vd[2]=vsrc[2]; vd[3]=vsrc[3];
      }
    }
    __syncthreads();
    const int npos = min(32, cend - t0);
    for (int pi = warp; pi < npos; pi += nwarp) {
      const int p = t0 + pi;
      const bool valid = p < end_r;
      #pragma unroll
      for (int h6 = 0; h6 < 6; h6++) {
        float dpt = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++)
          dpt += q[h6][j] * __half2float(sK[pi][lane * 8 + j]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dpt += __shfl_xor_sync(0xffffffffu, dpt, off);
        const float raw = dpt * 0.0625f;
        if (lane == 0) data0_7225344[(size_t)(h0 + h6) * (3 * 100352) + (size_t)r * 100352 + p] = valid ? raw : NEG_INF;
        if (valid) {
          const float s2 = raw * 1.4426950216293335f;
          if (s2 > m[h6]) {
            const float sc = exp2f(m[h6] - s2);
            l[h6] *= sc;
            #pragma unroll
            for (int j = 0; j < 8; j++) o[h6][j] *= sc;
            m[h6] = s2;
          }
          const float e = exp2f(s2 - m[h6]);
          l[h6] += e;
          #pragma unroll
          for (int j = 0; j < 8; j++) o[h6][j] += e * __half2float(sV[pi][lane * 8 + j]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int h6 = 0; h6 < 6; h6++) {
    float* w = ws + ((((size_t)(s * 12 + gr) * 6) + h6) * 8 + warp) * 258;
    #pragma unroll
    for (int j = 0; j < 8; j++) w[lane * 8 + j] = o[h6][j];
    if (lane == 0) { w[256] = m[h6]; w[257] = l[h6]; }
  }

}
