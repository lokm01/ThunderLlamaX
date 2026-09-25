// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#include <cuda_fp16.h>
#define NEG_INF (__int_as_float(0xff800000))
struct __align__(16) skv_u4 { unsigned x, y, z, w; };
extern "C" __global__ void __launch_bounds__(256) k2_test(float* data0_18432, float* data1_7225344, float* data2_72, float* data3_72, half* data4_205520896, float* data5_36864) {
  const float* __restrict__ ws = (const float*)0x1234567890ULL;
  const int hr = blockIdx.x;            // 72 = h*3 + r
  const int h = hr / 3, r = hr % 3, d = threadIdx.x;
  const int gr = (h / 6) * 3 + r, h6 = h % 6;
  float M = NEG_INF;
  for (int s = 0; s < 20; s++)
    for (int wv = 0; wv < 8; wv++)
      M = fmaxf(M, ws[((((size_t)(s * 12 + gr) * 6) + h6) * 8 + wv) * 258 + 256]);
  float L_ = 0.0f, O = 0.0f;
  for (int s = 0; s < 20; s++)
    for (int wv = 0; wv < 8; wv++) {
      const float* w = ws + ((((size_t)(s * 12 + gr) * 6) + h6) * 8 + wv) * 258;
      const float wt = exp2f(w[256] - M);
      L_ += wt * w[257];
      O += wt * w[d];
    }
  const float gt = data5_36864[d + (size_t)h * 512 + (size_t)r * 12288 + 256];
  data0_18432[d + (size_t)h * 256 + (size_t)r * 6144] = (O / L_) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));

}
