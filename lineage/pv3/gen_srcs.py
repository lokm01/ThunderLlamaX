# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os
D = os.path.expanduser("~/tinygrad-metal/pv3")

PV = """#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pvL(
    float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2, float* __restrict__ data3,
    half* __restrict__ data4, float* __restrict__ data5, float* __restrict__ data6, float* __restrict__ data7,
    float* __restrict__ data8, float* __restrict__ data9, float* __restrict__ data10, float* __restrict__ data11,
    const int dummy)
{
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x + (blockIdx.z << 7);
  const float* P  = (r == 0) ? data8 : (r == 1) ? data5 : data1;
  const float  mx = (r == 0) ? data9[g]  : (r == 1) ? data6[g]  : data2[g];
  const float  rs = (r == 0) ? data10[g] : (r == 1) ? data7[g] : data3[g];
  const float* P_g = P + ((size_t)g * %d);
  const __half* V_g = (const __half*)data4 + (((size_t)(g / 6)) * %dLL) + %dLL;
  float acc = 0.0f;
  for (int p = 0; p < %d; ++p)
    acc += exp2f((P_g[p] - mx) * 1.4426950216293334f) * __half2float(V_g[((size_t)p << 8) + d]);
  const float gate = data11[d + ((size_t)g << 9) + 256 + (size_t)r * 12288];
  data0[d + ((size_t)g << 8) + (size_t)r * 6144] =
    (acc / rs) * (1.0f / (1.0f + exp2f(-gate * 1.4426950216293334f)));
}
"""

def emu(L):
    return """#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(16) emu(
    float* __restrict__ data0, float* __restrict__ data1, float* __restrict__ data2, float* __restrict__ data3,
    half* __restrict__ data4, float* __restrict__ data5, float* __restrict__ data6, float* __restrict__ data7,
    float* __restrict__ data8, float* __restrict__ data9, float* __restrict__ data10, float* __restrict__ data11,
    const int dummy)
{
  const int gidx1 = blockIdx.y;
  const int alu0 = (threadIdx.x + (blockIdx.x << 4));
  const size_t alu1 = (size_t)gidx1 * %d;
  const size_t alu2 = (size_t)(gidx1 / 6) * %dLL + %dLL;
  { float acc = 0.0f; const float mx = data9[gidx1];
    for (int p = 0; p < %d; ++p)
      acc += exp2f((data8[alu1 + p] - mx) * 1.4426950216293335f) * __half2float(data4[alu2 + (((size_t)p) << 8) + alu0]);
    const float gate = data11[alu0 + ((size_t)gidx1 << 9) + 256];
    data0[alu0 + ((size_t)gidx1 << 8)] = (acc / data10[gidx1]) * (1.0f / (1.0f + exp2f(-gate * 1.4426950216293335f))); }
  { float acc = 0.0f; const float mx = data6[gidx1];
    for (int p = 0; p < %d; ++p)
      acc += exp2f((data5[alu1 + p] - mx) * 1.4426950216293335f) * __half2float(data4[alu2 + (((size_t)p) << 8) + alu0]);
    const float gate = data11[alu0 + ((size_t)gidx1 << 9) + 12544];
    data0[alu0 + ((size_t)gidx1 << 8) + 6144] = (acc / data7[gidx1]) * (1.0f / (1.0f + exp2f(-gate * 1.4426950216293335f))); }
  { float acc = 0.0f; const float mx = data2[gidx1];
    for (int p = 0; p < %d; ++p)
      acc += exp2f((data1[alu1 + p] - mx) * 1.4426950216293335f) * __half2float(data4[alu2 + (((size_t)p) << 8) + alu0]);
    const float gate = data11[alu0 + ((size_t)gidx1 << 9) + 24832];
    data0[alu0 + ((size_t)gidx1 << 8) + 12288] = (acc / data3[gidx1]) * (1.0f / (1.0f + exp2f(-gate * 1.4426950216293335f))); }
}
""" % (L, L*256, L*1024, L, L, L)

for L in (8192, 100352):
    with open(f"{D}/pvL{L}.cu", "w") as f: f.write(PV % (L, L*256, L*1024, L))
    with open(f"{D}/emu{L}.cu", "w") as f: f.write(emu(L))
print("sources written")
