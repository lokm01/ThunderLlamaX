# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os
D = os.path.expanduser("~/tinygrad-metal/pv3")
L = 100352

# Kernel 1: partsum + IN-PLACE exp transform of P. Same signature/grid as stock r_256_72_49_8.
# Bit-identical partial sums (same grouping/order); P becomes exp2((P-mx)*log2e).
PS = """#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(49) pstransform(float* data0, float* data1, float* data2) {
  __shared__ __align__(16) float buf1[49];
  int gidx0 = blockIdx.x; /* 72 */
  float val0 = (*(data2+gidx0));
  int gidx1 = blockIdx.y; /* 256 */
  int lidx0 = threadIdx.x; /* 49 */
  int alu0 = ((gidx0*%d)+(gidx1*392)+(lidx0<<3));
  float s = 0.0f;
  #pragma unroll
  for (int k = 0; k < 8; k++) {
    float e = exp2f(((data1[alu0+k]-val0)*1.4426950216293335f));
    data1[alu0+k] = e;
    s += e;
  }
  buf1[lidx0] = s;
  __syncthreads();
  if (lidx0 == 0) {
    float t = 0.0f;
    for (int i = 0; i < 49; i++) t += buf1[i];
    *(data0+(gidx1+(gidx0<<8))) = t;
  }
}
"""

def pvA(Z):  # acc[3][6]: grid (4, Z, 1), 256/Z threads; V read once for all rows+heads
    NT = 256 // Z
    return """#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(%(NT)d) pvA(
    float* __restrict__ out, float* __restrict__ P, float* __restrict__ mx, float* __restrict__ sm,
    half* __restrict__ V, float* __restrict__ gate, const int dummy)
{
  const int s = blockIdx.x, d = threadIdx.x + (blockIdx.y * %(NT)d);
  float m[3][6], si[3][6], acc[3][6];
  #pragma unroll
  for (int r = 0; r < 3; r++) {
    #pragma unroll
    for (int j = 0; j < 6; j++) {
      const int h = s*6 + j;
      m[r][j] = mx[h*3 + r]; si[r][j] = sm[h*3 + r]; acc[r][j] = 0.0f;
    }
  }
  const float* PE = P;
  const __half* Vg = (const __half*)V + ((size_t)s * %(L256)d) + %(L1024)d;
  for (int p = 0; p < %(L)d; ++p) {
    const float v = __half2float(Vg[((size_t)p << 8) + d]);
    #pragma unroll
    for (int r = 0; r < 3; r++)
      #pragma unroll
      for (int j = 0; j < 6; j++)
        acc[r][j] += __ldg(PE + ((size_t)(s*6+j) * %(T3L)d) + ((size_t)r * %(L)d) + p) * v;
  }
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int j = 0; j < 6; j++) {
      const int h = s*6 + j;
      const float gt = gate[d + ((size_t)h << 9) + 256 + (size_t)r * 12288];
      out[d + ((size_t)h << 8) + (size_t)r * 6144] = (acc[r][j] / si[r][j]) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));
    }
}
""" % dict(NT=NT, L=L, L256=L*256, L1024=L*1024, T3L=3*L)

def pvB(Z):  # acc[6]: grid (4, Z, 3); row = blockIdx.z; V read once per row (3x total)
    NT = 256 // Z
    return """#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(%(NT)d) pvB(
    float* __restrict__ out, float* __restrict__ P, float* __restrict__ mx, float* __restrict__ sm,
    half* __restrict__ V, float* __restrict__ gate, const int dummy)
{
  const int s = blockIdx.x, r = blockIdx.z, d = threadIdx.x + (blockIdx.y * %(NT)d);
  float si[6], acc[6];
  #pragma unroll
  for (int j = 0; j < 6; j++) {
    const int h = s*6 + j;
    si[j] = sm[h*3 + r]; acc[j] = 0.0f;
  }
  const __half* Vg = (const __half*)V + ((size_t)s * %(L256)d) + %(L1024)d;
  for (int p = 0; p < %(L)d; ++p) {
    const float v = __half2float(Vg[((size_t)p << 8) + d]);
    #pragma unroll
    for (int j = 0; j < 6; j++)
      acc[j] += __ldg(P + ((size_t)(s*6+j) * %(T3L)d) + ((size_t)r * %(L)d) + p) * v;
  }
  #pragma unroll
  for (int j = 0; j < 6; j++) {
    const int h = s*6 + j;
    const float gt = gate[d + ((size_t)h << 9) + 256 + (size_t)r * 12288];
    out[d + ((size_t)h << 8) + (size_t)r * 6144] = (acc[j] / si[j]) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));
  }
}
""" % dict(NT=NT, L=L, L256=L*256, L1024=L*1024, T3L=3*L)

# D: pv3 shape minus exp (P pre-transformed), grid (3,24,2) x 128 — pure-mem reference point
DV = """#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) pvD(
    float* __restrict__ out, float* __restrict__ P, float* __restrict__ mx, float* __restrict__ sm,
    half* __restrict__ V, float* __restrict__ gate, const int dummy)
{
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x + (blockIdx.z << 7);
  const float si = sm[g*3 + r];
  const float* PE_g = P + ((size_t)g * 301056) + ((size_t)r * 100352);
  const __half* V_g = (const __half*)V + (((size_t)(g/6)) * 25690112LL) + 102760448LL;
  float acc = 0.0f;
  for (int p = 0; p < 100352; ++p)
    acc += PE_g[p] * __half2float(V_g[((size_t)p << 8) + d]);
  const float gt = gate[d + ((size_t)g << 9) + 256 + (size_t)r * 12288];
  out[d + ((size_t)g << 8) + (size_t)r * 6144] = (acc / si) * (1.0f / (1.0f + exp2f(-gt * 1.4426950216293334f)));
}
"""

with open(f"{D}/pstransform.cu", "w") as f: f.write(PS % L)
for Z in (1, 2, 4):
    with open(f"{D}/pvA_Z{Z}.cu", "w") as f: f.write(pvA(Z))
    with open(f"{D}/pvB_Z{Z}.cu", "w") as f: f.write(pvB(Z))
with open(f"{D}/pvD.cu", "w") as f: f.write(DV)
print("gen2 done")
