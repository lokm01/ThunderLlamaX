// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W2D-L1v2: half2-packed M=3 GEMV cores (old weight layout). Per-element
// fp contract IDENTICAL to m3.cu: same half values (float2half RN == halves2half2 RN
// per element; hmul2 == hmul elementwise; half22float2 == half2float), same per-acc
// add order (j ascending) -> BIT-IDENTICAL outputs. x loads stay half (no
// float round-trip: half->float->half is identity, values unchanged).
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define DIM 5120
#define FFN_N 17408

#define RED3(A0,A1,A2) { \
  _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) { A0 += __shfl_down_sync(FULL, A0, o); A1 += __shfl_down_sync(FULL, A1, o); A2 += __shfl_down_sync(FULL, A2, o); } }

// load row T k-chunk KO as 4 half2 (one uint4)
#define LDH2(NM, XB, TS, T, KO) const uint4 NM##_raw = *(const uint4*)((XB) + (T)*(TS) + (KO)); \
  const __half2* NM = (const __half2*)&NM##_raw;

// core: weight floats WV[8] -> half2 pairs, x rows X{0,1,2} as half2[4], acc A{0,1,2}
// order per acc: elements 2k, 2k+1 ascending == j ascending (m3.cu ACC3 order)
#define ACC3H2(X0, X1, X2, WV, A0, A1, A2) { \
  const __half2 w01 = __halves2half2(__float2half((WV)[0]), __float2half((WV)[1])); \
  const __half2 w23 = __halves2half2(__float2half((WV)[2]), __float2half((WV)[3])); \
  const __half2 w45 = __halves2half2(__float2half((WV)[4]), __float2half((WV)[5])); \
  const __half2 w67 = __halves2half2(__float2half((WV)[6]), __float2half((WV)[7])); \
  const __half2* x0 = (X0); const __half2* x1 = (X1); const __half2* x2 = (X2); \
  { const float2 p = __half22float2(__hmul2(x0[0], w01)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[1], w23)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[2], w45)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x0[3], w67)); A0 += p.x; A0 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[0], w01)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[1], w23)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[2], w45)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x1[3], w67)); A1 += p.x; A1 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[0], w01)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[1], w23)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[2], w45)); A2 += p.x; A2 += p.y; } \
  { const float2 p = __half22float2(__hmul2(x2[3], w67)); A2 += p.x; A2 += p.y; } }

__device__ __forceinline__ __half hsilu_h2(__half h){
  return __hmul(h, hrcp((__half)1.0f + hexp2(__hmul(h, __float2half(-1.4423828125f)))));
}

// ---- ffn8v_3: gate+up IQ3 GEMVs + silu-mul, 3 rows (NB=20, old 1960B layout) ----
extern "C" __global__ void __launch_bounds__(1024) ao8nw32_3(
    const unsigned char* __restrict__ wo, const float* __restrict__ grid512,
    const __half* __restrict__ ao_in3, __half* __restrict__ attn_out3)
{
  const int warp = (blockIdx.x << 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= DIM) return;
  const unsigned char* rowp = wo + (size_t)warp * 2640u;
  float a0=0.f, a1=0.f, a2=0.f;
  #pragma unroll 5
  for (int b = 0; b < 24; ++b) {
    const unsigned char* blk = rowp + (size_t)b * 110u;
    const float d = __half2float(*((const __half*)blk));
    const int g0i = lane*2, g1i = lane*2 + 1;
    const int sraw = lane >> 2;
    const float sc = 1.0f + 2.0f*(float)((blk[106 + (sraw>>1)] >> ((sraw&1)<<2)) & 0xF);
    const unsigned char* sgnb = blk + 74 + lane;
    const int koff = (b << 8) + (lane << 3);
    LDH2(xv0, ao_in3, 6144, 0, koff) LDH2(xv1, ao_in3, 6144, 1, koff) LDH2(xv2, ao_in3, 6144, 2, koff)
    const unsigned short q16 = *(const unsigned short*)(blk + 2 + 2*lane);
    const int qb0 = (q16 & 0xFF) + ((((blk[66 + (g0i>>3)] >> (g0i&7)) & 1u) << 8));
    const int qb1 = (q16 >> 8)   + ((((blk[66 + (g1i>>3)] >> (g1i&7)) & 1u) << 8));
    float wv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int g = (j < 4) ? qb0 : qb1;
      const float gv = grid512[(g << 2) + (j & 3)];
      const float sgn = ((*sgnb >> j) & 1) ? -1.f : 1.f;
      wv[j] = d * sc * gv * sgn;
    }
    ACC3H2(xv0, xv1, xv2, wv, a0, a1, a2)
  }
  RED3(a0,a1,a2)
  if (lane == 0) { attn_out3[0*DIM+warp] = (__half)a0; attn_out3[1*DIM+warp] = (__half)a1; attn_out3[2*DIM+warp] = (__half)a2; }
}

// ---- q5g8v_3: qkv (Q5 raw) + gate (IQ3 packed), 3 rows, half2 core ----
