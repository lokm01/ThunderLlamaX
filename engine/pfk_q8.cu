// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// P8w4: activation-quant kernel — fp16 x rows -> per-(row,128k-chunk) int8.
// One CTA per row (g = #rows, M-agnostic), 8 warps, each warp owns NCH/8
// chunks. Per chunk: lane loads its 4 halfs (k = c*128 + lane*4), absmax via
// fixed shfl-xor tree, s = max(am,1e-6)/127, q = rintf(x/s) clamped +-127,
// char4 store, rowsum (int sum of the STORED s8s) via shfl-xor tree.
// Outputs: xq [M][KDIM] s8, sx [M][NCH] f32, rs [M][NCH] i32.
// DETERMINISTIC: fixed reduction trees + rintf (half-even). LAWS: flat
// indexing, no gridDim/blockDim reads, full-warp masks, zero smem, 0 spill.
// Build: -DKNAME=p8q8x -DKDIM=5120
#include <cuda_fp16.h>
#define NWARP 8
#define NTHR (NWARP * 32)
#define NCH (KDIM / 128)
#define CPC (NCH / NWARP)
#if CPC * NWARP != NCH
#error "NCH must be a multiple of 8 warps"
#endif

// P8W4 LAW: eager launches with exactly 4 buffer args fault at param 3+
// on this fork/dext (params 1-2 land, 3+ stale) — pad to 6 params (dummies
// never read) and pass 6 buffers at every launch site.
extern "C" __global__ void __launch_bounds__(NTHR) KNAME(
    const __half* __restrict__ x, signed char* __restrict__ xq,
    float* __restrict__ sx, int* __restrict__ rs,
    const float* __restrict__ pad5, const float* __restrict__ pad6) {
  (void)pad5; (void)pad6;
  const int m = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  _Pragma("unroll")
  for (int j = 0; j < CPC; ++j) {
    const int c = warp * CPC + j;
    const int k0 = c * 128 + lane * 4;
    const __half2 h01 = *(const __half2*)(x + (size_t)m * KDIM + k0);
    const __half2 h23 = *(const __half2*)(x + (size_t)m * KDIM + k0 + 2);
    const float f0 = __half2float(__low2half(h01)), f1 = __half2float(__high2half(h01));
    const float f2 = __half2float(__low2half(h23)), f3 = __half2float(__high2half(h23));
    float am = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
    _Pragma("unroll")
    for (int o = 16; o > 0; o >>= 1)
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float s = fmaxf(am, 1e-6f) * (1.0f / 127.0f);
    int q0 = (int)rintf(f0 / s), q1 = (int)rintf(f1 / s);
    int q2 = (int)rintf(f2 / s), q3 = (int)rintf(f3 / s);
    q0 = max(-127, min(127, q0)); q1 = max(-127, min(127, q1));
    q2 = max(-127, min(127, q2)); q3 = max(-127, min(127, q3));
    *(int*)(xq + (size_t)m * KDIM + k0) =
        (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    int rsum = q0 + q1 + q2 + q3;
    _Pragma("unroll")
    for (int o = 16; o > 0; o >>= 1)
      rsum += __shfl_xor_sync(0xffffffffu, rsum, o);
    if (lane == 0) {
      sx[(size_t)m * NCH + c] = s;
      rs[(size_t)m * NCH + c] = rsum;
    }
  }
}
