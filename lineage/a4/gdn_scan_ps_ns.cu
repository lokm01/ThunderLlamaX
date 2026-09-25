// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// bisect A: no-smem variant — direct global reads, full scan loop kept.
#define HEADS 32
#define V_DIM 128
#define K_DIM 128
#define ROWS_PER_WARP (V_DIM / 8)

extern "C" __global__ void __launch_bounds__(256) gdn_scan_ps(
    float* __restrict__ state,
    float* __restrict__ outs,
    float* __restrict__ states_out,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int T)
{
  const int h = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int v_base = warp * ROWS_PER_WARP;

  for (int vv = 0; vv < ROWS_PER_WARP; vv++) {
    const int v_idx = v_base + vv;
    float* state_row = state + (h * V_DIM + v_idx) * K_DIM;

    float st0 = state_row[lane * 4];
    float st1 = state_row[lane * 4 + 1];
    float st2 = state_row[lane * 4 + 2];
    float st3 = state_row[lane * 4 + 3];

    for (int t = 0; t < T; t++) {
      const float a = alpha[h * T + t];
      st0 *= a; st1 *= a; st2 *= a; st3 *= a;

      const float* krow = k + (h * T + t) * K_DIM;
      float kd = st0 * krow[lane*4] + st1 * krow[lane*4+1]
               + st2 * krow[lane*4+2] + st3 * krow[lane*4+3];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) kd += __shfl_xor_sync(0xffffffffu, kd, o);

      const float d = (v[(h * T + t) * V_DIM + v_idx] - kd) * beta[h * T + t];
      st0 += d * krow[lane*4];
      st1 += d * krow[lane*4+1];
      st2 += d * krow[lane*4+2];
      st3 += d * krow[lane*4+3];

      float* ps = states_out + (((size_t)t * HEADS + h) * V_DIM + v_idx) * K_DIM;
      ps[lane*4]   = st0;
      ps[lane*4+1] = st1;
      ps[lane*4+2] = st2;
      ps[lane*4+3] = st3;

      const float* qrow = q + (h * T + t) * K_DIM;
      float qd = st0 * qrow[lane*4] + st1 * qrow[lane*4+1]
               + st2 * qrow[lane*4+2] + st3 * qrow[lane*4+3];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) qd += __shfl_xor_sync(0xffffffffu, qd, o);

      if (lane == 0) outs[h * T * V_DIM + t * V_DIM + v_idx] = qd;
    }

    state_row[lane*4]   = st0;
    state_row[lane*4+1] = st1;
    state_row[lane*4+2] = st2;
    state_row[lane*4+3] = st3;
  }
}
