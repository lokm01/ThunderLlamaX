// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
#define HEADS 32
#define V_DIM 128
#define K_DIM 128
#define T_MAX 8
#define ROWS_PER_WARP (V_DIM / 8)

extern "C" __global__ void __launch_bounds__(256) gdn_scan(
    float* __restrict__ state,
    float* __restrict__ outs,
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

  __shared__ float s_k[T_MAX][K_DIM];
  __shared__ float s_q[T_MAX][K_DIM];
  __shared__ float s_beta[T_MAX];
  __shared__ float s_alpha[T_MAX];

  for (int i = threadIdx.x; i < T * K_DIM; i += blockDim.x) {
    int t = i / K_DIM, kk = i % K_DIM;
    s_k[t][kk] = k[h * T * K_DIM + t * K_DIM + kk];
    s_q[t][kk] = q[h * T * K_DIM + t * K_DIM + kk];
  }
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    s_beta[t] = beta[h * T + t];
    s_alpha[t] = alpha[h * T + t];
  }
  __syncthreads();

  for (int vv = 0; vv < ROWS_PER_WARP; vv++) {
    const int v_idx = v_base + vv;
    float* state_row = state + (h * V_DIM + v_idx) * K_DIM;

    float st0 = state_row[lane * 4];
    float st1 = state_row[lane * 4 + 1];
    float st2 = state_row[lane * 4 + 2];
    float st3 = state_row[lane * 4 + 3];

    for (int t = 0; t < T; t++) {
      float a = s_alpha[t];
      st0 *= a; st1 *= a; st2 *= a; st3 *= a;

      float kd = st0 * s_k[t][lane*4] + st1 * s_k[t][lane*4+1]
               + st2 * s_k[t][lane*4+2] + st3 * s_k[t][lane*4+3];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) kd += __shfl_down_sync(0xffffffffu, kd, o);
      kd = __shfl_sync(0xffffffffu, kd, 0);  // broadcast to all lanes

      float d = (v[h * T * V_DIM + t * V_DIM + v_idx] - kd) * s_beta[t];
      st0 += d * s_k[t][lane*4];
      st1 += d * s_k[t][lane*4+1];
      st2 += d * s_k[t][lane*4+2];
      st3 += d * s_k[t][lane*4+3];

      float qd = st0 * s_q[t][lane*4] + st1 * s_q[t][lane*4+1]
               + st2 * s_q[t][lane*4+2] + st3 * s_q[t][lane*4+3];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) qd += __shfl_down_sync(0xffffffffu, qd, o);
      qd = __shfl_sync(0xffffffffu, qd, 0);  // broadcast

      if (lane == 0) outs[h * T * V_DIM + t * V_DIM + v_idx] = qd;
    }

    state_row[lane*4]   = st0;
    state_row[lane*4+1] = st1;
    state_row[lane*4+2] = st2;
    state_row[lane*4+3] = st3;
  }
}
