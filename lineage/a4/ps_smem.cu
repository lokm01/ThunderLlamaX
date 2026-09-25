// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// bisect: smem staging + 256thr structure of gdn_scan_ps, NO scan logic —
// writes a trivial combine of the staged data to all outputs.
#define HEADS 32
#define V_DIM 128
#define K_DIM 128
#define T_MAX 8
extern "C" __global__ void __launch_bounds__(256) ps_smem(
    float* __restrict__ state, float* __restrict__ outs, float* __restrict__ states_out,
    const float* __restrict__ alpha, const float* __restrict__ beta,
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const int T)
{
  const int h = blockIdx.x;
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
  const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i0 < T) {
    const int t = i0;
    float acc = s_alpha[t] + s_beta[t];
    for (int kk = 0; kk < K_DIM; kk++) acc += s_k[t][kk] + s_q[t][kk];
    for (int vv = threadIdx.x; vv < V_DIM; vv += blockDim.x) {
      outs[h * T * V_DIM + t * V_DIM + vv] = acc;
      states_out[((size_t)t * HEADS + h) * V_DIM * K_DIM + vv * K_DIM] = acc;
    }
    if (threadIdx.x == 0) state[h * V_DIM * K_DIM + t] = acc;
  }
}
