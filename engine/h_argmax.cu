// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 W1-b: embed (IQ3_S gather+dequant), argmax over half logits.
// The head GEMV reuses k0_norm (output_norm -> xh half) + k1_q5 (Q5_K 248320-row
// GEMV, grid 31040) from W1-a cubins.
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define VOCAB 248320

// ---- h_embed: read token id from device slot, gather+dequant its IQ3_S row.
// grid=(1,), 256 threads; row = 2200B = 20 blocks x 110B. Output fp32 x[5120]. ----
extern "C" __global__ void __launch_bounds__(256) h_argmax(
    const __half* __restrict__ logits, int* __restrict__ tok_slot,
    int* __restrict__ pos_slot, int* __restrict__ tok_hist)
{
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  __shared__ int sm[16];   // single-array smem rule: [0..7] float-bits of val, [8..15] idx
  float best = -1e30f; int bidx = 0;
  for (int i = tid; i < VOCAB; i += 256) {
    const float v = __half2float(logits[i]);
    if (v > best || (v == best && i < bidx)) { best = v; bidx = i; }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(FULL, best, o);
    const int oi = __shfl_down_sync(FULL, bidx, o);
    if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
  }
  if (lane == 0) { sm[warp] = __float_as_int(best); sm[8+warp] = bidx; }
  __syncthreads();
  if (tid == 0) {
    for (int w = 1; w < 8; ++w)
      if (__int_as_float(sm[w]) > __int_as_float(sm[0]) || (__int_as_float(sm[w]) == __int_as_float(sm[0]) && sm[8+w] < sm[8])) { sm[0] = sm[w]; sm[8] = sm[8+w]; }
    const int bidx_f = sm[8];
    const int pos = pos_slot[0];
    tok_slot[0] = bidx_f;
    tok_hist[pos] = bidx_f;
    pos_slot[0] = pos + 1;
  }
}

// ---- k3a_iq3: GDN o_proj for the 24 blocks whose ssm_out is IQ3_XXS (6144 in -> 5120 out,
//      row 2352B = 24 blocks x 98B) ----
