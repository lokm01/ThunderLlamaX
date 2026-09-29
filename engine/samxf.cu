// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// TLX P10-dense (Route A rung 1): the FULL-VOCAB draft argmax. amx3's proven
// single-row body (warp-shuffle + smem reduce, lowest-index tiebreak) writing
// the row index DIRECTLY as the token id (the full head needs no stab gather).
// Launched with global_size=(1,1,1) per chain step; writes dring[j].
#include <cuda_fp16.h>
#define FULL 0xffffffffu
#define VOCAB 248320

extern "C" __global__ void __launch_bounds__(256) samxf(
    const __half* __restrict__ slogits, int* __restrict__ out_tok)
{
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  __shared__ int sm[16];
  float best = -1e30f; int bidx = 0;
  for (int i = tid; i < VOCAB; i += 256) {
    const float v = __half2float(slogits[i]);
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
    out_tok[0] = sm[8];
  }
}
