// ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 lokm01
// engine0 M1-A: device memset (int32 words) for fixed-handle state resets.
// No scalar kernel args (empty-sig vals unwritten gotcha): value + count come
// from buffers. Flat indexing, 8 sequential words per thread, per-element guard
// (NO grid-stride law). 256 threads/CTA -> no warp-count name token needed.
// grid = ceil(n/2048), local 256. Eager-only (never in a graph).
extern "C" __global__ void __launch_bounds__(256) mfill(
    int* __restrict__ dst, const int* __restrict__ val, const int* __restrict__ nbuf)
{
  const int base = blockIdx.x * 2048;
  const int v = val[0];
  const int n = nbuf[0];
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = base + threadIdx.x + k * 256;
    if (i < n) dst[i] = v;
  }
}
