# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R4 race fix: k2s5 t==4 conv write -> conv5x scratch (unread in-kernel);
acceptsel5k copies conv5x into live slot 4 when m==4 (per-block CTAs own their
slices — race-free). Applied to BOTH m5.cu (master) and the split k2s5.cu."""
import re, sys
BASE = "~/tinygrad-metal/engine0"

OLD_SIG = """extern "C" __global__ void __launch_bounds__(256) k2s5(
    float* __restrict__ conv_b, float* __restrict__ rec_b,
    const __half* __restrict__ qkv4, const __half* __restrict__ gate4,"""
NEW_SIG = """extern "C" __global__ void __launch_bounds__(256) k2s5(
    float* __restrict__ conv_b, float* __restrict__ rec_b, float* __restrict__ conv5x,
    const __half* __restrict__ qkv4, const __half* __restrict__ gate4,"""
OLD_W = """    {
      float* dst = conv_b + (size_t)t * (3*CONV_CH);
      const int tid = (h << 8) + threadIdx.x;"""
NEW_W = """    {
      // R4 RACE FIX: t==4 would write conv slot 4 = the LIVE window other CTAs
      // still read at their t=0..2 (conv writes are cross-head strided — no CTA
      // owns its channels). Route t==4's window to conv5x (unread in-kernel);
      // acceptsel5k copies conv5x -> slot 4 only when m==4. rec slot writes are
      // head-local slices (race-free) and stay in-kernel for all t.
      float* dst = (t == 4) ? conv5x : (conv_b + (size_t)t * (3*CONV_CH));
      const int tid = (h << 8) + threadIdx.x;"""

for fn in (f"{BASE}/m5.cu", f"{BASE}/k2s5.cu"):
    src = open(fn).read()
    assert OLD_SIG in src, fn + " sig"
    assert OLD_W in src, fn + " write"
    src = src.replace(OLD_SIG, NEW_SIG).replace(OLD_W, NEW_W)
    open(fn, "w").write(src)
    print(f"[fix] {fn} patched")

open(f"{BASE}/acceptsel5k.cu", "w").write("""// R4 acceptsel5k: deep-set state select. rec: slot m -> live 4 (head-local
// per-block CTAs; m==4 is a self-copy no-op). conv: m<4 copies slot m as the
// legacy acceptsel; m==4 copies conv5x (k2s5's race-safe t=4 window) -> slot 4.
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) acceptsel5k(
    float* __restrict__ rec4, float* __restrict__ conv4, const int* __restrict__ m_slot,
    const float* __restrict__ conv5x)
{
  const int b = blockIdx.x;
  const int m = m_slot[0];
  {
    const float* src = rec4 + ((size_t)b*5 + m) * 786432u;
    float* dst = rec4 + ((size_t)b*5 + 4) * 786432u;
    for (int i = threadIdx.x; i < 786432; i += 256) dst[i] = src[i];
  }
  if (m == 4) {
    const float* src = conv5x + (size_t)b * 30720u;
    float* dst = conv4 + ((size_t)b*5 + 4) * 30720u;
    for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
  } else {
    const float* src = conv4 + ((size_t)b*5 + m) * 30720u;
    float* dst = conv4 + ((size_t)b*5 + 4) * 30720u;
    for (int i = threadIdx.x; i < 30720; i += 256) dst[i] = src[i];
  }
}
""")
print("[fix] acceptsel5k.cu written")
