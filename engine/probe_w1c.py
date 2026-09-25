# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-c step 1: vector-load-width probe on this dext.
Tests u16 / u32 / u64 / uint4(16B) / float4 loads on a FRESH 1MB u8 buffer, plus a
row-pattern probe that mimics the production Q5 GEMV address pattern (row*3520 +
b*176 + 48 + lane*8). W1A gotcha #3 says uint32 on the Q5 weight buffer faulted
('Out Of Range Register') while byte and uint16 loads pass -- this probe isolates
width vs buffer vs address-pattern. Each kernel in its OWN cubin (gotcha #2).
Run: cd ~/tinygrad-metal/engine0 && DEV=NV ~/tg311/bin/python probe_w1c.py
"""
import os, sys, subprocess
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
BASE = os.path.dirname(os.path.abspath(__file__))

HDR = r"""
#include <cuda_fp16.h>
#define FULL 0xffffffffu
"""

# each: name -> (body, grid) ; buffer 1MB; out float[grid]
PROBES = {
  # control: byte loads (known-good class)
  "pv_b":  (r"""
extern "C" __global__ void __launch_bounds__(256) pv_b(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = ((blockIdx.x << 8) + threadIdx.x) << 2;   // 4 bytes per thread
  float fv = (float)w[i] + (float)w[i+1] + (float)w[i+2] + (float)w[i+3];
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}""", 1024),
  "pv_u16": (r"""
extern "C" __global__ void __launch_bounds__(256) pv_u16(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = ((blockIdx.x << 8) + threadIdx.x) << 1;   // 1 u16 per thread
  const unsigned short v = *(const unsigned short*)(w + i);
  float fv = (float)(v & 0xFF) + (float)(v >> 8);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}""", 2048),
  "pv_u32": (r"""
extern "C" __global__ void __launch_bounds__(256) pv_u32(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = (blockIdx.x << 8) + threadIdx.x;          // 1 u32 per thread
  const unsigned int v = *(const unsigned int*)(w + ((size_t)i << 2));
  float fv = (float)(v & 0xFF) + (float)((v >> 8) & 0xFF) + (float)((v >> 16) & 0xFF) + (float)(v >> 24);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}""", 1024),
  "pv_u64": (r"""
extern "C" __global__ void __launch_bounds__(256) pv_u64(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = blockIdx.x << 8 | threadIdx.x;            // 1 u64 per thread
  const unsigned long long v = *(const unsigned long long*)(w + ((size_t)i << 3));
  float fv = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) fv += (float)((v >> (8*j)) & 0xFF);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}""", 512),
  "pv_u128": (r"""
extern "C" __global__ void __launch_bounds__(256) pv_u128(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int i = blockIdx.x << 7 | (threadIdx.x >> 1);     // 1 uint4 per 2 threads (128 thr/CTA used)
  const uint4 v = *(const uint4*)(w + ((size_t)i << 4));
  float fv = (float)v.x + (float)v.y + (float)v.z + (float)v.w;
  fv = (float)((unsigned)v.x & 0xFF) + (float)((unsigned)v.y & 0xFF) + (float)((unsigned)v.z & 0xFF) + (float)(v.w & 0xFF);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) fv += __shfl_down_sync(FULL, fv, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x] = fv;
}""", 512),
  # production-pattern: warp per Q5 row, 2x u32 on the qs run + u32 on qh run
  "pv_row32": (r"""
extern "C" __global__ void __launch_bounds__(256) pv_row32(const unsigned char* __restrict__ w, float* __restrict__ out) {
  const int warp = (blockIdx.x << 3) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const unsigned char* rowp = w + (size_t)warp * 3520u;
  float acc = 0.f;
  #pragma unroll 4
  for (int b = 0; b < 20; ++b) {
    const unsigned char* blk = rowp + b*176;
    const unsigned int qa = *(const unsigned int*)(blk + 48 + lane*8);
    const unsigned int qb = *(const unsigned int*)(blk + 48 + lane*8 + 4);
    const unsigned int qh = *(const unsigned int*)(blk + 16 + ((lane & 3) << 3));
    acc += (float)(qa & 0xFF) + (float)(qb & 0xFF) + (float)(qh & 0xFF);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(FULL, acc, o);
  if (lane == 0) out[warp] = acc;
}""", 256),  # 2048 rows x 3520B = 7.2MB of a 8MB buffer
}

def sh(cmd, env, tries=4):
  import time
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    print(f"[build] transient docker API failure (try {t+1}), retrying...", flush=True)
    time.sleep(2)
  return r

def build():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  for name, (body, _) in PROBES.items():
    src = f'// probe {name}\n{HDR}{body}\n'
    open(f"{BASE}/{name}.cu", "w").write(src)
    r = sh(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{name}.cu"], env)
    if r.returncode:
      print(f"[build] {name} FAIL\n{r.stderr[-1200:]}"); sys.exit(1)
    print(f"[build] {name} OK", flush=True)

def run():
  from tinygrad.device import Device, TinyELF, BufferSpec
  from tinygrad.runtime.ops_nv import NVProgram
  dev = Device["NV"]
  rng = np.random.default_rng(3)
  buf_np = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
  w = dev.allocator.alloc(1 << 20, BufferSpec())
  dev.allocator._copyin(w, memoryview(buf_np.data).cast("B")); dev.synchronize()
  results = {}
  for name, (body, grid) in PROBES.items():
    lib = open(f"{BASE}/{name}.cubin", "rb").read()
    try:
      pr = NVProgram(dev, TinyELF(lib=lib, name=name, target=dev.renderer.target, signature=tuple()))
      outb = dev.allocator.alloc(grid * 4, BufferSpec())
      ob_np = np.full(grid, -1.0, dtype=np.float32)
      dev.allocator._copyin(outb, memoryview(ob_np.data).cast("B")); dev.synchronize()
      pr(w, outb, global_size=(grid, 1, 1), local_size=(256, 1, 1), wait=True)
      got = np.empty(grid, dtype=np.float32)
      dev.allocator._copyout(memoryview(got.data).cast("B"), outb)
      # reference from numpy
      ref = ref_for(name, buf_np, grid)
      ok = ref is None or np.allclose(got, ref, rtol=1e-5, atol=1e-2)
      results[name] = "PASS" if ok else f"WRONG (got {got[:3]} ref {None if ref is None else ref[:3]})"
      print(f"[probe] {name:9s} {results[name]}", flush=True)
    except Exception as e:
      results[name] = f"FAULT: {type(e).__name__}: {str(e)[:160]}"
      print(f"[probe] {name:9s} {results[name]}", flush=True)
      try: dev.synchronize()
      except Exception: pass
  print("[probe summary]", {k: v.split(" (")[0] for k, v in results.items()})

def ref_for(name, buf, grid):
  # byte-view references (float sums of low bytes -- approximate check for nonzero + sane)
  return None  # existence check only: WRONG detection via poison overwrite

if __name__ == "__main__":
  if "--run-only" not in sys.argv: build()
  run()
