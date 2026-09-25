# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""One variant per process: python a5_iso2.py <variant>"""
import sys, os
sys.path.insert(0, "~/tinygrad-src")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("JIT", "2")
import time
import numpy as np
from tinygrad.device import Device, BufferSpec, TinyELF
from tinygrad import dtypes as _dt

dev = Device["NV"]
rend = dev.renderer
N = 786432
V = sys.argv[1]

BODIES = {
  # exact a5 kernel (float4, grid-stride)
  "f4_768x256": ("""extern "C" __global__ void __launch_bounds__(256) K(float* data0_%d, float* data1_%d) {
  const int n4 = %d >> 2;
  const float4* s4 = (const float4*)data1_%d;
  float4* d4 = (float4*)data0_%d;
  for (int i = (blockIdx.x*256)+threadIdx.x; i < n4; i += gridDim.x*256) d4[i] = s4[i];
}
""" % (N, N, N, N, N), (768, 1, 1), (256, 1, 1)),
  # scalar grid-stride, same dims
  "sc_768x256": ("""extern "C" __global__ void __launch_bounds__(256) K(float* data0_%d, float* data1_%d) {
  for (int i = (blockIdx.x*256)+threadIdx.x; i < %d; i += gridDim.x*256) data0_%d[i] = data1_%d[i];
}
""" % (N, N, N, N, N), (768, 1, 1), (256, 1, 1)),
  # float4, ONE float4 per thread, grid exactly n4/256, no stride loop
  "f4_flat": ("""extern "C" __global__ void __launch_bounds__(256) K(float* data0_%d, float* data1_%d) {
  const int i = (blockIdx.x*256)+threadIdx.x;
  ((float4*)data0_%d)[i] = ((const float4*)data1_%d)[i];
}
""" % (N, N, N, N), (N // 1024, 1, 1), (256, 1, 1)),
  # float2 (8-byte) grid-stride
  "f2_768x256": ("""extern "C" __global__ void __launch_bounds__(256) K(float* data0_%d, float* data1_%d) {
  const int n2 = %d >> 1;
  const float2* s2 = (const float2*)data1_%d;
  float2* d2 = (float2*)data0_%d;
  for (int i = (blockIdx.x*256)+threadIdx.x; i < n2; i += gridDim.x*256) d2[i] = s2[i];
}
""" % (N, N, N, N, N), (N // 512, 1, 1), (256, 1, 1)),
  # float4 loads + SCALAR stores (is it the 16B store?)
  "f4ld_scst_768x256": ("""extern "C" __global__ void __launch_bounds__(256) K(float* data0_%d, float* data1_%d) {
  const int n4 = %d >> 2;
  const float4* s4 = (const float4*)data1_%d;
  for (int i = (blockIdx.x*256)+threadIdx.x; i < n4; i += gridDim.x*256) {
    float4 v = s4[i];
    float* d = data0_%d + (i << 2);
    d[0]=v.x; d[1]=v.y; d[2]=v.z; d[3]=v.w;
  }
}
""" % (N, N, N, N, N), (768, 1, 1), (256, 1, 1)),
}
src, grid, local = BODIES[V]
x = np.arange(N, dtype=np.float32)
xb = dev.allocator.alloc(4 * N, BufferSpec()); dev.allocator._copyin(xb, memoryview(x.tobytes()).cast("B"))
yb = dev.allocator.alloc(4 * N, BufferSpec()); dev.allocator._copyin(yb, memoryview(bytes(4 * N)).cast("B"))
lib = rend.compiler.compile_cached(src)
elf = TinyELF(lib, "K", rend.target, (("data0", 0, _dt.float, (N,)), ("data1", 1, _dt.float, (N,))), None)
fn = dev.runtime(elf)
t0 = time.perf_counter()
fn(yb, xb, global_size=grid, local_size=local, vals=(), wait=True)
dt = (time.perf_counter() - t0) * 1e3
mv = memoryview(bytearray(4 * N)); dev.allocator._copyout(mv, yb)
got = np.frombuffer(mv, dtype=np.float32)
print(f"[{V}] returned {dt:.2f}ms match={np.array_equal(got, x)}", flush=True)
# repeat 20x for timing
t0 = time.perf_counter()
for _ in range(20): fn(yb, xb, global_size=grid, local_size=local, vals=(), wait=True)
print(f"[{V}] b2b {(time.perf_counter()-t0)/20*1e3:.2f}ms", flush=True)
