# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Standalone isolation of the E_2048_a5c copy-kernel hang. No model load.
Builds the exact kernel via the a3b compile path, launches on raw buffers."""
import sys, os
sys.path.insert(0, "~/tinygrad-src")
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("JIT", "2")
import numpy as np
from tinygrad.device import Device, BufferSpec
from tinygrad.codegen.a3b import a5_xform_copy

dev = Device["NV"]
rend = dev.renderer
N = 786432

src = a5_xform_copy(open("~/a5dump/E_2048_32_4_3_orig.cu").read(), "E_copy_iso", [("float*", "data0", N), ("float*", "data1", N)])
assert src is not None
print("=== kernel src ==="); print(src)

VARIANTS = [
    ("float4-768x256", src, (768, 1, 1), (256, 1, 1)),
]
# also a scalar-fallback variant generated inline
scalar_src = f'''extern "C" __global__ void __launch_bounds__(256) E_copy_iso_s(float* data0_{N}, float* data1_{N}) {{
  for (int i = (blockIdx.x*256)+threadIdx.x; i < {N}; i += gridDim.x*256) data0_{N}[i] = data1_{N}[i];
}}
'''
VARIANTS.append(("scalar-768x256", scalar_src, (768, 1, 1), (256, 1, 1)))
VARIANTS.append(("float4-192x256", src.replace("E_copy_iso", "E_copy_iso_f2"), (192, 1, 1), (256, 1, 1)))

x = np.arange(N, dtype=np.float32)
dst = np.zeros(N, dtype=np.float32)
for vname, vsrc, grid, local in VARIANTS:
    xb = dev.allocator.alloc(4 * N, BufferSpec()); dev.allocator._copyin(xb, memoryview(x.tobytes()).cast("B"))
    yb = dev.allocator.alloc(4 * N, BufferSpec()); dev.allocator._copyin(yb, memoryview(dst.tobytes()).cast("B"))
    try:
        from tinygrad.device import TinyELF
        from tinygrad import dtypes as _dt
        lib = rend.compiler.compile_cached(vsrc)
        fname = vsrc.split("void __launch_bounds__(256) ")[1].split("(")[0]
        elf = TinyELF(lib, fname, rend.target,
                      (("data0", 0, _dt.float, (N,)), ("data1", 1, _dt.float, (N,))), None)
        fn = dev.runtime(elf)
        import time
        t0 = time.perf_counter()
        fn(yb, xb, global_size=grid, local_size=local, vals=(), wait=True)
        dt = (time.perf_counter() - t0) * 1e3
        mv = memoryview(bytearray(4 * N)); dev.allocator._copyout(mv, yb)
        got = np.frombuffer(mv, dtype=np.float32)
        ok = np.array_equal(got, x)
        print(f"[{vname}] OK {dt:.2f}ms match={ok}")
    except Exception as e:
        print(f"[{vname}] FAIL {type(e).__name__}: {str(e)[:120]}")
    try: dev.allocator.free(xb, 4 * N, BufferSpec())
    except Exception: pass
    try: dev.allocator.free(yb, 4 * N, BufferSpec())
    except Exception: pass
    dev.synchronize()
print("DONE")
