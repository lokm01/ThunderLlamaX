# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad import dtypes
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
dev = Device["NV"]
bufs = []
for i in range(14):
    b = dev.allocator.alloc(256, BufferSpec())
    dev.allocator._copyin(b, memoryview(np.full(64, float(i), dtype=np.float32).tobytes()).cast("B"))
    bufs.append(b)
dev.synchronize()
lib = open("~/tinygrad-metal/engine0/argtest.cubin","rb").read()
def mk(n, k): return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
for name, k in (("k6",6),("k8",8),("k10",10),("k12",12),("k14",14)):
    prg = mk(name, k)
    prg(*bufs[:k], global_size=(1,1,1), local_size=(256,1,1), wait=True)
    mv = memoryview(bytearray(4)); dev.allocator._copyout(mv, bufs[0])
    val = np.frombuffer(mv, dtype=np.float32)[0]
    print(f"[arg] {name} ({k} bufs) OK out={val}", flush=True)
print("[arg done]", flush=True)
