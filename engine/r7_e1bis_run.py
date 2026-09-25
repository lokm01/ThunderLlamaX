# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
from tinygrad.device import Device, TinyELF, BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
dev = Device["NV"]
GRID, SLICE = 82, 393216
BYTES = GRID*SLICE*16
src = dev.allocator.alloc(BYTES, BufferSpec())
outb = dev.allocator.alloc(GRID*16*4, BufferSpec(cpu_access=True, nolru=True))
lib = open("~/tinygrad-metal/engine0/r7_e1bis.cubin", "rb").read()
for kn in ("b1_basic", "b2_twoargs", "b3_ldg8", "b4_full"):
    prg = NVProgram(dev, TinyELF(lib=lib, name=kn, target=dev.renderer.target, signature=tuple()))
    print(f"[bis] {kn}: regs={prg.regs_usage}", flush=True)
    t0 = time.perf_counter()
    if kn == "b1_basic":
        prg(outb, global_size=(GRID,1,1), local_size=(256,1,1), wait=True)
    else:
        prg(src, outb, global_size=(GRID,1,1), local_size=(256,1,1), wait=True)
    print(f"[bis] {kn}: OK {(time.perf_counter()-t0)*1e3:.3f}ms", flush=True)
print("[bis] DONE")
