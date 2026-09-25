# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
P = Bufs()
rng = np.random.default_rng(3)
P.up("src", rng.integers(0, 2**32, 1<<18, dtype=np.uint32))
P.up("out", np.zeros(4096, dtype=np.uint32))
dev.synchronize()
lib = open("~/tinygrad-metal/engine0/r7_e1bis5.cubin", "rb").read()
for kn, gsz in (("b6_ld4", 8), ("b7_ld8", 8), ("b5_ld1", 8), ("b5_ld1", 1)):
    prg = NVProgram(dev, TinyELF(lib=lib, name=kn, target=dev.renderer.target, signature=tuple()))
    try:
        t0 = time.perf_counter()
        prg(P.d["src"], P.d["out"], global_size=(gsz,1,1), local_size=(256,1,1), wait=True)
        print(f"[bis5] {kn} grid{gsz}: OK", flush=True)
    except Exception as e:
        print(f"[bis5] {kn} grid{gsz}: FAULT", flush=True)
        break
print("[bis5] DONE")
