# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
BASE = "~/tinygrad-metal/engine0"
P = Bufs()
rng = np.random.default_rng(3)
A = (rng.standard_normal((16,16)).astype(np.float32)*0.5).astype(np.float16)
B = (rng.standard_normal((16,8)).astype(np.float32)*0.5).astype(np.float16)
P.up("A", A.reshape(-1)); P.up("B", B.reshape(-1)); P.poison("D", 16*8*4, np.float32, 7.7e31)
dev.synchronize()
lib = open(f"{BASE}/mma_micro.cubin","rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name="mma_micro", target=dev.renderer.target, signature=tuple()))
pr(P.d["A"], P.d["B"], P.d["D"], global_size=(1,1,1), local_size=(32,1,1)); dev.synchronize()
D = P.down("D", (16,8), np.float32)
ref = A.astype(np.float32) @ B.astype(np.float32)
err = np.abs(D - ref) / np.maximum(np.abs(ref), 1e-3)
print("max relerr", err.max(), "median", np.median(err))
bad = np.argwhere(err > 0.05)
print("bad cells (m,n):", bad[:10].tolist(), "count", len(bad))
