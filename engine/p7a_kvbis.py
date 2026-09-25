# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-A kv bisection: pure-stream vs +dp4a vs +sc, isolated runs.
usage: test kv variant NAME NOSC NODP [grid nthr]"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad import dtypes
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

BASE = "~/tinygrad-metal/engine0"
dev = Device["NV"]
P = Bufs()
rng = np.random.default_rng(7)
INT = (None, 4, dtypes.int32, ())
name = sys.argv[1]
grid, nthr = int(sys.argv[2]), int(sys.argv[3])
CTXK = 50176
ROWS = 8 * CTXK
slab = rng.integers(-128, 128, ROWS * 256, dtype=np.int8)
sc16 = rng.integers(0, 65536, ROWS * 8, dtype=np.uint16)
slab32 = slab.view(np.uint32)
u = np.arange(ROWS * 16, dtype=np.int64)
scidx = (u >> 4) * 8 + ((u & 15) >> 1)
dp = int(slab.astype(np.int64).sum()) & 0xFFFFFFFF
cs_full = (int(slab32.sum(dtype=np.uint64)) + dp + int(sc16[scidx].sum(dtype=np.uint64))) & 0xFFFFFFFF
cs_nosc = (int(slab32.sum(dtype=np.uint64)) + dp) & 0xFFFFFFFF
del u, scidx
P.up("kv", slab); P.up("sc", sc16); P.up("o", np.zeros(1024, np.uint32))
dev.synchronize()
lib = open(f"{BASE}/{name}.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name=name, target=dev.renderer.target,
                            signature=(INT, INT)))
print(f"[bisect] {name} g{grid} t{nthr}: launching", flush=True)
pr(P.d["kv"], P.d["sc"], P.d["o"], global_size=(grid,1,1), local_size=(nthr,1,1), vals=(grid, 1))
dev.synchronize()
print(f"[bisect] {name}: LAUNCH CLEAN", flush=True)
got = int(P.down("o", (1024,), np.uint32)[:grid].sum(dtype=np.uint64)) & 0xFFFFFFFF
exp = {"pure": None, "sc": cs_full, "dp": cs_nosc, "full": cs_full}.get(sys.argv[4], cs_full)
print(f"[bisect] {name} checksum got={got} exp={exp} -> {'PASS' if got == exp else 'DIFF'}", flush=True)
t0 = time.perf_counter()
pr(P.d["kv"], P.d["sc"], P.d["o"], global_size=(grid,1,1), local_size=(nthr,1,1), vals=(grid, 3))
dev.synchronize()
t = time.perf_counter() - t0
BYTES = ROWS * 256 + ROWS * 16
print(f"[bisect] {name}: bench 3pass {t*1e3:.3f} ms -> {BYTES*3/t/1e9:.1f} GB/s", flush=True)
