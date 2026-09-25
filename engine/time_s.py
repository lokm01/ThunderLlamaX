# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Time pfa16 at several S splits + correctness spot at S=best (min-of-10)."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1); LS1K = (1024, 1, 1)
M = 16; CTXK = 100352
SMAX = 256
P = Bufs()
rng = np.random.default_rng(5)
def prog(f, sym):
  return NVProgram(dev, TinyELF(lib=open(f"{BASE}/{f}.cubin","rb").read(), name=sym, target=dev.renderer.target, signature=tuple()))
P.up("qw16", (rng.standard_normal((M, 24*256)) * 0.1).astype(np.float16))
P.poison("pm16", 4*SMAX*96*4, np.float32, 7.7e31)
P.poison("ps16", 4*SMAX*96*4, np.float32, 7.7e31)
P.poison("pA16", 4*SMAX*96*256*4, np.float32, 7.7e31)
P.up("pos_slot", np.array([100320], dtype=np.int32))
P.up("kv", (rng.integers(0, 255, 2*4*CTXK*256)).astype(np.uint8))
P.up("sc", (rng.standard_normal(2*4*CTXK*8) * 0.02 + 0.02).astype(np.float16))
dev.synchronize()
d = P.d
for S in (32, 64, 128, 256):
  k1 = prog(f"pfa16nw32_s{S}_100k", "pfa16")
  k2 = prog(f"pfc16_s{S}", "pfc16")
  k1(d["kv"], d["sc"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"], global_size=(4*S,1,1), local_size=LS1K)
  dev.synchronize()
  best = 1e30
  for _ in range(10):
    t0 = time.perf_counter()
    k1(d["kv"], d["sc"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"], global_size=(4*S,1,1), local_size=LS1K)
    dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  gb = 2*4*CTXK*256 / best / 1e9
  print(f"S={S}: K1 {best*1e3:.3f} ms ({gb:.0f} GB/s KV)", flush=True)
print("[done]", flush=True)
