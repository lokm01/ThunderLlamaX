# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P5 pfa16 A/B: shipped (pfa16nw32_s32_100k.p4.bak) vs the DBUF-restructured
build (same name) at end-of-100k split lengths. Gates: pm/ps/pA BIT-IDENTICAL
(no arithmetic order change) + synced timing min-of-5 (async launch +
dev.synchronize; program objects hoisted)."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
S, CTXK, M = 32, 100352, 16
LS32 = (1024, 1, 1)
P = Bufs()
def prog(path, name):
  lib = open(path, "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=name, target=dev.renderer.target, signature=tuple()))

rng = np.random.default_rng(21)
P.up("kv8", rng.integers(0, 256, 2*4*CTXK*256, dtype=np.uint8))
P.up("sc8", (rng.standard_normal(64*CTXK) * 0.01).astype(np.float16))  # 8 groups x CTXK*8 halfs (half-unit stride)
P.up("qw16b", (rng.standard_normal(M*24*256) * 0.3).astype(np.float16))
P.up("pos_slot", np.array([CTXK - M], dtype=np.int32))
P.poison("pm", 4*S*96*4, np.float32, 7.7e31)
P.poison("ps", 4*S*96*4, np.float32, 7.7e31)
P.poison("pA", 4*S*96*256*4, np.float32, 7.7e31)
dev.synchronize()

args = (P.d["kv8"], P.d["sc8"], P.d["qw16b"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"])
G = (4*S, 1, 1)

def run_pair(path, tag):
  pr = prog(path, "pfa16")
  pr(*args, global_size=G, local_size=LS32); dev.synchronize()
  pm = P.down("pm", (4*S*96,), np.float32).copy()
  ps = P.down("ps", (4*S*96,), np.float32).copy()
  pA = P.down("pA", (4*S*96*256,), np.float32).copy()
  best = 1e9
  for _ in range(5):
    t0 = time.perf_counter(); pr(*args, global_size=G, local_size=LS32); dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  print(f"[{tag}] {best*1e3:7.3f} ms  (16-layer chunk attn est {best*1e3*16:7.2f} ms)", flush=True)
  return pm, ps, pA, best

pm0, ps0, pA0, t0 = run_pair(f"{BASE}/pfa16nw32_s32_100k.p4.bak", "SHIPPED")
pm1, ps1, pA1, t1 = run_pair(f"{BASE}/pfa16nw32_s32_100k.cubin", "P5-DBUF")
for nm, a, b in [("pm", pm0, pm1), ("ps", ps0, ps1), ("pA", pA0, pA1)]:
  nz = int((a != b).sum())
  fin = int(np.isfinite(a).sum())
  print(f"[bit] {nm}: mismatches {nz}/{a.size} (finite {fin}/{a.size})", flush=True)
print(f"[time] speedup {t0/t1:.2f}x", flush=True)
print("[test_p5a] BIT-IDENTICAL + WIN" if (pm0 == pm1).all() and (ps0 == ps1).all() and (pA0 == pA1).all() and t1 < t0 else "[test_p5a] CHECK RESULTS", flush=True)
