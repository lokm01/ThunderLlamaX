# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2G L3 standalone validation: half2-core draft GEMVs vs originals on the
REAL packed weights (poison-first, DISTINCT output buffers, synced bench).
Expect BIT-IDENTICAL outputs (order-preserving port)."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np, time
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
DPACK = f"{BASE}/draft_pack"
LS = (256, 1, 1)
rng = np.random.default_rng(11)

P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

# weights
for nm in ("d_eh", "d_q", "d_k", "d_v", "d_o", "d_fg", "d_fu", "d_fd"):
  P.up(nm, np.load(f"{DPACK}/{nm}.npy"))
# inputs
P.up("cat", (rng.standard_normal(10240).astype(np.float32) * 0.3).astype(np.float16))
P.up("xh", (rng.standard_normal(5120).astype(np.float32) * 0.3).astype(np.float16))
P.up("ao", (rng.standard_normal(6144).astype(np.float32) * 0.3).astype(np.float16))
P.up("hhx", (rng.standard_normal(5120).astype(np.float32) * 0.3).astype(np.float16))
P.up("gact_in", (rng.standard_normal(17408).astype(np.float32) * 0.3).astype(np.float16))
P.up("hhf", (rng.standard_normal(5120).astype(np.float32) * 0.7))
P.up("zed", np.zeros(5120, dtype=np.float32))
dev.synchronize()
print("[fx] weights+inputs up", flush=True)

# (name, args builder, out specs [(bufname, numel, dtype, poisonval)], grid)
JOBS = [
  ("ehproj",  lambda t: (P.d["d_eh"], P.d["cat"], P.d["zed"], P.d[f"o{t}"]),  640,
   [("o", 5120, np.float32, 7.7e31)]),
  ("dq",      lambda t: (P.d["d_q"], P.d["xh"], P.d["hhf"], P.d[f"o{t}"]),    1536,
   [("o", 12288, np.float16, 7.7)]),
  ("dkv",     lambda t: (P.d["d_k"], P.d["d_v"], P.d["xh"], P.d[f"k{t}"], P.d[f"v{t}"]), 256,
   [("k", 1024, np.float16, 7.7), ("v", 1024, np.float16, 7.7)]),
  ("doproj",  lambda t: (P.d["d_o"], P.d["ao"], P.d["hhf"], P.d[f"o{t}"]),    640,
   [("o", 5120, np.float16, 7.7)]),
  ("dfgu",    lambda t: (P.d["d_fg"], P.d["d_fu"], P.d["hhx"], P.d[f"o{t}"]), 2176,
   [("o", 17408, np.float16, 7.7)]),
  ("ddown",   lambda t: (P.d["d_fd"], P.d["gact_in"], P.d["hhf"], P.d[f"o{t}"]), 640,
   [("o", 5120, np.float32, 7.7e31)]),
]

for base, argb, grid, outs in JOBS:
  new = base + "h"
  res = {}
  for tag, nm in (("A", base), ("B", new)):
    for on, n, dt, pv in outs:
      P.poison(f"{on}{tag}", n * (2 if dt == np.float16 else 4), dt, pv)
    dev.synchronize()
    pr = prog(nm)
    args = argb(tag)
    pr(*args, global_size=(grid, 1, 1), local_size=LS); dev.synchronize()
    t0 = time.perf_counter()
    for r in range(100): pr(*args, global_size=(grid, 1, 1), local_size=LS)
    dev.synchronize()
    dtm = (time.perf_counter() - t0) / 100
    res[tag] = (dtm, {on: P.down(f"{on}{tag}", (n,), dt) for on, n, dt, pv in outs})
  dts = {on: (res["A"][1][on], res["B"][1][on]) for on, *_ in outs}
  bit = all(bool((a == b).all()) for a, b in dts.values())
  print(f"[{base}] {base} {res['A'][0]*1e3:.3f} ms vs {new} {res['B'][0]*1e3:.3f} ms "
        f"delta {(res['B'][0]-res['A'][0])*1e3:+.4f} ms  BIT-IDENTICAL: {bit}", flush=True)
print("[dh validation done]", flush=True)
