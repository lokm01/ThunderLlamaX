# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""D0: the SMEM-LUT discriminator — ffn8v_3 (global grid gathers) vs ffn8l_3
(smem-staged grid LUT). Bit-identity on real weights + synced min-of-10 bench
(the P9 discipline) + the W2D-style mean-of-30 for continuity.
KILL-LINE: <1.25x on the synced bench -> the LUT class is dead."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256,1,1)
DIM, FFN_N = 5120, 17408
P = Bufs()
pr = {}
for n in ["ffn8v_3", "ffn8l_3"]:
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
P.up("gridf", iq3_grid_f32())
ds, infos = parse_gguf()
SZ = {}
def upk(key, nm):
  arr = np.ascontiguousarray(np.load(f"{BASE}/packed/{key}.npy"))
  P.up(nm, arr); SZ[nm] = arr.nbytes
upk("fg8", "fg8"); upk("fu8", "fu8")
rng = np.random.default_rng(7)
P.up("hhx3", (rng.standard_normal(3*DIM)*0.3).astype(np.float16))
dev.synchronize(); d = P.d
bf = SZ["fg8"] + SZ["fu8"]
print(f"[d0] weights up ({bf/1e6:.1f} MB per launch)", flush=True)

def run(nm):
  pr[nm](d["fg8"], d["fu8"], d["gridf"], d["hhx3"], d["gact3"], global_size=(2176,1,1), local_size=LS, wait=True)

# --- bit-identity (poison-first, args after up, assert active) ---
o = {}
for nm in ("ffn8v_3", "ffn8l_3"):
  P.poison("gact3", 3*FFN_N*2, np.float16, 7.7)
  run(nm)
  o[nm] = P.down("gact3", (3*FFN_N,), np.float16).copy()
  assert np.nanmax(np.abs(o[nm].astype(np.float32))) < 1e3, f"{nm} output looks poison"
same = np.array_equal(o["ffn8v_3"], o["ffn8l_3"])
print(f"[d0] BIT-IDENTICAL ffn8l_3 vs ffn8v_3: {same}", flush=True)
if not same:
  diff = np.nonzero(o["ffn8v_3"] != o["ffn8l_3"])[0]
  print(f"[d0] MISMATCH count {len(diff)} / {o['ffn8v_3'].size} first {diff[:8]}")
  sys.exit(1)

# --- synced min-of-10 (P9) ---
mins = {}
for nm in ("ffn8v_3", "ffn8l_3"):
  run(nm)  # warm
  ts = []
  for _ in range(10):
    t0 = time.perf_counter(); run(nm); ts.append(time.perf_counter()-t0)
  mins[nm] = min(ts)
  print(f"[d0] {nm}: min-of-10 {mins[nm]*1e6:8.1f}us  ({bf/mins[nm]/1e9:6.1f} GB/s)", flush=True)
x = mins["ffn8v_3"]/mins["ffn8l_3"]
print(f"[d0] SPEEDUP min-of-10: {x:.3f}x  (kill-line 1.25x)", flush=True)

# --- mean-of-30 synced loop (W2D continuity) ---
for nm in ("ffn8v_3", "ffn8l_3"):
  t0 = time.perf_counter()
  for _ in range(30): run(nm)
  dt = (time.perf_counter()-t0)/30
  print(f"[d0] {nm}: mean-of-30 {dt*1e6:8.1f}us  ({bf/dt/1e9:6.1f} GB/s)", flush=True)
print("[d0] DONE", flush=True)
