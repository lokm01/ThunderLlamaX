# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2D L1: validate T-layout kernels BIT-EXACT vs m3.cu originals on real weights
(poison-first), then bench sync+pipe."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256,1,1)
DIM, FFN_N = 5120, 17408
REPS = 30
P = Bufs()
pr = {}
for n in ["ffn8_3","down8_3","op38_3","ffn8t_3","down8t_3","op38t_3"]:
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
P.up("gridf", iq3_grid_f32())
rng = np.random.default_rng(1234)
def mk(name, arr, dt):
  a = np.asarray(arr, dtype=dt)
  P.up(name, a)
  return a.nbytes
# weights: old pack + new pack for blk8
sz = {}
for nm, f in [("fg8","fg8"),("fu8","fu8"),("fd8","fd8"),("o8","out8")]:
  sz[nm] = mk(nm+"_old", np.load(f"{BASE}/packed/{f}.npy"), np.uint8)
  sz[nm] = mk(nm+"_new", np.load(f"{BASE}/packed_t/{f}.npy"), np.uint8)
# inputs (deterministic, in-range halves)
hhx = (rng.standard_normal(3*DIM)*0.3).astype(np.float16)
gact = (rng.standard_normal(3*FFN_N)*0.3).astype(np.float16)
z3 = (rng.standard_normal(3*6144)*0.3).astype(np.float16)
hh3f = (rng.standard_normal(3*DIM)*0.5).astype(np.float32)
P.up("hhx3", hhx); P.up("gact3_in", gact); P.up("z3", z3); P.up("hh3f", hh3f)
P.poison("gactA", 3*FFN_N*2, np.float16, 0.5)
P.poison("y3", 3*DIM*4, np.float32, 0.5)
P.poison("ao3", 3*DIM*2, np.float16, 0.5)
dev.synchronize(); d = P.d

def run_old_new(kold, knew, args_old, args_new, outnm, out_bytes, dt, cmp_fn=None):
  # old
  P.poison(outnm, out_bytes, dt, 7.7 if dt==np.float16 else 7.7e31)
  pr[kold](*args_old, global_size=(2176,1,1) if "ffn" in kold else (640,1,1), local_size=LS, wait=True)
  old = P.down(outnm, (out_bytes//np.dtype(dt).itemsize,), dt)
  # new
  P.poison(outnm, out_bytes, dt, 7.7 if dt==np.float16 else 7.7e31)
  pr[knew](*args_new, global_size=(2176,1,1) if "ffn" in knew else (640,1,1), local_size=LS, wait=True)
  new = P.down(outnm, (out_bytes//np.dtype(dt).itemsize,), dt)
  ok = np.array_equal(old, new)
  print(f"[t] {knew:12s} vs {kold:10s}: {'BIT-IDENTICAL' if ok else 'MISMATCH maxdiff %.3e' % np.max(np.abs(old.astype(np.float64)-new.astype(np.float64)))}", flush=True)
  return ok

ok1 = run_old_new("ffn8_3","ffn8t_3",
  (d["fg8_old"], d["fu8_old"], d["gridf"], d["hhx3"], d["gactA"]),
  (d["fg8_new"], d["fu8_new"], d["gridf"], d["hhx3"], d["gactA"]),
  "gactA", 3*FFN_N*2, np.float16)
ok2 = run_old_new("down8_3","down8t_3",
  (d["fd8_old"], d["gridf"], d["gact3_in"], d["hh3f"], d["y3"]),
  (d["fd8_new"], d["gridf"], d["gact3_in"], d["hh3f"], d["y3"]),
  "y3", 3*DIM*4, np.float32)
ok3 = run_old_new("op38_3","op38t_3",
  (d["o8_old"], d["gridf"], d["z3"], d["ao3"]),
  (d["o8_new"], d["gridf"], d["z3"], d["ao3"]),
  "ao3", 3*DIM*2, np.float16)
print("[t] ALL PASS" if (ok1 and ok2 and ok3) else "[t] FAIL", flush=True)
if not (ok1 and ok2 and ok3): sys.exit(1)

def bench(name, nbytes, launch):
  launch(False); dev.synchronize()
  t0 = time.perf_counter()
  for _ in range(REPS): launch(True)
  dt = (time.perf_counter()-t0)/REPS
  t0 = time.perf_counter()
  for _ in range(REPS): launch(False)
  dev.synchronize()
  dp = (time.perf_counter()-t0)/REPS
  print(f"[t] {name:12s} sync {dt*1e6:8.1f}us pipe {dp*1e6:8.1f}us  GB/s sync {nbytes/dt/1e9:6.1f} pipe {nbytes/dp/1e9:6.1f} ({nbytes/1e6:.1f} MB)", flush=True)

P.poison("gactA", 3*FFN_N*2, np.float16, 0.5)
P.poison("y3", 3*DIM*4, np.float32, 0.5)
P.poison("ao3", 3*DIM*2, np.float16, 0.5)
print("[t] ---- OLD ----", flush=True)
bench("ffn8_3", sz["fg8"]+sz["fu8"], lambda w: pr["ffn8_3"](d["fg8_old"], d["fu8_old"], d["gridf"], d["hhx3"], d["gactA"], global_size=(2176,1,1), local_size=LS, wait=w))
bench("down8_3", sz["fd8"], lambda w: pr["down8_3"](d["fd8_old"], d["gridf"], d["gact3_in"], d["hh3f"], d["y3"], global_size=(640,1,1), local_size=LS, wait=w))
bench("op38_3", sz["o8"], lambda w: pr["op38_3"](d["o8_old"], d["gridf"], d["z3"], d["ao3"], global_size=(640,1,1), local_size=LS, wait=w))
print("[t] ---- NEW (T) ----", flush=True)
bench("ffn8t_3", sz["fg8"]+sz["fu8"], lambda w: pr["ffn8t_3"](d["fg8_new"], d["fu8_new"], d["gridf"], d["hhx3"], d["gactA"], global_size=(2176,1,1), local_size=LS, wait=w))
bench("down8t_3", sz["fd8"], lambda w: pr["down8t_3"](d["fd8_new"], d["gridf"], d["gact3_in"], d["hh3f"], d["y3"], global_size=(640,1,1), local_size=LS, wait=w))
bench("op38t_3", sz["o8"], lambda w: pr["op38t_3"](d["o8_new"], d["gridf"], d["z3"], d["ao3"], global_size=(640,1,1), local_size=LS, wait=w))
print("[t] DONE", flush=True)
