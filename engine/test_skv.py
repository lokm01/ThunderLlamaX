# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W3 split-KV validation + bench.
MODE=py    : small-L numpy check (poison-first), aattn3-vs-trio at pos=2000 (relerr),
             T=1-vs-T=3 row bit-equality (Tier-1 kernel contract).
MODE=bench : k1s at CTXK=100352 KV streaming BW (variants via PROG env)."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)
MODE = os.getenv("MODE", "py")
S = int(os.getenv("S", "32"))

def load(nm):
  lib = open(f"{BASE}/{nm}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=nm, target=dev.renderer.target, signature=tuple()))

def relerr(a, b):
  d = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
  s = np.abs(b.astype(np.float64)).max()
  return d / max(s, 1e-30)

def mk_inputs(P, rng, rows=3, CTXK=2304, pos=2000):
  P.up("qrowA", rng.standard_normal(rows*12288).astype(np.float16))
  P.up("krowA", rng.standard_normal(rows*1024).astype(np.float16))
  P.up("vrowA", rng.standard_normal(rows*1024).astype(np.float16))
  P.up("qnwA", (1.0 + 0.1*rng.standard_normal(256)).astype(np.float32))
  P.up("knwA", (1.0 + 0.1*rng.standard_normal(256)).astype(np.float32))
  freqs = (1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
  P.up("freqsA", freqs)
  kv = (0.1*rng.standard_normal(2*4*CTXK*256)).astype(np.float16)
  kv.reshape(-1)[(2*4*pos*256):] = 7.7   # poison rows >= pos
  P.up("kvA", kv)
  P.up("posA", np.array([pos], dtype=np.int32))
  for nm, nb, dt, pv in [("qw3A", 3*24*256*4, np.float32, 7.7e31), ("qw1A", 24*256*4, np.float32, 7.7e31),
                         ("pm3A", 4*S*18*4, np.float32, 7.7e31), ("ps3A", 4*S*18*4, np.float32, 7.7e31),
                         ("pA3A", 4*S*18*256*4, np.float32, 7.7e31),
                         ("pm1A", 4*S*6*4, np.float32, 7.7e31), ("ps1A", 4*S*6*4, np.float32, 7.7e31),
                         ("pA1A", 4*S*6*256*4, np.float32, 7.7e31),
                         ("ao3A", 3*6144*2, np.float16, 7.7), ("ao1A", 6144*2, np.float16, 7.7)]:
    P.poison(nm, nb, dt, pv)
  dev.synchronize()

def numpy_ref(P, pos, rows=3, upto=None):
  qw = P.down("qw3A", (rows, 24, 256), np.float32)
  kv = P.down("kvA", (2, 4, 2304, 256), np.float16)
  qrow = P.down("qrowA", (rows*12288,), np.float16)
  out = np.zeros((rows, 24, 256))
  for h in range(24):
    kvh = h // 6
    for t in range(rows):
      n = min(pos + t + 1, upto if upto else 10**9)
      K = kv[0, kvh, :n, :].astype(np.float64)
      V = kv[1, kvh, :n, :].astype(np.float64)
      sc = K @ qw[t, h].astype(np.float64)
      sc = sc - sc.max()
      p = np.exp(sc); p /= p.sum()
      o = p @ V
      gf = qrow[t*12288 + h*512 + 256 : t*12288 + h*512 + 512].astype(np.float64)
      out[t, h] = o * (1.0/(1.0+np.exp(-gf)))
  return out

if MODE == "py":
  rng = np.random.default_rng(7)
  P = Bufs(); mk_inputs(P, rng)
  d, = (P.d,)
  pr = {n: load(n) for n in ("aattn3", "spk_pre3_2k", "spk_a3_2k", "spk_c3", "spk_pre1_2k", "spk_a1_2k", "spk_c1")}
  pos = 2000
  # ---- reference: old aattn3 on a fresh copy of kv ----
  kv0 = P.down("kvA", (2*4*2304*256,), np.float16).copy()
  pr["aattn3"](d["qrowA"], d["krowA"], d["vrowA"], d["qnwA"], d["knwA"], d["freqsA"], d["kvA"], d["posA"], d["ao3A"],
               global_size=(24,1,1), local_size=LS, wait=True)
  aoA = P.down("ao3A", (3, 6144), np.float16).copy()
  kvA = P.down("kvA", (2, 4, 2304, 256), np.float16).copy()
  print(f"[py] aattn3 done absmax={np.abs(aoA.astype(np.float32)).max():.3f}", flush=True)
  # ---- trio on a fresh copy ----
  P.up("kvA", kv0)
  P.poison("ao3A", 3*6144*2, np.float16, 7.7)
  dev.synchronize()
  pr["spk_pre3_2k"](d["qrowA"], d["krowA"], d["vrowA"], d["qnwA"], d["knwA"], d["freqsA"], d["kvA"], d["posA"], d["qw3A"],
                 global_size=(24,1,1), local_size=LS)
  pr["spk_a3_2k"](d["kvA"], d["qw3A"], d["posA"], d["pm3A"], d["ps3A"], d["pA3A"], global_size=(4*S,1,1), local_size=LS)
  pr["spk_c3"](d["pm3A"], d["ps3A"], d["pA3A"], d["qrowA"], d["ao3A"], global_size=(24,1,1), local_size=LS, wait=True)
  aoB = P.down("ao3A", (3, 6144), np.float16).copy()
  kvB = P.down("kvA", (2, 4, 2304, 256), np.float16).copy()
  # appended rows identical?
  append_same = bool((kvA[:, :, pos:pos+3, :] == kvB[:, :, pos:pos+3, :]).all())
  print(f"[py] trio: appended-KV bit-identical to aattn3: {append_same}", flush=True)
  print(f"[py] relerr(trio vs aattn3): {relerr(aoB, aoA):.3e}  (gate 1e-3)", flush=True)
  # ---- numpy reference from the trio's own qw + kv ----
  ref = numpy_ref(P, pos)
  print(f"[py] relerr(trio vs numpy-fp64): {relerr(aoB.astype(np.float32).reshape(3,24,256), ref):.3e}", flush=True)
  # ---- T=1 row-0 bit-equality (Tier-1 contract) ----
  P.up("qrowB", P.down("qrowA", (3*12288,), np.float16)[:12288].copy())
  P.up("krowB", P.down("krowA", (3*1024,), np.float16)[:1024].copy())
  P.up("vrowB", P.down("vrowA", (3*1024,), np.float16)[:1024].copy())
  P.up("kvA", kv0)
  P.poison("ao1A", 6144*2, np.float16, 7.7)
  dev.synchronize()
  pr["spk_pre1_2k"](d["qrowB"], d["krowB"], d["vrowB"], d["qnwA"], d["knwA"], d["freqsA"], d["kvA"], d["posA"], d["qw1A"],
                 global_size=(24,1,1), local_size=LS)
  pr["spk_a1_2k"](d["kvA"], d["qw1A"], d["posA"], d["pm1A"], d["ps1A"], d["pA1A"], global_size=(4*S,1,1), local_size=LS)
  pr["spk_c1"](d["pm1A"], d["ps1A"], d["pA1A"], d["qrowB"], d["ao1A"], global_size=(24,1,1), local_size=LS, wait=True)
  ao1 = P.down("ao1A", (6144,), np.float16).copy()
  qw1 = P.down("qw1A", (24, 256), np.float32).copy()
  qw3 = P.down("qw3A", (3, 24, 256), np.float32).copy()
  print(f"[py] qw1 == qw3[0] bitwise: {bool((qw1.view(np.uint16) == qw3[0].view(np.uint16)).all())}", flush=True)
  print(f"[py] ao1 == aoB[0] bitwise: {bool((ao1.view(np.uint16) == aoB[0].view(np.uint16)).all())}", flush=True)
  print("[py done]", flush=True)

elif MODE == "bench":
  CTXK = 100352
  PROG = os.getenv("PROG", "spk_a3_100k")
  GS = int(os.getenv("GS", str(4*S)))
  pos = int(os.getenv("POS", "97810"))
  REPS = int(os.getenv("REPS", "30"))
  rng = np.random.default_rng(3)
  P = Bufs()
  P.up("kvb", (0.1*rng.standard_normal(2*4*CTXK*256)).astype(np.float16))
  P.up("qwb", rng.standard_normal(3*24*256).astype(np.float32))
  P.up("posb", np.array([pos], dtype=np.int32))
  for nm, nb, dt, pv in [("pmb", 4*S*18*4, np.float32, 7.7e31), ("psb", 4*S*18*4, np.float32, 7.7e31),
                         ("pAb", 4*S*18*256*4, np.float32, 7.7e31)]:
    P.poison(nm, nb, dt, pv)
  dev.synchronize()
  print(f"[bench] {PROG} grid={GS} pos={pos} KV={2*4*CTXK*256*2/1e9:.3f} GB", flush=True)
  pr = load(PROG)
  args = (P.d["kvb"], P.d["qwb"], P.d["posb"], P.d["pmb"], P.d["psb"], P.d["pAb"])
  for _ in range(3): pr(*args, global_size=(GS,1,1), local_size=LS)
  dev.synchronize()
  t0 = time.perf_counter()
  for _ in range(REPS): pr(*args, global_size=(GS,1,1), local_size=LS)
  dev.synchronize()
  dt = (time.perf_counter() - t0) / REPS
  byt = 2*4*CTXK*256*2 * (pos/CTXK)  # read up to pos+3 only
  print(f"[bench] {PROG}: {dt*1e3:.3f} ms/launch -> {byt/dt/1e9:.1f} GB/s effective", flush=True)
