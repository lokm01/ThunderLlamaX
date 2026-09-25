# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2G HMMA standalone validation (100k dims, poison-first, DISTINCT output
buffers): canonical scalar spk_g4nw32qh{3,1}p_100k vs tensor-core
spk_g4nw32hm{3,1}_100k. Expect Tier-2/3 relerr (fp16 inputs, fp32 acc, different
reduction order; BETTER precision class than the scalar fp16-chunk accs) +
synced bench."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np, time
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
CTXK, S, LS32 = 100352, 256, (1024, 1, 1)
POS = 97810

P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

rng = np.random.default_rng(7)
print("[fx] generating kv fixture...", flush=True)
kv16 = (rng.standard_normal((2, 4, CTXK, 256)).astype(np.float32) * 0.5).astype(np.float16)
g = kv16.reshape(2, 4, CTXK, 8, 32)
amax = np.abs(g).max(axis=-1)
sc = (np.maximum(amax, 1e-8) * (1.0/127.0)).astype(np.float16)
q = (np.clip(np.rint(g.astype(np.float32) / sc.astype(np.float32)[..., None]), -127, 127) + 128).astype(np.uint8)
del g, kv16
P.up("kv8", q.reshape(-1)); P.up("sc", sc.reshape(-1))
P.up("pos_slot", np.array([POS], dtype=np.int32))
dev.synchronize()
del q, sc
print("[fx] uploaded", flush=True)

def run_rows(rm, names, reps=8):
  qw = (rng.standard_normal((rm*24, 256)).astype(np.float32) * 0.05)
  qw16 = qw.astype(np.float16)
  P.up(f"qw16_{rm}", qw16); dev.synchronize(); del qw, qw16
  n6 = 4*S*6*rm
  for tag in "AB":
    P.poison(f"pm{tag}{rm}", n6*4, np.float32, 7.7e31)
    P.poison(f"ps{tag}{rm}", n6*4, np.float32, 7.7e31)
    P.poison(f"pA{tag}{rm}", n6*256*4, np.float32, 7.7e31)
  dev.synchronize()
  res = []
  for i, n in enumerate(names):
    tag = "AB"[i]
    pr = prog(n)
    P.up(f"pm{tag}{rm}", np.full(n6, 7.7e31, dtype=np.float32))
    P.up(f"ps{tag}{rm}", np.full(n6, 7.7e31, dtype=np.float32))
    P.up(f"pA{tag}{rm}", np.full(n6*256, 7.7e31, dtype=np.float32))
    dev.synchronize()
    args = (P.d["kv8"], P.d["sc"], P.d[f"qw16_{rm}"], P.d["pos_slot"],
            P.d[f"pm{tag}{rm}"], P.d[f"ps{tag}{rm}"], P.d[f"pA{tag}{rm}"])
    for r in range(2): pr(*args, global_size=(4*S,1,1), local_size=LS32); dev.synchronize()
    t0 = time.perf_counter()
    for r in range(reps): pr(*args, global_size=(4*S,1,1), local_size=LS32); dev.synchronize()
    dt = (time.perf_counter()-t0)/reps
    pm = P.down(f"pm{tag}{rm}", (n6,), np.float32); ps = P.down(f"ps{tag}{rm}", (n6,), np.float32)
    pA = P.down(f"pA{tag}{rm}", (n6*256,), np.float32)
    res.append((dt, pm, ps, pA))
    print(f"[rows{rm}] {n}: {dt*1e3:.3f} ms", flush=True)
  return res

for rm, canon, new in ((3, "spk_g4nw32qh3p_100k", "spk_g4nw32hm3_100k"),
                       (1, "spk_g4nw32qh1p_100k", "spk_g4nw32hm1_100k")):
  (dt0, pm0, ps0, pA0), (dt1, pm1, ps1, pA1) = run_rows(rm, (canon, new))
  act = (ps0 > 1e-20) & (ps1 > 1e-20)
  for nm_, v0, v1 in (("pm", pm0, pm1), ("ps", ps0, ps1), ("pA", pA0, pA1)):
    pz0 = int((v0 > 1e30).sum()); pz1 = int((v1 > 1e30).sum())
    eq = int((v0 == v1).sum())
    print(f"[rows{rm}] RAW {nm_}: poison0={pz0}/{v0.size} poison1={pz1}/{v1.size} eq={eq}/{v0.size} v0[:3]={v0[:3]} v1[:3]={v1[:3]}", flush=True)
  print(f"[rows{rm}] {canon} {dt0*1e3:.3f} vs {new} {dt1*1e3:.3f} ms  delta {(dt1-dt0)*1e3:+.3f} ms", flush=True)
  print(f"[rows{rm}] active splits both {int(act.sum())}/{act.size}", flush=True)
  for nm, a, b in (("pm", pm0, pm1), ("ps", ps0, ps1)):
    err = np.abs(b[act]-a[act]) / np.maximum(np.abs(a[act]), 1e-6)
    print(f"[rows{rm}] {nm} relerr max {err.max():.3e} median {np.median(err):.3e}", flush=True)
  a3 = pA0.reshape(-1, 256); b3 = pA1.reshape(-1, 256)
  num = np.linalg.norm((b3 - a3)[act], axis=1); den = np.linalg.norm(a3[act], axis=1)
  ok = den > 0
  print(f"[rows{rm}] pA F-norm relerr max {np.max(num[ok]/den[ok]):.3e} median {np.median(num[ok]/den[ok]):.3e}", flush=True)
print("[hm validation done]", flush=True)
