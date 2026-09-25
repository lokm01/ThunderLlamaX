# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2E KV8 standalone validation (100k dims, poison-first):
 (1) K1: fp16 spk_g4nw32a3_100k vs int8 spk_g4nw32qa3_100k on the SAME KV
     content (int8 quantized from the fp16 fixture) -> pm/ps/pA relerr
     (expect <=2e-2 storage-quant class).
 (2) KPRE: fp16 spk_pre3_100k vs int8 spk_pre3q_100k on the same q/k/v rows:
     qw must be BIT-IDENTICAL; stored int8 KV dequantized must match the fp16
     stores within scale/2 (+fp16-scale rounding)."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
CTXK, S, RM, LS = 100352, 256, 3, (256, 1, 1)
LS32 = (1024, 1, 1)
POS = 97810

P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

rng = np.random.default_rng(7)

# ---------------- fixture: fp16 kv, int8+sc derived, qw ----------------
print("[fx] generating kv fixture (411MB fp16)...", flush=True)
kv16 = (rng.standard_normal((2, 4, CTXK, 256)).astype(np.float32) * 0.5).astype(np.float16)
g = kv16.reshape(2, 4, CTXK, 8, 32)
amax = np.abs(g).max(axis=-1)                       # (2,4,CTXK,8) fp16
sc = (np.maximum(amax, 1e-8) * (1.0/127.0)).astype(np.float16)
q = (np.clip(np.rint(g.astype(np.float32) / sc.astype(np.float32)[..., None]), -127, 127) + 128).astype(np.uint8)
kv8 = q.reshape(2, 4, CTXK, 256)
del g
qw = (rng.standard_normal((RM*24, 256)).astype(np.float32) * 0.05)
P.up("kv16", kv16); P.up("kv8", kv8.reshape(-1)); P.up("sc", sc.reshape(-1)); P.up("qw", qw)
P.up("pos_slot", np.array([POS], dtype=np.int32))
P.poison("pm", 4*S*6*RM*4, np.float32, 7.7e31)
P.poison("ps", 4*S*6*RM*4, np.float32, 7.7e31)
P.poison("pA", 4*S*6*RM*256*4, np.float32, 7.7e31)
dev.synchronize()
del kv16, q, kv8
print("[fx] uploaded", flush=True)

def poison_partials():
  P.up("pm", np.full(4*S*6*RM, 7.7e31, dtype=np.float32))
  P.up("ps", np.full(4*S*6*RM, 7.7e31, dtype=np.float32))
  P.up("pA", np.full(4*S*6*RM*256, 7.7e31, dtype=np.float32))
  dev.synchronize()

def run_k1(name, kvbuf, scbuf, is8, reps=5, sync_each=True):
  pr = prog(name)
  import time
  args8 = (P.d[kvbuf], P.d[scbuf], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"])
  args16 = (P.d[kvbuf], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"])
  t0 = time.perf_counter()
  for r in range(reps):
    pr(*(args8 if is8 else args16), global_size=(4*S, 1, 1), local_size=LS32)
    if sync_each: dev.synchronize()
  if not sync_each: dev.synchronize()
  dt = (time.perf_counter() - t0) / reps
  pm = P.down("pm", (4*S*6*RM,), np.float32)
  ps = P.down("ps", (4*S*6*RM,), np.float32)
  pA = P.down("pA", (4*S*6*RM*256,), np.float32)
  return dt, pm, ps, pA

# ---------------- (1) K1 fp16 vs int8 ----------------
poison_partials()
dt16, pm16, ps16, pA16 = run_k1("spk_g4nw32a3_100k", "kv16", "qw", False, 1)   # fp16: no sc arg
poison_partials()
dt8, pm8, ps8, pA8 = run_k1("spk_g4nw32qa3_100k", "kv8", "sc", True)
print(f"[k1] fp16 {dt16*1e3:.2f} ms  int8 {dt8*1e3:.2f} ms  speedup {dt16/dt8:.2f}x", flush=True)
act = ps16 > 1e-20                                    # splits with real content
print(f"[k1] active splits {int(act.sum())}/{act.size}", flush=True)
for nm, a, b in (("pm", pm16, pm8), ("ps", ps16, ps8)):
  err = np.abs(b[act] - a[act]) / np.maximum(np.abs(a[act]), 1e-6)
  print(f"[k1] {nm} relerr max {err.max():.3e} mean {err.mean():.3e}", flush=True)
ra = np.abs(pA8.reshape(-1, 4*S*6*RM) - pA16.reshape(-1, 4*S*6*RM))
ref2 = np.abs(pA16.reshape(-1, 4*S*6*RM))
num = np.linalg.norm(ra[:, act], axis=1); den = np.linalg.norm(ref2[:, act], axis=1)
ok = den > 0
print(f"[k1] pA F-norm relerr (per 256-row) max {np.max(num[ok]/den[ok]):.3e} median {np.median(num[ok]/den[ok]):.3e}", flush=True)

# ---------------- (2) KPRE fp16 vs int8 ----------------
print("[pre] fixtures...", flush=True)
qrow = (rng.standard_normal((RM*12288)).astype(np.float32) * 0.3).astype(np.float16)
krow = (rng.standard_normal((RM*1024)).astype(np.float32) * 0.3).astype(np.float16)
vrow = (rng.standard_normal((RM*1024)).astype(np.float32) * 0.3).astype(np.float16)
qnw = (1.0 + 0.05*rng.standard_normal(256)).astype(np.float32)
knw = (1.0 + 0.05*rng.standard_normal(256)).astype(np.float32)
freqs = (1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
P.up("qrow", qrow); P.up("krow", krow); P.up("vrow", vrow)
P.up("qnw", qnw); P.up("knw", knw); P.up("freqs", freqs)
P.poison("qwA", RM*24*256*4, np.float32, 7.7e31)
P.poison("qwB", RM*24*256*4, np.float32, 7.7e31)
# fresh KV targets: zero-filled so writes are obvious; only rows pos..pos+2 written
print("[pre] zero kv targets...", flush=True)
P.up("kv16o", np.zeros(2*4*CTXK*256, dtype=np.float16))
P.up("kv8o", np.zeros(2*4*CTXK*256, dtype=np.uint8))
P.up("sco", np.zeros(2*4*CTXK*8, dtype=np.float16))
dev.synchronize()
print("[pre] launching fp16 pre3...", flush=True)

prf = prog("spk_pre3_100k")
prf(P.d["qrow"], P.d["krow"], P.d["vrow"], P.d["qnw"], P.d["knw"], P.d["freqs"],
    P.d["kv16o"], P.d["pos_slot"], P.d["qwA"], global_size=(24, 1, 1), local_size=LS)
dev.synchronize()
print("[pre] launching int8 pre3q...", flush=True)
prq = prog("spk_pre3q_100k")
prq(P.d["qrow"], P.d["krow"], P.d["vrow"], P.d["qnw"], P.d["knw"], P.d["freqs"],
    P.d["kv8o"], P.d["sco"], P.d["pos_slot"], P.d["qwB"], global_size=(24, 1, 1), local_size=LS)
dev.synchronize()
print("[pre] downs...", flush=True)

qwA = P.down("qwA", (RM*24*256,), np.float32)
qwB = P.down("qwB", (RM*24*256,), np.float32)
print(f"[pre] qw bit-identical: {bool((qwA == qwB).all())} (nonzero {int((qwA != 0).sum())})", flush=True)

kvo = P.down("kv16o", (2, 4, CTXK, 256), np.float16)
kv8o = P.down("kv8o", (2, 4, CTXK, 256), np.uint8)
sco = P.down("sco", (2, 4, CTXK, 8), np.float16)
for t in range(3):
  ref = kvo[:, :, POS+t, :].astype(np.float32)
  got = kv8o[:, :, POS+t, :].astype(np.float32) * sco[:, :, POS+t, :][:, :, :, None].astype(np.float32).repeat(32, axis=3).reshape(2,4,256)
  # rebuild scale-expanded properly
  scx = np.repeat(sco[:, :, POS+t, :].astype(np.float32), 32, axis=2)   # (2,4,256)
  got = (kv8o[:, :, POS+t, :].astype(np.float32) - 128.0) * scx
  err = np.abs(got - ref)
  bound = 0.5 * scx + 1e-7
  bad = int((err > bound + 1e-6).sum())
  print(f"[pre] t={t}: dequant max err {err.max():.3e} (max bound {bound.max():.3e}), out-of-bound {bad}/2048, scale range [{sco[:,:,POS+t,:].min():.2e},{sco[:,:,POS+t,:].max():.2e}]", flush=True)
# untouched rows stay zero
print(f"[pre] untouched rows zero: {bool((kv8o[:, :, :POS, :] == 0).all())}", flush=True)
print("[kv8 validation done] (biased-uint8 encode)", flush=True)
