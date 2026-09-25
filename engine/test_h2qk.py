# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2F QH standalone validation (100k dims, poison-first):
 (1) K1: spk_g4nw32qa3_100k (fp32 q) vs spk_g4nw32qh3_100k (half2 QK, q from
     qw16) on the SAME kv8/sc content -> pm/ps relerr + pA F-norm relerr
     (expect ~1e-3 fp16-chunk-accumulate class) + synced bench of both.
 (2) KPRE: spk_pre3q_100k vs spk_pre3qh_100k on same rows: qw BIT-IDENTICAL,
     qw16 == fp16(qw) exactly, int8/sc stores BIT-IDENTICAL."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np, time
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
print("[fx] generating kv fixture (411MB fp16)...", flush=True)
kv16 = (rng.standard_normal((2, 4, CTXK, 256)).astype(np.float32) * 0.5).astype(np.float16)
g = kv16.reshape(2, 4, CTXK, 8, 32)
amax = np.abs(g).max(axis=-1)
sc = (np.maximum(amax, 1e-8) * (1.0/127.0)).astype(np.float16)
q = (np.clip(np.rint(g.astype(np.float32) / sc.astype(np.float32)[..., None]), -127, 127) + 128).astype(np.uint8)
del g, kv16
qw = (rng.standard_normal((RM*24, 256)).astype(np.float32) * 0.05)
qw16 = qw.astype(np.float16)
P.up("kv8", q.reshape(-1)); P.up("sc", sc.reshape(-1)); P.up("qw", qw); P.up("qw16", qw16)
P.up("pos_slot", np.array([POS], dtype=np.int32))
P.poison("pm", 4*S*6*RM*4, np.float32, 7.7e31)
P.poison("ps", 4*S*6*RM*4, np.float32, 7.7e31)
P.poison("pA", 4*S*6*RM*256*4, np.float32, 7.7e31)
dev.synchronize()
del q, qw, qw16
print("[fx] uploaded", flush=True)

def poison_partials():
  P.up("pm", np.full(4*S*6*RM, 7.7e31, dtype=np.float32))
  P.up("ps", np.full(4*S*6*RM, 7.7e31, dtype=np.float32))
  P.up("pA", np.full(4*S*6*RM*256, 7.7e31, dtype=np.float32))
  dev.synchronize()

def run_k1(name, qbuf, reps=8):
  pr = prog(name)
  args = (P.d["kv8"], P.d["sc"], P.d[qbuf], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"])
  for r in range(2): pr(*args, global_size=(4*S,1,1), local_size=LS32); dev.synchronize()
  t0 = time.perf_counter()
  for r in range(reps): pr(*args, global_size=(4*S,1,1), local_size=LS32); dev.synchronize()
  dt = (time.perf_counter()-t0)/reps
  pm = P.down("pm", (4*S*6*RM,), np.float32); ps = P.down("ps", (4*S*6*RM,), np.float32)
  pA = P.down("pA", (4*S*6*RM*256,), np.float32)
  return dt, pm, ps, pA

poison_partials(); dt0, pm0, ps0, pA0 = run_k1("spk_g4nw32qa3_100k", "qw", 4)
poison_partials(); dt1, pm1, ps1, pA1 = run_k1("spk_g4nw32qh3_100k", "qw16")
print(f"[k1] fp32q {dt0*1e3:.3f} ms  half2q {dt1*1e3:.3f} ms  delta {(dt0-dt1)*1e3:+.3f} ms", flush=True)
act = ps0 > 1e-20
print(f"[k1] active splits {int(act.sum())}/{act.size}", flush=True)
for nm, a, b in (("pm", pm0, pm1), ("ps", ps0, ps1)):
  err = np.abs(b[act]-a[act]) / np.maximum(np.abs(a[act]), 1e-6)
  print(f"[k1] {nm} relerr max {err.max():.3e} mean {err.mean():.3e}", flush=True)
ra = np.abs(pA1.reshape(-1,4*S*6*RM) - pA0.reshape(-1,4*S*6*RM))
ref2 = np.abs(pA0.reshape(-1,4*S*6*RM))
num = np.linalg.norm(ra[:, act], axis=1); den = np.linalg.norm(ref2[:, act], axis=1)
ok = den > 0
print(f"[k1] pA F-norm relerr max {np.max(num[ok]/den[ok]):.3e} median {np.median(num[ok]/den[ok]):.3e}", flush=True)

# ---------------- (2) KPRE q vs qh ----------------
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
P.poison("qw16B", RM*24*256*2, np.float16, 7.7)
P.up("kv8A", np.zeros(2*4*CTXK*256, dtype=np.uint8)); P.up("scA", np.zeros(2*4*CTXK*8, dtype=np.float16))
P.up("kv8B", np.zeros(2*4*CTXK*256, dtype=np.uint8)); P.up("scB", np.zeros(2*4*CTXK*8, dtype=np.float16))
dev.synchronize()
prq = prog("spk_pre3q_100k")
prq(P.d["qrow"], P.d["krow"], P.d["vrow"], P.d["qnw"], P.d["knw"], P.d["freqs"], P.d["kv8A"], P.d["scA"], P.d["pos_slot"], P.d["qwA"], global_size=(24,1,1), local_size=LS)
dev.synchronize()
prh = prog("spk_pre3qh_100k")
prh(P.d["qrow"], P.d["krow"], P.d["vrow"], P.d["qnw"], P.d["knw"], P.d["freqs"], P.d["kv8B"], P.d["scB"], P.d["pos_slot"], P.d["qwB"], P.d["qw16B"], global_size=(24,1,1), local_size=LS)
dev.synchronize()
qwA = P.down("qwA", (RM*24*256,), np.float32); qwB = P.down("qwB", (RM*24*256,), np.float32)
qw16B = P.down("qw16B", (RM*24*256,), np.float16)
print(f"[pre] qw bit-identical: {bool((qwA==qwB).all())}", flush=True)
print(f"[pre] qw16 == fp16(qw): {bool((qw16B == qwA.astype(np.float16)).all())}", flush=True)
ka = P.down("kv8A", (2*4*CTXK*256,), np.uint8); kb = P.down("kv8B", (2*4*CTXK*256,), np.uint8)
sa = P.down("scA", (2*4*CTXK*8,), np.float16); sb = P.down("scB", (2*4*CTXK*8,), np.float16)
print(f"[pre] kv8 bit-identical: {bool((ka==kb).all())}  sc bit-identical: {bool((sa==sb).all())}", flush=True)
print("[h2qk validation done]", flush=True)
