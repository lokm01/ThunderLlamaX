# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""SKV-G validation + bench (spk_g1/g2 from spk_g.cu; KPRE/K2S reused verbatim).
MODE=py    : CTXK=2304/pos=2000/S=32 poison-first differential vs aattn3
             (gate 1e-3), numpy-fp64 sanity, T1-vs-T3 bitwise, determinism,
             empty-split identity partials.
MODE=bench : 100k sweep table (GB/s effective), POS/REPS env; POS2 list for
             the best config."""
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
def lsz(nm): return ((32*(32 if "nw32" in nm else 24 if "nw24" in nm else 16 if "nw16" in nm else 8)), 1, 1)
MODE = os.getenv("MODE", "py")
S = int(os.getenv("S", "32"))
CTXKV, POSV = 2304, 2000

def load(nm):
  lib = open(f"{BASE}/{nm}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=nm, target=dev.renderer.target, signature=tuple()))

def relerr(a, b):
  d = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
  s = np.abs(b.astype(np.float64)).max()
  return d / max(s, 1e-30)

def mk_inputs(P, rng):
  P.up("qrowA", rng.standard_normal(3*12288).astype(np.float16))
  P.up("krowA", rng.standard_normal(3*1024).astype(np.float16))
  P.up("vrowA", rng.standard_normal(3*1024).astype(np.float16))
  P.up("qnwA", (1.0 + 0.1*rng.standard_normal(256)).astype(np.float32))
  P.up("knwA", (1.0 + 0.1*rng.standard_normal(256)).astype(np.float32))
  freqs = (1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
  P.up("freqsA", freqs)
  kv = (0.1*rng.standard_normal(2*4*CTXKV*256)).astype(np.float16)
  kv.reshape(-1)[(2*4*POSV*256):] = 7.7   # poison rows >= pos
  P.up("kvA", kv)
  P.up("posA", np.array([POSV], dtype=np.int32))
  for nm, nb, dt, pv in [("qw3A", 3*24*256*4, np.float32, 7.7e31), ("qw1A", 24*256*4, np.float32, 7.7e31),
                         ("pm3A", 4*S*18*4, np.float32, 7.7e31), ("ps3A", 4*S*18*4, np.float32, 7.7e31),
                         ("pA3A", 4*S*18*256*4, np.float32, 7.7e31),
                         ("pm1A", 4*S*6*4, np.float32, 7.7e31), ("ps1A", 4*S*6*4, np.float32, 7.7e31),
                         ("pA1A", 4*S*6*256*4, np.float32, 7.7e31),
                         ("ao3A", 3*6144*2, np.float16, 7.7), ("ao1A", 6144*2, np.float16, 7.7)]:
    P.poison(nm, nb, dt, pv)
  dev.synchronize()

def numpy_ref(P):
  qw = P.down("qw3A", (3, 24, 256), np.float32)
  kv = P.down("kvA", (2, 4, CTXKV, 256), np.float16)
  qrow = P.down("qrowA", (3*12288,), np.float16)
  out = np.zeros((3, 24, 256))
  for h in range(24):
    kvh = h // 6
    for t in range(3):
      n = min(POSV + t + 1, CTXKV)
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
  d = P.d
  pr = {n: load(n) for n in ("aattn3", "spk_pre3_2k", "spk_c3", "spk_pre1_2k", "spk_c1",
                             "spk_g1a3_2k", "spk_g2a3_2k", "spk_g1a1_2k", "spk_g2a1_2k", "spk_g3a3_2k", "spk_g3a1_2k", "spk_g4nw16a3_2k", "spk_g4nw16a1_2k", "spk_g4nw32a3_2k", "spk_g4nw32a1_2k")}
  # ---- reference: old aattn3 on a fresh kv ----
  kv0 = P.down("kvA", (2*4*CTXKV*256,), np.float16).copy()
  pr["aattn3"](d["qrowA"], d["krowA"], d["vrowA"], d["qnwA"], d["knwA"], d["freqsA"], d["kvA"], d["posA"], d["ao3A"],
               global_size=(24,1,1), local_size=LS, wait=True)
  aoA = P.down("ao3A", (3, 6144), np.float16).copy()
  print(f"[py] aattn3 done absmax={np.abs(aoA.astype(np.float32)).max():.3f}", flush=True)

  for v in os.getenv("VARS", "g1,g2").split(","):
    # ---- trio (fresh kv copy) ----
    P.up("kvA", kv0)
    P.poison("ao3A", 3*6144*2, np.float16, 7.7)
    P.poison("pm3A", 4*S*18*4, np.float32, 7.7e31)
    P.poison("ps3A", 4*S*18*4, np.float32, 7.7e31)
    P.poison("pA3A", 4*S*18*256*4, np.float32, 7.7e31)
    dev.synchronize()
    pr["spk_pre3_2k"](d["qrowA"], d["krowA"], d["vrowA"], d["qnwA"], d["knwA"], d["freqsA"], d["kvA"], d["posA"], d["qw3A"],
                      global_size=(24,1,1), local_size=LS)
    pr[f"spk_{v}a3_2k"](d["kvA"], d["qw3A"], d["posA"], d["pm3A"], d["ps3A"], d["pA3A"], global_size=(4*S,1,1), local_size=lsz(f"spk_{v}a3_2k"))
    pr["spk_c3"](d["pm3A"], d["ps3A"], d["pA3A"], d["qrowA"], d["ao3A"], global_size=(24,1,1), local_size=LS, wait=True)
    aoB = P.down("ao3A", (3, 6144), np.float16).copy()
    pmB = P.down("pm3A", (4*S*18,), np.float32).copy()
    psB = P.down("ps3A", (4*S*18,), np.float32).copy()
    pAB = P.down("pA3A", (4*S*18*256,), np.float32).copy()
    print(f"[py] {v}: relerr(trio vs aattn3): {relerr(aoB, aoA):.3e}  (gate 1e-3)  bitwise={bool((aoB.view(np.uint16)==aoA.view(np.uint16)).all())}", flush=True)
    print(f"[py] {v}: relerr(trio vs numpy-fp64): {relerr(aoB.astype(np.float32).reshape(3,24,256), numpy_ref(P)):.3e}", flush=True)
    # ---- empty-split identity partials ----
    CH = CTXKV // S
    l1g = min(POSV + 3, CTXKV)
    empty = [s2 for s2 in range(S) if s2*CH >= l1g]
    pmv = pmB.reshape(4, S, 18); psv = psB.reshape(4, S, 18); pAv = pAB.reshape(4, S, 18, 256)
    ok = all((pmv[:, s2, :] == np.float32(-1e30)).all() and (psv[:, s2, :] == 0).all() and (pAv[:, s2, :, :] == 0).all() for s2 in empty)
    print(f"[py] {v}: empty splits {len(empty)}/{S} identity partials: {ok}", flush=True)
    # ---- determinism: rerun K1 (+combine) bitwise ----
    pr[f"spk_{v}a3_2k"](d["kvA"], d["qw3A"], d["posA"], d["pm3A"], d["ps3A"], d["pA3A"], global_size=(4*S,1,1), local_size=lsz(f"spk_{v}a3_2k"))
    pr["spk_c3"](d["pm3A"], d["ps3A"], d["pA3A"], d["qrowA"], d["ao3A"], global_size=(24,1,1), local_size=LS, wait=True)
    pmB2 = P.down("pm3A", (4*S*18,), np.float32).copy()
    psB2 = P.down("ps3A", (4*S*18,), np.float32).copy()
    pAB2 = P.down("pA3A", (4*S*18*256,), np.float32).copy()
    print(f"[py] {v}: rerun bitwise pm/ps/pA: {(pmB==pmB2).all()} {(psB==psB2).all()} {(pAB==pAB2).all()}", flush=True)
    # ---- T=1 row-0 bit-equality (Tier-1 contract) ----
    P.up("qrowB", P.down("qrowA", (3*12288,), np.float16)[:12288].copy())
    P.up("krowB", P.down("krowA", (3*1024,), np.float16)[:1024].copy())
    P.up("vrowB", P.down("vrowA", (3*1024,), np.float16)[:1024].copy())
    P.up("kvA", kv0)
    P.poison("ao1A", 6144*2, np.float16, 7.7)
    P.poison("pm1A", 4*S*6*4, np.float32, 7.7e31)
    P.poison("ps1A", 4*S*6*4, np.float32, 7.7e31)
    P.poison("pA1A", 4*S*6*256*4, np.float32, 7.7e31)
    dev.synchronize()
    pr["spk_pre1_2k"](d["qrowB"], d["krowB"], d["vrowB"], d["qnwA"], d["knwA"], d["freqsA"], d["kvA"], d["posA"], d["qw1A"],
                     global_size=(24,1,1), local_size=LS)
    pr[f"spk_{v}a1_2k"](d["kvA"], d["qw1A"], d["posA"], d["pm1A"], d["ps1A"], d["pA1A"], global_size=(4*S,1,1), local_size=lsz(f"spk_{v}a1_2k"))
    pr["spk_c1"](d["pm1A"], d["ps1A"], d["pA1A"], d["qrowB"], d["ao1A"], global_size=(24,1,1), local_size=LS, wait=True)
    ao1 = P.down("ao1A", (6144,), np.float16).copy()
    qw1 = P.down("qw1A", (24, 256), np.float32).copy()
    qw3 = P.down("qw3A", (3, 24, 256), np.float32).copy()
    print(f"[py] {v}: qw1 == qw3[0] bitwise: {bool((qw1.view(np.uint16) == qw3[0].view(np.uint16)).all())}", flush=True)
    print(f"[py] {v}: ao1 == aoB[0] bitwise: {bool((ao1.view(np.uint16) == aoB[0].view(np.uint16)).all())}", flush=True)
  print("[py done]", flush=True)

elif MODE == "bench":
  CTXK = 100352
  REPS = int(os.getenv("REPS", "30"))
  POS = int(os.getenv("POS", "97810"))
  SWEEP = [("spk_g1s64a3_100k", 64), ("spk_g1s128a3_100k", 128),
           ("spk_g2s32a3_100k", 32), ("spk_g2s64a3_100k", 64), ("spk_g2a3_100k", 128),
           ("spk_g2pfa3_100k", 128), ("spk_g2d2a3_100k", 128), ("spk_g2s256a3_100k", 256),
           ("spk_g2t16s64a3_100k", 64)]
  if os.getenv("PROGS"):  # override: PROGS="name:S,name:S"
    SWEEP = [(x.split(":")[0], int(x.split(":")[1])) for x in os.getenv("PROGS").split(",")]
  rng = np.random.default_rng(3)
  P = Bufs()
  P.up("kvb", (0.1*rng.standard_normal(2*4*CTXK*256)).astype(np.float16))
  P.up("qwb", rng.standard_normal(3*24*256).astype(np.float32))
  P.up("posb", np.array([POS], dtype=np.int32))
  dev.synchronize()
  byt = 2*4*CTXK*256*2 * (POS/CTXK)
  print(f"[bench] KV={2*4*CTXK*256*2/1e9:.3f} GB pos={POS} REPS={REPS}", flush=True)
  results = {}
  for name, s_ in SWEEP:
    for nm, nb, dt, pv in [("pmb", 4*s_*18*4, np.float32, 7.7e31), ("psb", 4*s_*18*4, np.float32, 7.7e31),
                           ("pAb", 4*s_*18*256*4, np.float32, 7.7e31)]:
      P.poison(nm, nb, dt, pv)
    dev.synchronize()
    pr = load(name)
    args = (P.d["kvb"], P.d["qwb"], P.d["posb"], P.d["pmb"], P.d["psb"], P.d["pAb"])
    for _ in range(3): pr(*args, global_size=(4*s_,1,1), local_size=lsz(name))
    dev.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS): pr(*args, global_size=(4*s_,1,1), local_size=lsz(name))
    dev.synchronize()
    dt = (time.perf_counter() - t0) / REPS
    results[name] = byt/dt/1e9
    print(f"[bench] {name:24s} S={s_:4d}: {dt*1e3:8.3f} ms -> {byt/dt/1e9:7.1f} GB/s", flush=True)
  best = max(results, key=results.get)
  print(f"[bench] BEST: {best} {results[best]:.1f} GB/s", flush=True)
  # pos sweep for the best config
  bs_ = dict(SWEEP)[best]
  pr = load(best)
  for nm, nb, dt, pv in [("pmb", 4*bs_*18*4, np.float32, 7.7e31), ("psb", 4*bs_*18*4, np.float32, 7.7e31),
                         ("pAb", 4*bs_*18*256*4, np.float32, 7.7e31)]:
    P.poison(nm, nb, dt, pv)
  dev.synchronize()
  args = (P.d["kvb"], P.d["qwb"], P.d["posb"], P.d["pmb"], P.d["psb"], P.d["pAb"])
  for pos2 in [int(x) for x in os.getenv("POS2", "2000,32768").split(",")]:
    P.up("posb", np.array([pos2], dtype=np.int32)); dev.synchronize()
    byt2 = 2*4*CTXK*256*2 * (pos2/CTXK)
    t0 = time.perf_counter()
    for _ in range(REPS): pr(*args, global_size=(4*bs_,1,1), local_size=lsz(best))
    dev.synchronize()
    dt2 = (time.perf_counter() - t0) / REPS
    print(f"[bench] {best} pos={pos2}: {dt2*1e3:.3f} ms -> {byt2/dt2/1e9:.1f} GB/s", flush=True)
  print("[bench done]", flush=True)
