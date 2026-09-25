# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P9 probe: pfa32c co-resident attention vs the shipped pfa16 pair.
corr  = 2k-class correctness vs pfa16 (numpy flash-combine of pm/ps/pA partials,
        poison-first, determinism x2); bench = synced min-of-5 at pos=100336.
Carveout via env (NV_SMEM_CFG_AUTO + AUTO_NAMES=pfa32c matches all 3 entries).
Run: cd ~/tinygrad-metal/engine0 && <FULL env> ~/tg311/bin/python -u pf32c_probe.py corr|bench
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad import dtypes
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

BASE = "~/tinygrad-metal/engine0"
dev = Device["NV"]
P = Bufs()
mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
CTXK = 2048 if mode == "corr" else 100352
rng = np.random.default_rng(11)
INT = (None, 4, dtypes.int32, ())
print(f"[env] AUTO={os.getenv('NV_SMEM_CFG_AUTO','-')} ANAMES={os.getenv('NV_SMEM_CFG_AUTO_NAMES','-')} "
      f"TGT={os.getenv('NV_SMEM_CFG_AUTO_TGT','-')} mode={mode} CTXK={CTXK}", flush=True)

# name -> (cubin, entry, S, NHP, HRP, RMAX, NTHR) ; pfa16 ref first
K = {
  "pfa16": (("pfa16r_s8_2k", "pfa16r", 8, None, 6, 96, 1024) if mode=="corr" else ("pfa16nw32_s32_100k", "pfa16", 32, None, 6, 96, 1024)),
  "t64":   (f"pfa32c_t64_s13_{'2k' if mode=='corr' else '100k'}", "pfa32c", 13, 3, 2, 32, 512),
  "t32":   (f"pfa32c_t32_s13_{'2k' if mode=='corr' else '100k'}", "pfa32ct", 13, 3, 2, 32, 512),
  "16c":   (f"pfa32c_16c_s10_{'2k' if mode=='corr' else '100k'}", "pfa32c16", 10, 6, 1, 16, 256),
}
_pc = {}
def prog(n):
  if n in _pc: return _pc[n]
  cub, ent, S, NHP, HRP, RMAX, NTHR = K[n]
  lib = open(f"{BASE}/{cub}.cubin", "rb").read()
  _pc[n] = NVProgram(dev, TinyELF(lib=lib, name=ent, target=dev.renderer.target, signature=tuple()))
  return _pc[n]

def run(pk, n, pos):
  _, _, S, NHP, HRP, RMAX, NTHR = K[n]
  gs = 4 * S * (NHP or 1)
  for nm, nb in [("pm", 4*max(S*(NHP or 1),1)*0)]: pass
  P.poison("pm", gs*RMAX*4, np.float32, 7.7e31); P.poison("ps", gs*RMAX*4, np.float32, 7.7e31)
  P.poison("pA", gs*RMAX*256*4, np.float32, 7.7e31)
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
  pk(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
     global_size=(gs,1,1), local_size=(NTHR,1,1))
  dev.synchronize()
  pm = P.down("pm", (gs*RMAX,), np.float32).copy().reshape(gs, RMAX)
  ps = P.down("ps", (gs*RMAX,), np.float32).copy().reshape(gs, RMAX)
  pA = P.down("pA", (gs*RMAX*256,), np.float32).copy().reshape(gs, RMAX, 256)
  return pm, ps, pA

def combine(n, pm, ps, pA):
  # flash-combine partials over splits -> out [384, 256]; row map per kernel family
  _, _, S, NHP, HRP, RMAX, _ = K[n]
  gs = pm.shape[0]
  rows = np.zeros((gs, RMAX), dtype=np.int64)
  for bx in range(gs):
    if NHP is None:
      g, s = bx // S, bx % S
      for r in range(RMAX): rows[bx, r] = (r % 16)*24 + g*6 + r//16
    else:
      hp = bx % NHP; s = (bx // NHP) % S; g = bx // (NHP*S)
      for r in range(RMAX): rows[bx, r] = (r % 16)*24 + g*6 + hp*HRP + r//16
  m = np.full((384, gs), -1e30, dtype=np.float32); nu = np.zeros((384, gs), dtype=np.float32)
  de = np.zeros((384, gs), dtype=np.float32)
  O = np.zeros((384, gs, 256), dtype=np.float32)
  for bx in range(gs):
    m[rows[bx], bx] = pm[bx]; nu[rows[bx], bx] = ps[bx]; O[rows[bx], bx] = pA[bx]
  gmax = m.max(axis=1, keepdims=True)
  with np.errstate(under="ignore"):
    w = np.exp(m - gmax)
  out = (O * w[:, :, None]).sum(axis=1)
  den = (nu * w).sum(axis=1)
  return out / np.maximum(den, 1e-30)[:, None], den

# ---- buffers (real-scale synthetic, pfq8_probe distributions) ----
kv = np.clip(rng.integers(0, 256, (2*4*CTXK*256,)).astype(np.uint8), 0, 255)
P.poison("kv", 2*4*CTXK*256, np.uint8, 200); P.up("kv", kv)
sc = (rng.uniform(0.001, 0.02, (2*4*CTXK*8))).astype(np.float16)
P.poison("sc", 2*4*CTXK*8*2, np.float16, np.float16(7.7)); P.up("sc", sc)
qw = (rng.standard_normal(32*24*256) * 0.5).astype(np.float16)
P.poison("qw", 32*24*256*2, np.float16, __import__("numpy").float16(7.7)); P.up("qw", qw)
P.poison("pos_slot", 4, np.int32, -1)
P._keep.clear(); dev.synchronize()

pos = 2032 if mode == "corr" else 100336
res = {}
for n in ["pfa16", "t64", "t32", "16c"]:
  pk = prog(n)
  pm, ps, pA = run(pk, n, pos); res[n] = (pm, ps, pA)
  # determinism x2
  pm2, ps2, pA2 = run(pk, n, pos)
  det = np.array_equal(pm, pm2) and np.array_equal(ps, ps2) and np.array_equal(pA, pA2)
  out, den = combine(n, pm, ps, pA); res[n] = (out, den)
  print(f"[{mode}] {n}: determinism x2 = {det}", flush=True)

if mode == "corr":
  ref, refden = res["pfa16"]
  for n in ["t64", "t32", "16c"]:
    out, den = res[n]
    d = np.maximum(np.abs(ref), 1e-6)
    e = np.abs(out - ref) / d
    print(f"[corr] {n} vs pfa16 (combined out, pos=2032): relerr med {np.median(e):.3e} max {np.max(e):.3e} "
          f"| den relerr med {np.median(np.abs(den-refden)/np.maximum(np.abs(refden),1e-6)):.3e}", flush=True)
else:
  for n in ["pfa16", "t64", "t32", "16c"]:
    pk = prog(n)
    _, _, S, NHP, HRP, RMAX, NTHR = K[n]
    gs = 4 * S * (NHP or 1)
    P.poison("pm", gs*RMAX*4, np.float32, 7.7e31); P.poison("ps", gs*RMAX*4, np.float32, 7.7e31)
    P.poison("pA", gs*RMAX*256*4, np.float32, 7.7e31); dev.synchronize()
    def fn():
      pk(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
         global_size=(gs,1,1), local_size=(NTHR,1,1))
    for _ in range(2): fn()
    dev.synchronize()
    best = 1e9
    for _ in range(5):
      t0 = time.perf_counter(); fn(); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    print(f"[bench] {n} pos={pos}: {best*1e3:8.2f} ms/launch (grid {gs} x {NTHR}thr)", flush=True)
P._keep.clear()
print(f"[{mode}] DONE", flush=True)
