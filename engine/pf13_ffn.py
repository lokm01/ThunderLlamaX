# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P13 kernel B harness — the PERSISTENT-CTA FFN tile probe (validate + bench).
Gates (READOUT-ORDER LAW: correctness first, timing after):
  1. pf13ffn_w{2,4,8} vs the SHIPPED pfg3_ffn_r7_m32_nw8k128 (grid 272 control)
     on real packed7 weights, 8 blocks: BIT-IDENTICAL (decode/mma order verbatim).
Bench: synced min-of-10.
  - shipped control: 8 launches (one per block, grid 272 = 3.3 waves)
  - persistent: ONE launch grid 82 (1 CTA/SM), NPARC=2176 parcels, stride-82 map
Metrics: T_blk (per 32-row block pass), wGB/s = wb/T_blk (original bytes),
amort = (32/16)*wb/T_blk (the P6/P7B metric), DRAM = w * 512/392.
GATE: amort >= 350 GB/s AND >= 1.15x the same-session shipped control.
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
P7 = f"{BASE}/packed7"
NBLK = 8
KD, ND = 5120, 17408
NTILE, NGRP, NCH, NGRID = 64, 8, 40, 272
WBLK = NGRID * NGRP * NCH * 32 * 16          # bytes per plane per block
WB = 2 * 17408 * 1960                        # original weight bytes per block (fg+fu)
R7F = 512.0 / 392.0
LS = (256, 1, 1)
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

P.up("gridf", iq3_grid_f32())
fg = np.concatenate([np.load(f"{P7}/fg{b}.npy") for b in range(NBLK)])
fu = np.concatenate([np.load(f"{P7}/fu{b}.npy") for b in range(NBLK)])
assert fg.size == NBLK * WBLK and fu.size == NBLK * WBLK, (fg.size, NBLK * WBLK)
P.up("w1", fg); P.up("w2", fu)
rng = np.random.default_rng(7)
P.up("x", (rng.standard_normal((NBLK, 32, KD)) * 0.8).astype(np.float16).reshape(-1))
dev.synchronize()
print(f"[setup] weights {2*fg.nbytes/1e6:.0f} MB (8 blocks fg+fu), WBLK={WBLK}", flush=True)

def poison_out(tag):
  P.poison(tag, NBLK * 32 * ND, np.float16, 7.7)

# ---- reference: the shipped r7 m32 kernel, one launch per block (grid 272) ----
ref_name = "pfg3_ffn_r7_m32_nw8k128"
pr_ref = prog(ref_name)
poison_out("oref")
dev.synchronize()
for b in range(NBLK):
  w1b = P.d["w1"].offset(offset=b * WBLK, size=WBLK)
  w2b = P.d["w2"].offset(offset=b * WBLK, size=WBLK)
  xb = P.d["x"].offset(offset=b * 32 * KD * 2, size=32 * KD * 2)
  ob = P.d["oref"].offset(offset=b * 32 * ND * 2, size=32 * ND * 2)
  pr_ref(w1b, w2b, P.d["gridf"], xb, ob, global_size=(NGRID, 1, 1), local_size=LS)
dev.synchronize()
ref = P.down("oref", (NBLK, 32, ND), np.float16)
print("[gate] reference (shipped r7 m32 x8 blocks) computed", flush=True)

def run_persist(name):
  pr = prog(name)
  poison_out("omine")
  dev.synchronize()
  pr(P.d["w1"], P.d["w2"], P.d["gridf"], P.d["x"], P.d["omine"],
     global_size=(82, 1, 1), local_size=LS)
  dev.synchronize()
  return pr

def bench(fn, n=10):
  fn(); dev.synchronize()
  best = 1e9
  for _ in range(n):
    t0 = time.perf_counter()
    fn(); dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  return best

def ship_one():
  for b in range(NBLK):
    w1b = P.d["w1"].offset(offset=b * WBLK, size=WBLK)
    w2b = P.d["w2"].offset(offset=b * WBLK, size=WBLK)
    xb = P.d["x"].offset(offset=b * 32 * KD * 2, size=32 * KD * 2)
    ob = P.d["omine"].offset(offset=b * 32 * ND * 2, size=32 * ND * 2)
    pr_ref(w1b, w2b, P.d["gridf"], xb, ob, global_size=(NGRID, 1, 1), local_size=LS)

# ---- 1) gates: bit-identical vs shipped ----
only = [a for a in sys.argv[1:]] or ["pf13ffn_w2", "pf13ffn_w4", "pf13ffn_w8"]
ALL_OK = True
persist_pr = {}
for name in only:
  pr = run_persist(name)
  persist_pr[name] = pr
  mine = P.down("omine", (NBLK, 32, ND), np.float16)
  nz = int((mine != ref).sum())
  ok = nz == 0
  ALL_OK &= ok
  msg = "BIT-IDENTICAL" if ok else f"DIFF nz={nz}/{mine.size}"
  if not ok:
    d = np.abs(mine.astype(np.float32) - ref.astype(np.float32))
    rel = d / np.maximum(np.abs(ref.astype(np.float32)), 1e-6)
    bad = np.unravel_index(np.argmax(rel), rel.shape)
    msg += f" maxrel {rel.max():.3e} at {bad}"
  print(f"[gate] {name} vs {ref_name}: {msg}", flush=True)

# ---- 2) bench: shipped control first, then persistent variants ----
poison_out("omine"); dev.synchronize()
t_ship = bench(ship_one) / NBLK
print(f"[bench] SHIPPED-CONTROL {ref_name}: {t_ship*1e6:7.1f} us/blk | "
      f"w {WB/t_ship/1e9:6.1f} | amort {2*WB/t_ship/1e9:6.1f} | dram {R7F*WB/t_ship/1e9:6.1f} GB/s", flush=True)

for name in only:
  pr = persist_pr.get(name) or run_persist(name)
  def one():
    pr(P.d["w1"], P.d["w2"], P.d["gridf"], P.d["x"], P.d["omine"],
       global_size=(82, 1, 1), local_size=LS)
  t = bench(one) / NBLK
  print(f"[bench] PERSIST {name}: {t*1e6:7.1f} us/blk | x{t_ship/t:4.2f} vs ship | "
        f"w {WB/t/1e9:6.1f} | amort {2*WB/t/1e9:6.1f} | dram {R7F*WB/t/1e9:6.1f} GB/s"
        f" | amort-350-gate {'PASS' if 2*WB/t/1e9 >= 350 else 'FAIL'}", flush=True)

print(f"[done] gates={'ALL OK' if ALL_OK else 'FAILED'}", flush=True)
