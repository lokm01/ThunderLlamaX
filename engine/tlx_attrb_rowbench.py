# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX ATT_RB mission step 1 — THE ROW-BATCHED ATTENTION WIN-MATH MEASUREMENT.

Question: does the shipped probe attention (spk_g4nw32hm*) still have per-row
KV re-read headroom (the MM_P9 Part-3 design premise, written against the MoE
harness kernel spkq256s whose grid is (T*16, S) = one block per (t,h) seat)?

Source-level answer (engine0/spk_g4hm.cu + the W3 design comment in
spk_c4.cu: "K1S: grid (4 kv-groups x S splits). GQA-shared: ONE KV read
serves all 6 q-heads x ROWS rows"): the DENSE probe kernel already batches
ALL rows and ALL 6 heads of a KV group into ONE CTA per (group, split) —
the KV tile is staged to smem ONCE and every row's QK/PV runs against it.

This bench proves it EMPIRICALLY without booting the engine (no weights):
bench the T=1 / T=3 / T=5 cubins (spk_g4nw32hm{1,3,5}_100k — same source,
same grid (4*S,), same KV fixture, only ROWS differs) at the same positions.
  per-seat KV re-read  => time scales ~1 : 3 : 5 with ROWS
  batched KV reads     => time scales ~1 : 1.1 : 1.2 (row work only)
It also reports effective KV GB/s vs the 880 GB/s dext data-path ceiling —
the BW-bound vs phase-bound discriminator for any remaining attention win.

Timing: synced, min-of-N + avg (the launch-floor discipline). Fixture +
harness pattern verbatim test_hm.py.
"""
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
REPS = 12

P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

rng = np.random.default_rng(7)
print("[rb] generating kv fixture (2,4,CTXK,256 int8 g128)...", flush=True)
kv16 = (rng.standard_normal((2, 4, CTXK, 256)).astype(np.float32) * 0.5).astype(np.float16)
g = kv16.reshape(2, 4, CTXK, 8, 32)
amax = np.abs(g).max(axis=-1)
sc = (np.maximum(amax, 1e-8) * (1.0/127.0)).astype(np.float16)
q = (np.clip(np.rint(g.astype(np.float32) / sc.astype(np.float32)[..., None]), -127, 127) + 128).astype(np.uint8)
del g, kv16
P.up("kv8", q.reshape(-1)); P.up("sc", sc.reshape(-1))
del q, sc
print("[rb] fixture uploaded", flush=True)

def bench(name, rm, pos):
  # qw16 table sized for this ROWS build (RMAX = 6*ROWS rows of 256 dims)
  qw = (rng.standard_normal((rm*24, 256)).astype(np.float32) * 0.05).astype(np.float16)
  P.up("qw16x", qw); del qw
  n6 = 4*S*6*rm
  P.poison("pmx", n6*4, np.float32, 7.7e31)
  P.poison("psx", n6*4, np.float32, 7.7e31)
  P.poison("pAx", n6*256*4, np.float32, 7.7e31)
  P.up("posx", np.array([pos], dtype=np.int32))
  dev.synchronize()
  pr = prog(name)
  args = (P.d["kv8"], P.d["sc"], P.d["qw16x"], P.d["posx"], P.d["pmx"], P.d["psx"], P.d["pAx"])
  for _ in range(3):
    pr(*args, global_size=(4*S,1,1), local_size=LS32); dev.synchronize()
  ts = []
  for _ in range(REPS):
    t0 = time.perf_counter()
    pr(*args, global_size=(4*S,1,1), local_size=LS32); dev.synchronize()
    ts.append(time.perf_counter() - t0)
  tmin, tavg = min(ts), sum(ts)/len(ts)
  # bytes touched per launch (DRAM-class): KV quants 4g*pos*512B + scales
  # 4g*pos*32B + partial writes (pm/ps 2*4*S*RMAX*4B + pA 4*S*RMAX*256*4B)
  kvb = 4 * (pos + rm) * (512 + 32)
  pwb = 4*S*6*rm * (8 + 1024)
  bw = (kvb + pwb) / tmin / 1e9
  print(f"[rb] {name} ROWS={rm} @{pos:>6}: min {tmin*1e3:7.3f} ms  avg {tavg*1e3:7.3f} ms  "
        f"KV {kvb/1e9:.3f} GB + partial-w {pwb/1e6:.1f} MB -> {bw:6.1f} GB/s", flush=True)
  return tmin

print("[rb] === the row-scaling proof (same pos, same fixture, ROWS varies) ===", flush=True)
res = {}
for pos in (97810, 8192, 2048):
  print(f"[rb] --- pos {pos} ---", flush=True)
  res[pos] = {}
  for rm, n in ((1, "spk_g4nw32hm1_100k"), (3, "spk_g4nw32hm3_100k"), (5, "spk_g4nw32hm5_100k")):
    res[pos][rm] = bench(n, rm, pos)

print("[rb] === row-scaling ratios (t/t1) — per-seat re-read would be ~1:3:5 ===", flush=True)
for pos in res:
  t1, t3, t5 = res[pos][1], res[pos][3], res[pos][5]
  print(f"[rb] @{pos:>6}: hm1 {t1*1e3:7.3f} | hm3 {t3*1e3:7.3f} ({t3/t1:4.2f}x) | "
        f"hm5 {t5*1e3:7.3f} ({t5/t1:4.2f}x) | per-seat counterfactual hm3~3x hm1={3*t1*1e3:7.3f} ms", flush=True)
  print(f"[rb] @{pos:>6}: the ALREADY-BANKED batching win (3*hm1 - hm3) = {(3*t1-t3)*1e3:+7.3f} ms/layer-class", flush=True)
print("[rb] done", flush=True)
