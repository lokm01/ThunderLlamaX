# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2F L3 standalone validation: aq3k8v_3 / aq6k8v_3 / head8v_3 vs their
fp32-core originals on REAL weights -> BIT-IDENTICAL outputs required, plus
synced bench. Fixture pattern from bench_g3.py."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)
DIM, VOCAB = 5120, 248320
P = Bufs()
pr = {}
def load(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
for n in ["aq3k8_3", "aq3k8v_3", "aq6k8_3", "aq6k8v_3", "head8_3", "head8v_3"]:
  load(n)
ds, infos = parse_gguf()
P.up("gridf", iq3_grid_f32())
rng = np.random.default_rng(11)
# find one IQ3-q layer and one Q6-q layer
q3i = q6i = None
for i in (11, 3, 15, 7, 23):
  qt = infos[f"blk.{i}.attn_q.weight"][0]
  if qt == 18 and q3i is None: q3i = i
  if qt == 14 and q6i is None: q6i = i
print(f"[fx] IQ3-q layer {q3i}, Q6-q layer {q6i}", flush=True)
def upraw(bufnm, gg):
  arr = np.frombuffer(read_raw(infos[gg], ds), dtype=np.uint8)
  return P.up(bufnm, arr)
for tag, i in (("iq3", q3i), ("q6", q6i)):
  P.up(f"wq_{tag}", np.ascontiguousarray(np.load(f"{BASE}/packed/q{i}.npy")))
  P.up(f"wk_{tag}", np.ascontiguousarray(np.load(f"{BASE}/packed/k{i}.npy")))
  upraw(f"wv_{tag}", f"blk.{i}.attn_v.weight")
upraw("headw", "output.weight")
xh3 = (rng.standard_normal((3, DIM)).astype(np.float32) * 0.3).astype(np.float16)
P.up("xh3", xh3.reshape(-1))
P.poison("qrow3A", 3*12288*2, np.float16, 7.7); P.poison("qrow3B", 3*12288*2, np.float16, 7.7)
P.poison("krow3A", 3*1024*2, np.float16, 7.7); P.poison("krow3B", 3*1024*2, np.float16, 7.7)
P.poison("vrow3A", 3*1024*2, np.float16, 7.7); P.poison("vrow3B", 3*1024*2, np.float16, 7.7)
P.poison("logitsA", 3*VOCAB*2, np.float16, 7.7); P.poison("logitsB", 3*VOCAB*2, np.float16, 7.7)
dev.synchronize()

def run_pair(oldn, newn, argsA, argsB, grid, outs, nm):
  for r in range(2):
    pr[oldn](*argsA, global_size=(grid,1,1), local_size=LS)
    pr[newn](*argsB, global_size=(grid,1,1), local_size=LS)
  dev.synchronize()
  for oa, ob, sh in outs:
    a = P.down(oa, sh, np.float16); b = P.down(ob, sh, np.float16)
    print(f"[{nm}] {oa} vs {ob} bit-identical: {bool((a==b).all())} (nonzero {int((a!=0).sum())})", flush=True)
  t0 = time.perf_counter()
  for r in range(20): pr[oldn](*argsA, global_size=(grid,1,1), local_size=LS); dev.synchronize()
  dt0 = (time.perf_counter()-t0)/20
  t0 = time.perf_counter()
  for r in range(20): pr[newn](*argsB, global_size=(grid,1,1), local_size=LS); dev.synchronize()
  dt1 = (time.perf_counter()-t0)/20
  print(f"[{nm}] {oldn} {dt0*1e3:.3f} ms -> {newn} {dt1*1e3:.3f} ms  delta {(dt0-dt1)*1e3:+.3f}", flush=True)

d = P.d
for tag, oldn, newn in (("iq3", "aq3k8_3", "aq3k8v_3"), ("q6", "aq6k8_3", "aq6k8v_3")):
  run_pair(oldn, newn,
    (d[f"wq_{tag}"], d[f"wk_{tag}"], d[f"wv_{tag}"], d["gridf"], d["xh3"], d["qrow3A"], d["krow3A"], d["vrow3A"]),
    (d[f"wq_{tag}"], d[f"wk_{tag}"], d[f"wv_{tag}"], d["gridf"], d["xh3"], d["qrow3B"], d["krow3B"], d["vrow3B"]),
    1792, [("qrow3A","qrow3B",(3*12288,)),("krow3A","krow3B",(3*1024,)),("vrow3A","vrow3B",(3*1024,))], tag)
run_pair("head8_3", "head8v_3", (d["headw"], d["xh3"], d["logitsA"]), (d["headw"], d["xh3"], d["logitsB"]),
  VOCAB//8, [("logitsA","logitsB",(3*VOCAB,))], "head")
print("[l3 validation done]", flush=True)
