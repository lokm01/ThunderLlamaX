# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8 A/B: (1) gdnqg packed5 (pfg3m_gdnqg_r7q4p5) vs shipped r7q4 on REAL
weights — BIT-IDENTICAL required (det x2); (2) iq3s o-proj M-grid fold
(pfg3_iq3s_m64 g=160) vs the shipped 4x pfg_iq3s_m32_hm g=80 — BIT-IDENTICAL.
Then synced min-of-8 perf per class at the TRUE M128 in-plan shapes.
Usage: ~/tg311/bin/python -u p8_ab.py"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK
import pf_prefill

E = MTPEngine(theta=1e7)
E.reset_fresh(101)
E.stload_trunk()
for i in E.gdn_idx:
  E._mfill(f"conv{i}_1", 0, CBLK)
dev.synchronize()
pf_prefill.ensure64(E)
pf_prefill.ensure128(E)
dev.synchronize()
E.P._keep.clear()
P, d, W, pr = E.P, E.P.d, E.W, E.pr
W7 = getattr(E, "_pf_W7", {})
LS = (256, 1, 1)
rng = np.random.default_rng(0)
print("[ab] world ready", flush=True)

# ---- inputs: xh128 random fp16 (moderate), out bufs poisoned fresh each run ----
P.up("xh128t", (rng.standard_normal(128*5120) * 0.7).astype(np.float16))
dev.synchronize(); P._keep.clear()
gb = [i for i in E.gdn_idx if ("gate", i) in W7 and ("qkv", i) in W][:6]
oi = [i for i in E.qtypes if ("o", i) in W][:6]
print(f"[ab] gdn blocks {len(gb)} attn blocks {len(oi)}", flush=True)

OLD = "pfg3m_gdnqg_r7q4_m64_nw8k128"
NEW = "pfg3m_gdnqg_r7q4p5_m64_nw8k128"
for n in (OLD, NEW, "pfg3_iq3s_m64_nw8k128", "pfg_iq3s_m32_hm_nw8k128"):
  if n not in pr:
    from tinygrad.device import TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    lib = open(f"~/tinygrad-metal/engine0/{n}.cubin", "rb").read()
    pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

# packed5 upload for the A/B blocks (small: 6 tensors ~236MB)
P5 = {}
for i in gb:
  P5[i] = P.up(f"p5_{i}", np.load(f"~/tinygrad-metal/engine0/packed5/qkv{i}.npy"))
dev.synchronize(); P._keep.clear()

def run_qg(p, i, wqkv, outq, outg):
  p(wqkv, W7[("gate", i)], d["gridf"], d["xh128t"], outq, outg, global_size=(512,1,1), local_size=LS, wait=True)

# ---- gdnqg A/B: bit-identity det-x2 ----
nq = 128*10240; ng = 128*6144
for it in range(2):
  for side, name, getw in (("old", OLD, lambda i: W[("qkv", i)]), ("new", NEW, lambda i: P5[i])):
    P.poison(f"qo_{side}", nq*2, np.float16, 7.7); P.poison(f"gg_{side}", ng*2, np.float16, 7.7)
    for i in gb[:2]:
      run_qg(pr[name], i, getw(i), d[f"qo_{side}"], d[f"gg_{side}"])
    dev.synchronize(); P._keep.clear()
  a = P.down("qo_old", (nq,), np.float16); b = P.down("qo_new", (nq,), np.float16)
  ag = P.down("gg_old", (ng,), np.float16); bg = P.down("gg_new", (ng,), np.float16)
  nz = int((a.view(np.uint16) != b.view(np.uint16)).sum()) + int((ag.view(np.uint16) != bg.view(np.uint16)).sum())
  print(f"[ab] gdnqg det{it}: nz={nz}/{nq+ng}", flush=True)
  assert nz == 0, (it, nz)
  P._keep.clear()
print("[ab] gdnqg OLD==NEW BIT-IDENTICAL det-x2", flush=True)

# ---- o-proj A/B: 4x m32 (shipped) vs 1x m64 g=160 (fold) ----
na = 128*5120
def V(nm, off, sz): return d[nm].offset(offset=off, size=sz)
for it in range(2):
  for side in ("old", "new"):
    P.poison(f"ao_{side}", 128*6144*2, np.float16, 7.7)
    P.poison(f"at_{side}", na*2, np.float16, 7.7)
    for i in oi[:2]:
      if side == "old":
        for pp in range(4):
          pr["pfg_iq3s_m32_hm_nw8k128"](W[("o", i)], d["grid512"],
            V(f"ao_{side}", pp*32*6144*2, 32*6144*2), V(f"at_{side}", pp*32*5120*2, 32*5120*2),
            global_size=(80,1,1), local_size=LS, wait=True)
      else:
        pr["pfg3_iq3s_m64_nw8k128"](W[("o", i)], d["grid512"], d[f"ao_{side}"], d[f"at_{side}"],
            global_size=(160,1,1), local_size=LS, wait=True)
    dev.synchronize(); P._keep.clear()
  a = P.down("at_old", (na,), np.float16); b = P.down("at_new", (na,), np.float16)
  nz = int((a.view(np.uint16) != b.view(np.uint16)).sum())
  print(f"[ab] oproj det{it}: nz={nz}/{na}", flush=True)
  assert nz == 0, (it, nz)
  P._keep.clear()
print("[ab] oproj OLD==NEW BIT-IDENTICAL det-x2", flush=True)

# ---- perf: min-of-8, true shapes ----
def bench(name, fn, blocks):
  best = 1e9
  for _ in range(3): fn(blocks[0]); dev.synchronize()
  for _ in range(8):
    t0 = time.perf_counter()
    for i in blocks: fn(i)
    dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  per = best / len(blocks)
  print(f"[perf] {name:26s} {per*1e6:9.1f} us/launch | pool48 {per*48*1e3:7.1f} ms/chunk", flush=True)
  return per

P.poison("qpb", nq*2, np.float16, 7.7); P.poison("gpb", ng*2, np.float16, 7.7)
r_old = bench("gdnqg r7q4 (old)", lambda i: run_qg(pr[OLD], i, W[("qkv", i)], d["qpb"], d["gpb"]), gb)
r_new = bench("gdnqg r7q4p5 (packed5)", lambda i: run_qg(pr[NEW], i, P5[i], d["qpb"], d["gpb"]), gb)
print(f"[perf] gdnqg speedup x{r_old/max(r_new,1e-12):.3f}  (pool delta {(r_old-r_new)*48*1e3:+.1f} ms/chunk)", flush=True)

P.poison("aopb", 128*6144*2, np.float16, 7.7); P.poison("atpb", na*2, np.float16, 7.7)
def op_old(i):
  for pp in range(4):
    pr["pfg_iq3s_m32_hm_nw8k128"](W[("o", i)], d["grid512"],
      V("aopb", pp*32*6144*2, 32*6144*2), V("atpb", pp*32*5120*2, 32*5120*2), global_size=(80,1,1), local_size=LS)
def op_new(i):
  pr["pfg3_iq3s_m64_nw8k128"](W[("o", i)], d["grid512"], d["aopb"], d["atpb"], global_size=(160,1,1), local_size=LS)
best_o = best_n = 1e9
for _ in range(3): op_old(oi[0]); dev.synchronize()
for _ in range(8):
  t0 = time.perf_counter()
  for i in oi: op_old(i)
  dev.synchronize(); best_o = min(best_o, time.perf_counter() - t0)
for _ in range(3): op_new(oi[0]); dev.synchronize()
for _ in range(8):
  t0 = time.perf_counter()
  for i in oi: op_new(i)
  dev.synchronize(); best_n = min(best_n, time.perf_counter() - t0)
po, pn = best_o/len(oi), best_n/len(oi)
print(f"[perf] {'o-proj 4xM32 (old)':26s} {po*1e6/4:9.1f} us/launch | pool48(16blk) {po*16*1e3:7.1f} ms/chunk", flush=True)
print(f"[perf] {'o-proj m64 g160 (new)':26s} {pn*1e6:9.1f} us/launch | pool48(16blk) {pn*16*1e3:7.1f} ms/chunk", flush=True)
print(f"[perf] oproj speedup x{po/max(pn,1e-12):.3f}  (pool delta {(po-pn)*16*1e3:+.1f} ms/chunk)", flush=True)
print("[ab] DONE", flush=True)
