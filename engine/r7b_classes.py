# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7b per-class prefill GEMM bench: time each m64 class STANDALONE at the TRUE
M128 in-plan shapes (real weights, plan buffers, min-of-8 synced) -> pool ranking
(launch-count x per-launch) for the warp-spec target decision. r2d_ring4b pattern.
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK
import pf_prefill

E = MTPEngine(theta=1e7)
P, d, pr = E.P, E.P.d, E.pr
rng = np.random.default_rng(0)
ids = [int(t) for t in rng.integers(1000, 60000, 2048)]

E.reset_fresh(ids[0])
E.stload_trunk()
for i in E.gdn_idx:
  E._mfill(f"conv{i}_1", 0, CBLK)
dev.synchronize()
from mtp import SLICE
_seen, sl = set(), []
for t in ids:
  if t not in _seen:
    _seen.add(t); sl.append(t)
_base = sl[:]
while len(sl) < SLICE:
  sl += _base
E.init_draft(sl[:SLICE])
E.fill_draft(ids, start_pos=0, seed_hd=None)
dev.synchronize()

CT = []
dt = pf_prefill.prefill_batch(E, None, ids, chunk_times=CT)
dev.synchronize()
print(f"[boot] prefill 2048: {dt:.2f}s (chunk med {np.median(CT[len(CT)//2:]):.1f}ms)", flush=True)

W7, W = getattr(E, "_pf_W7", {}), E.W
LS = (256, 1, 1)

def fill(nm, shape, dt_, seed):
  a = (np.random.default_rng(seed).standard_normal(shape) * 0.7)
  P.win_up(nm, 0, a.astype(dt_).reshape(-1))
fill("xh128", (128, 5120), np.float16, 14)
fill("z128", (128, 6144), np.float16, 13)
fill("hhx128", (128, 5120), np.float16, 12)
fill("gact128", (128, 17408), np.float16, 11)
dev.synchronize()
P._keep.clear()

gdn_blocks = [i for i in E.gdn_idx if ("fd", i) in W7][:6]
out_blocks = [i for i in E.gdn_idx if (not E.gdn_oq8[i]) and ("out", i) in W7][:6]
ffn_blocks = [i for i in E.gdn_idx if ("fg", i) in W7][:6]
attn_i3 = [i for i in E.qtypes if E.qtypes[i] != 14 and ("k", i) in W7][:3]
attn_q6 = [i for i in E.qtypes if E.qtypes[i] == 14 and ("k", i) in W7][:3]

GQG = pr["pfg3m_gdnqg_r7q4_m64_nw8k128"]
FFN = pr["pfg3_ffn_r7_m64_nw4k128"]
FD = pr["pfg3_iq3d_r7_m64_nw8k128"]
OUT = pr["pfg3_iq3o_r7_m64_nw8k128"]
QKV_I3 = pr["pfg3m_attnqkvi3_r7_m64_nw8k128"]
QKV_Q6 = pr["pfg3m_attnqkvq6_r7_m64_nw8k128"]
OP = pr["pfg_iq3s_m32_hm_nw8k128"]

def qg(i):   GQG(W[("qkv", i)], W7[("gate", i)], d["gridf"], d["xh128"], d["qkv128"], d["gate128"], global_size=(512,1,1), local_size=LS)
def ffn(i):  FFN(W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx128"], d["gact128"], global_size=(1088,1,1), local_size=(128,1,1))
def fd(i):   FD(W7[("fd", i)], d["gridf"], d["gact128"], d["hh128"], d["xB128"], global_size=(160,1,1), local_size=LS)
def out(i):  OUT(W7[("out", i)], d["gridf"], d["z128"], d["attn_out128"], global_size=(160,1,1), local_size=LS)
def qkv(p, i):
  qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
  p(qw_, W7[("k", i)], W[("v", i)], d["gridf"],
    d["xh128"], d["qrow128"], d["krow128"], d["vrow128"], global_size=(448,1,1), local_size=LS)
def V(nm, off, sz): return d[nm].offset(offset=off, size=sz)
def op32(i):
  for pp in range(4):
    OP(W[("o", i)], d["grid512"], V("ao128", pp*32*6144*2, 32*6144*2),
       V("attn_out128", pp*32*5120*2, 32*5120*2), global_size=(80,1,1), local_size=LS)

def bench(name, fn, blocks, wmb, nlaunch=1):
  best = 1e9
  for _ in range(3):
    fn(blocks[0]); dev.synchronize()   # warm
  for _ in range(8):
    t0 = time.perf_counter()
    for i in blocks:
      fn(i)
    dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  per = best / len(blocks)
  pool = per * nlaunch * 48 / 1000.0 if nlaunch else None
  print(f"[cls] {name:12s} {per*1e6:9.1f} us/launch | {wmb:7.1f} MB W -> {wmb/1e3/max(per,1e-12):6.1f} GB/s"
        + (f" | pool48 {pool*1e3:7.1f} ms/chunk" if pool else ""), flush=True)
  return per

# W bytes per launch (approx, r7=5b/w, q5k=5.5b/w, iq3s~4.3b/w, q6=6.6b/w)
n_attn = 16
print("== per-class standalone (TRUE M128 shapes, min-of-8) ==", flush=True)
r_qg  = bench("gdnqg qg", qg, gdn_blocks, (10240+6144)*5120*0.6875/1024/1024 + 6144*5120*0.625/1024/1024)
r_ffn = bench("ffn fg+fu", ffn, ffn_blocks, 2*8704*5120*0.625/1024/1024)
r_fd  = bench("iq3d fd", fd, gdn_blocks, 8704*5120*0.625/1024/1024)
r_out = bench("iq3o out", out, out_blocks, 5120*6144*0.625/1024/1024)
r_i3  = bench("qkv i3", lambda i: qkv(QKV_I3, i), attn_i3, 14336*5120*0.6/1024/1024)
r_q6  = bench("qkv q6", lambda i: qkv(QKV_Q6, i), attn_q6, 14336*5120*0.65/1024/1024)
# o-proj: one BLOCK = 4 launches
best = 1e9
for _ in range(3): op32(attn_i3[0]); dev.synchronize()
for _ in range(8):
  t0 = time.perf_counter()
  for i in attn_i3 + attn_q6: op32(i)
  dev.synchronize()
  best = min(best, time.perf_counter() - t0)
per_blk = best / (len(attn_i3) + len(attn_q6))
print(f"[cls] {'o-proj 4xM32':12s} {per_blk*1e6/4:9.1f} us/launch | pool48(16blk) {per_blk*16*1e3:7.1f} ms/chunk", flush=True)

tot = 48*r_qg + 48*r_ffn + 48*r_fd + 48*r_out + 16*(r_i3 + r_q6)
print(f"\n[sum] GEMM pools (qg+ffn+fd+out+qkv): {tot*1e3:.1f} ms per 128-chunk", flush=True)
print("[cls] DONE")
