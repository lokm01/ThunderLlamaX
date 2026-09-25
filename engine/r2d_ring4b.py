# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2d follow-up bench (one boot): (a) r7q4 at the TRUE M128 in-plan shapes —
qg g=512 2-M-block, out g=160 2-M-block (adoption decision per class);
(b) SCRAP 2 pricing: attnqkv m64 as ONE g=448 2-M-block launch vs the shipped
2x g=224 (identity + bench, both flavors).
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK
import pf_prefill
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

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
print(f"[boot] prefill 2048: {dt:.2f}s", flush=True)

BASE = os.path.dirname(os.path.abspath(pf_prefill.__file__))
for n in ["pfg3_iq3o_r7q4_m64_nw8k128", "pfg3m_gdnqg_r7q4_m64_nw8k128"]:
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
dev.synchronize(); P._keep.clear()

W7, W = E._pf_W7, E.W
LS = (256, 1, 1)
def fill(nm, shape, dt_, seed):
  a = (np.random.default_rng(seed).standard_normal(shape) * 0.7)
  P.win_up(nm, 0, a.astype(dt_).reshape(-1))
fill("xh128", (128, 5120), np.float16, 14)
fill("z128", (128, 6144), np.float16, 13)
dev.synchronize()
def V(nm, off, sz): return d[nm].offset(offset=off, size=sz)
def grab(nm, shape, dt_): return P.down(nm, shape, dt_).copy()

gdn_blocks = [i for i in E.gdn_idx if ("fd", i) in W7][:8]
out_blocks = [i for i in E.gdn_idx if ("out", i) in W7][:8]
attn_i3 = [i for i in E.qtypes if E.qtypes[i] != 14 and ("k", i) in W7][:8]
attn_q6 = [i for i in E.qtypes if E.qtypes[i] == 14 and ("k", i) in W7][:8]

# ---- (a) M128 shapes: qg g=512, out g=160 ----
def qg128(p, i):
  p(W[("qkv", i)], W7[("gate", i)], d["gridf"], d["xh128"], d["qkv128"], d["gate128"], global_size=(512, 1, 1), local_size=LS)
def out128(p, i):
  p(W7[("out", i)], d["gridf"], d["z128"], d["attn_out128"], global_size=(160, 1, 1), local_size=LS)
SH_QG, Q4_QG = pr["pfg3m_gdnqg_r7_m64_nw8k128"], pr["pfg3m_gdnqg_r7q4_m64_nw8k128"]
SH_O, Q4_O = pr["pfg3_iq3o_r7_m64_nw8k128"], pr["pfg3_iq3o_r7q4_m64_nw8k128"]

print("== (a) identity at M128 shapes ==", flush=True)
qg128(SH_QG, gdn_blocks[0]); dev.synchronize()
r1, r2 = grab("qkv128", (128, 10240), np.float16), grab("gate128", (128, 6144), np.float16)
qg128(Q4_QG, gdn_blocks[0]); dev.synchronize()
m1, m2_ = grab("qkv128", (128, 10240), np.float16), grab("gate128", (128, 6144), np.float16)
nz = int((m1 != r1).sum()) + int((m2_ != r2).sum())
print(f"[gate] qg@512: nz={nz} -> {'BIT-IDENTICAL' if nz==0 else 'DIFF'}", flush=True)
out128(SH_O, out_blocks[0]); dev.synchronize()
r1 = grab("attn_out128", (128, 5120), np.float16)
out128(Q4_O, out_blocks[0]); dev.synchronize()
m1 = grab("attn_out128", (128, 5120), np.float16)
nz = int((m1 != r1).sum())
print(f"[gate] out@160: nz={nz} -> {'BIT-IDENTICAL' if nz==0 else 'DIFF'}", flush=True)

# ---- (b) scrap 2: attnqkv g=448 ONE launch vs 2x g=224 (shipped r7 cubin) ----
def qkv_2x(p, i):
  if E.qtypes[i] == 14: qw_ = W[("q", i)]
  else: qw_ = W7[("q", i)]
  for pp in range(2):
    p(qw_, W7[("k", i)], W[("v", i)], d["gridf"],
      V("xh128", pp*64*5120*2, 64*5120*2), V("qrow128", pp*64*12288*2, 64*12288*2),
      V("krow128", pp*64*1024*2, 64*1024*2), V("vrow128", pp*64*1024*2, 64*1024*2),
      global_size=(224, 1, 1), local_size=LS)
def qkv_1(p, i):
  qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
  p(qw_, W7[("k", i)], W[("v", i)], d["gridf"], d["xh128"],
    d["qrow128"], d["krow128"], d["vrow128"], global_size=(448, 1, 1), local_size=LS)

print("== (b) scrap2 identity: g=448 ONE launch vs 2x g=224 (shipped cubin) ==", flush=True)
for tag, blks in [("qkv_i3", attn_i3), ("qkv_q6", attn_q6)]:
  i0 = blks[0]
  qkv_2x(pr["pfg3m_attnqkvi3_r7_m64_nw8k128" if tag == "qkv_i3" else "pfg3m_attnqkvq6_r7_m64_nw8k128"], i0); dev.synchronize()
  r1, r2, r3 = grab("qrow128", (128, 12288), np.float16), grab("krow128", (128, 1024), np.float16), grab("vrow128", (128, 1024), np.float16)
  qkv_1(pr["pfg3m_attnqkvi3_r7_m64_nw8k128" if tag == "qkv_i3" else "pfg3m_attnqkvq6_r7_m64_nw8k128"], i0); dev.synchronize()
  m1, m2_, m3 = grab("qrow128", (128, 12288), np.float16), grab("krow128", (128, 1024), np.float16), grab("vrow128", (128, 1024), np.float16)
  nz = int((m1 != r1).sum()) + int((m2_ != r2).sum()) + int((m3 != r3).sum())
  print(f"[gate] {tag} g448-vs-2x224: nz={nz} -> {'BIT-IDENTICAL' if nz==0 else 'DIFF'}", flush=True)

def bench(fn, blks, n=10):
  fn(blks[0]); dev.synchronize()
  best = 1e9
  for _ in range(n):
    t0 = time.perf_counter()
    for i in blks: fn(i)
    dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  return best / len(blks)

print("== bench ==", flush=True)
t_sh_qg = bench(lambda i: qg128(SH_QG, i), gdn_blocks)
t_q4_qg = bench(lambda i: qg128(Q4_QG, i), gdn_blocks)
print(f"[bench] qg@512  shipped {t_sh_qg*1e6:8.1f} r7q4 {t_q4_qg*1e6:8.1f} us/blk | x{t_sh_qg/t_q4_qg:4.3f}", flush=True)
t_sh_o = bench(lambda i: out128(SH_O, i), out_blocks)
t_q4_o = bench(lambda i: out128(Q4_O, i), out_blocks)
print(f"[bench] out@160 shipped {t_sh_o*1e6:8.1f} r7q4 {t_q4_o*1e6:8.1f} us/blk | x{t_sh_o/t_q4_o:4.3f}", flush=True)
for tag, blks in [("qkv_i3", attn_i3), ("qkv_q6", attn_q6)]:
  shp = pr["pfg3m_attnqkvi3_r7_m64_nw8k128" if tag == "qkv_i3" else "pfg3m_attnqkvq6_r7_m64_nw8k128"]
  t2 = bench(lambda i: qkv_2x(shp, i), blks)
  t1 = bench(lambda i: qkv_1(shp, i), blks)
  print(f"[bench] {tag}: 2x224 {t2*1e6:8.1f} vs 1x448 {t1*1e6:8.1f} us/blk | x{t2/t1:4.3f}", flush=True)
print("[r2d b] done", flush=True)
