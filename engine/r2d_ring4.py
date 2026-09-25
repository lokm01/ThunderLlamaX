# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2d SCRAP 1 discriminator: DBUF ring-depth-4 (r7q4) vs the shipped r7 m64
cubins on the 4 non-FFN GEMM classes. In-plan, the pf14/pf16 laws: cubins
loaded at the proven-stable window (post-trunk-load, post-prefill); bench via
launches into EXISTING plan buffers only; synced min-of-10.
Arms (readout-order law — first clean run is the gate):
  G1 BIT-IDENTITY: r7q4 vs shipped r7 m64 per class (fd/iq3d, out/iq3o,
     qg/gdnqg, qkv/attnqkvi3+attnqkvq6) on REAL W7/W weights; determinism x2.
  G2 BENCH (synced min-of-10 over 8 blocks/class, m64 shapes g=80/80/256/224
     + the M128 2-M-block shape for iq3d g=160): shipped vs r7q4.
Verdict per class: shipped/r7q4 >= 1.03 keep | < 1.02 negative.
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
t0 = time.perf_counter()
dt = pf_prefill.prefill_batch(E, None, ids, chunk_times=CT)
dev.synchronize()
cts = sorted(ms for _, ms in CT)
print(f"[boot] prefill 2048: {dt:.2f}s chunks={len(cts)} med={cts[len(cts)//2]:.1f}ms", flush=True)

BASE = os.path.dirname(os.path.abspath(pf_prefill.__file__))
MINE = ["pfg3_iq3d_r7q4_m64_nw8k128", "pfg3_iq3o_r7q4_m64_nw8k128",
        "pfg3m_gdnqg_r7q4_m64_nw8k128", "pfg3m_attnqkvi3_r7q4_m64_nw8k128",
        "pfg3m_attnqkvq6_r7q4_m64_nw8k128"]
for n in MINE:
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  print(f"[load] {n}: shmem {pr[n].shmem_usage} regs {pr[n].regs_usage}", flush=True)
dev.synchronize(); P._keep.clear()

W7 = E._pf_W7
W = E.W
LS = (256, 1, 1)

# ---- deterministic inputs into plan buffers (win_up law) ----
def fill(nm, shape, dt_, seed):
  a = (np.random.default_rng(seed).standard_normal(shape) * 0.7)
  P.win_up(nm, 0, a.astype(dt_).reshape(-1))
fill("gact64", (64, 17408), np.float16, 11)
P.win_up("hh64", 0, (np.random.default_rng(12).standard_normal((64, 5120)) * 0.7).astype(np.float32).reshape(-1))
fill("z64", (64, 6144), np.float16, 13)
fill("xh64", (64, 5120), np.float16, 14)
fill("gact128", (128, 17408), np.float16, 15)
P.win_up("hh128", 0, (np.random.default_rng(16).standard_normal((128, 5120)) * 0.7).astype(np.float32).reshape(-1))
dev.synchronize()

# ---- per-class runners (m64 shapes, the ensure64 verbatim arg orders) ----
def k_fd(p, i, out):
  p(W7[("fd", i)], d["gridf"], d["gact64"], d["hh64"], out, global_size=(80, 1, 1), local_size=LS)
def k_fd128(p, i, out):
  p(W7[("fd", i)], d["gridf"], d["gact128"], d["hh128"], out, global_size=(160, 1, 1), local_size=LS)
def k_out(p, i):
  p(W7[("out", i)], d["gridf"], d["z64"], d["attn_out64"], global_size=(80, 1, 1), local_size=LS)
def k_qg(p, i):
  p(W[("qkv", i)], W7[("gate", i)], d["gridf"], d["xh64"], d["qkv64"], d["gate64"], global_size=(256, 1, 1), local_size=LS)
def k_qkv(p, i):
  if E.qtypes[i] == 14:
    p(W[("q", i)], W7[("k", i)], W[("v", i)], d["gridf"], d["xh64"],
      d["qrow64"], d["krow64"], d["vrow64"], global_size=(224, 1, 1), local_size=LS)
  else:
    p(W7[("q", i)], W7[("k", i)], W[("v", i)], d["gridf"], d["xh64"],
      d["qrow64"], d["krow64"], d["vrow64"], global_size=(224, 1, 1), local_size=LS)

SH = {"fd": pr["pfg3_iq3d_r7_m64_nw8k128"], "out": pr["pfg3_iq3o_r7_m64_nw8k128"],
      "qg": pr["pfg3m_gdnqg_r7_m64_nw8k128"],
      "qkv_i3": pr["pfg3m_attnqkvi3_r7_m64_nw8k128"], "qkv_q6": pr["pfg3m_attnqkvq6_r7_m64_nw8k128"]}
Q4 = {"fd": pr["pfg3_iq3d_r7q4_m64_nw8k128"], "out": pr["pfg3_iq3o_r7q4_m64_nw8k128"],
      "qg": pr["pfg3m_gdnqg_r7q4_m64_nw8k128"],
      "qkv_i3": pr["pfg3m_attnqkvi3_r7q4_m64_nw8k128"], "qkv_q6": pr["pfg3m_attnqkvq6_r7q4_m64_nw8k128"]}

gdn_blocks = [i for i in E.gdn_idx if ("fd", i) in W7][:8]
out_blocks = [i for i in E.gdn_idx if ("out", i) in W7][:8]
attn_i3 = [i for i in E.qtypes if E.qtypes[i] != 14 and ("k", i) in W7][:8]
attn_q6 = [i for i in E.qtypes if E.qtypes[i] == 14 and ("k", i) in W7][:8]
print(f"[boot] blocks: gdn(fd) {len(gdn_blocks)} out {len(out_blocks)} qkv_i3 {len(attn_i3)} qkv_q6 {len(attn_q6)}", flush=True)

def grab(nm, shape, dt_):
  return P.down(nm, shape, dt_).copy()

print("== G1: bit-identity (first clean run) ==", flush=True)
ALL_OK = True
# fd: distinct out bufs (xB64 ref, xA64 mine)
k_fd(SH["fd"], gdn_blocks[0], d["xB64"]); dev.synchronize()
r_fd = grab("xB64", (64, 5120), np.float32)
k_fd(Q4["fd"], gdn_blocks[0], d["xA64"]); dev.synchronize()
m1 = grab("xA64", (64, 5120), np.float32)
k_fd(Q4["fd"], gdn_blocks[0], d["xA64"]); dev.synchronize()
m2 = grab("xA64", (64, 5120), np.float32)
nz, det = int((m1 != r_fd).sum()), int((m1 != m2).sum())
ALL_OK &= (nz == 0 and det == 0)
print(f"[gate] fd(iq3d): nz={nz}/{m1.size} det-x2={det} -> {'BIT-IDENTICAL' if nz==0 and det==0 else 'DIFF'}", flush=True)
# out
k_out(SH["out"], out_blocks[0]); dev.synchronize()
r_out = grab("attn_out64", (64, 5120), np.float16)
k_out(Q4["out"], out_blocks[0]); dev.synchronize()
m1 = grab("attn_out64", (64, 5120), np.float16)
k_out(Q4["out"], out_blocks[0]); dev.synchronize()
m2 = grab("attn_out64", (64, 5120), np.float16)
nz, det = int((m1 != r_out).sum()), int((m1 != m2).sum())
ALL_OK &= (nz == 0 and det == 0)
print(f"[gate] out(iq3o): nz={nz}/{m1.size} det-x2={det} -> {'BIT-IDENTICAL' if nz==0 and det==0 else 'DIFF'}", flush=True)
# qg
k_qg(SH["qg"], gdn_blocks[0]); dev.synchronize()
r_qkvb, r_gateb = grab("qkv64", (64, 10240), np.float16), grab("gate64", (64, 6144), np.float16)
k_qg(Q4["qg"], gdn_blocks[0]); dev.synchronize()
m1a, m1b = grab("qkv64", (64, 10240), np.float16), grab("gate64", (64, 6144), np.float16)
k_qg(Q4["qg"], gdn_blocks[0]); dev.synchronize()
m2a, m2b = grab("qkv64", (64, 10240), np.float16), grab("gate64", (64, 6144), np.float16)
nz = int((m1a != r_qkvb).sum()) + int((m1b != r_gateb).sum())
det = int((m1a != m2a).sum()) + int((m1b != m2b).sum())
ALL_OK &= (nz == 0 and det == 0)
print(f"[gate] qg(gdnqg): nz={nz}/{m1a.size + m1b.size} det-x2={det} -> {'BIT-IDENTICAL' if nz==0 and det==0 else 'DIFF'}", flush=True)
# qkv (both flavors)
for tag, blks in [("qkv_i3", attn_i3), ("qkv_q6", attn_q6)]:
  if not blks: continue
  i0 = blks[0]
  k_qkv(SH[tag], i0); dev.synchronize()
  r1, r2, r3 = grab("qrow64", (64, 12288), np.float16), grab("krow64", (64, 1024), np.float16), grab("vrow64", (64, 1024), np.float16)
  k_qkv(Q4[tag], i0); dev.synchronize()
  a1, a2, a3 = grab("qrow64", (64, 12288), np.float16), grab("krow64", (64, 1024), np.float16), grab("vrow64", (64, 1024), np.float16)
  k_qkv(Q4[tag], i0); dev.synchronize()
  b1, b2, b3 = grab("qrow64", (64, 12288), np.float16), grab("krow64", (64, 1024), np.float16), grab("vrow64", (64, 1024), np.float16)
  nz = int((a1 != r1).sum()) + int((a2 != r2).sum()) + int((a3 != r3).sum())
  det = int((a1 != b1).sum()) + int((a2 != b2).sum()) + int((a3 != b3).sum())
  ALL_OK &= (nz == 0 and det == 0)
  print(f"[gate] {tag}: nz={nz} det-x2={det} -> {'BIT-IDENTICAL' if nz==0 and det==0 else 'DIFF'}", flush=True)
print(f"[G1] {'ALL BIT-IDENTICAL' if ALL_OK else 'IDENTITY FAILED'}", flush=True)

print("== G2: bench (synced min-of-10 over 8 blocks) ==", flush=True)
def bench(fn, blks, n=10):
  fn(blks[0]); dev.synchronize()
  best = 1e9
  for _ in range(n):
    t0 = time.perf_counter()
    for i in blks: fn(i)
    dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  return best / len(blks)

res = {}
for nm, fn, blks in [
    ("fd-m64",  lambda i: k_fd(SH["fd"], i, d["xB64"]), gdn_blocks),
    ("fd128",   lambda i: k_fd128(SH["fd"], i, d["xB128"]), gdn_blocks),
    ("out-m64", lambda i: k_out(SH["out"], i), out_blocks),
    ("qg-m64",  lambda i: k_qg(SH["qg"], i), gdn_blocks),
    ("qkv-i3",  lambda i: k_qkv(SH["qkv_i3"], i), attn_i3),
    ("qkv-q6",  lambda i: k_qkv(SH["qkv_q6"], i), attn_q6)]:
  res[nm] = bench(fn, blks)
  print(f"[bench] shipped {nm:8s} {res[nm]*1e6:8.1f} us/blk", flush=True)
for nm, fn, blks in [
    ("fd-m64",  lambda i: k_fd(Q4["fd"], i, d["xA64"]), gdn_blocks),
    ("fd128",   lambda i: k_fd128(Q4["fd"], i, d["xA128"]), gdn_blocks),
    ("out-m64", lambda i: k_out(Q4["out"], i), out_blocks),
    ("qg-m64",  lambda i: k_qg(Q4["qg"], i), gdn_blocks),
    ("qkv-i3",  lambda i: k_qkv(Q4["qkv_i3"], i), attn_i3),
    ("qkv-q6",  lambda i: k_qkv(Q4["qkv_q6"], i), attn_q6)]:
  t = bench(fn, blks)
  print(f"[bench] r7q4    {nm:8s} {t*1e6:8.1f} us/blk | speedup x{res[nm]/t:4.3f}", flush=True)
print("[r2d ring4] done", flush=True)
