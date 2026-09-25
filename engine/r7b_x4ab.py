# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7b X-uint4 A/B: rebuilt cubins vs the .prex4.bak originals — bit-identity
(nz=0, det-x2) + perf at TRUE M128 shapes."""
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
CLS = ["pfg3_ffn_r7_m64_nw4k128", "pfg3_iq3d_r7_m64_nw8k128", "pfg3_iq3o_r7_m64_nw8k128",
       "pfg3m_gdnqg_r7q4_m64_nw8k128", "pfg3m_attnqkvi3_r7_m64_nw8k128", "pfg3m_attnqkvq6_r7_m64_nw8k128"]
old, new = {}, {}
for n in CLS:
  old[n] = NVProgram(dev, TinyELF(lib=open(f"{BASE}/{n}.cubin.prex4.bak", "rb").read(), name=n,
                                   target=dev.renderer.target, signature=tuple()))
  # the in-plan program objects (pr) already point at the REBUILT cubin files (loaded at boot)
dev.synchronize(); P._keep.clear()

W7, W = E._pf_W7, E.W
LS = (256, 1, 1)
def fill(nm, shape, dt_, seed):
  a = (np.random.default_rng(seed).standard_normal(shape) * 0.7)
  P.win_up(nm, 0, a.astype(dt_).reshape(-1))
fill("xh128", (128, 5120), np.float16, 14)
fill("z128", (128, 6144), np.float16, 13)
fill("hhx128", (128, 5120), np.float16, 12)
fill("gact128", (128, 17408), np.float16, 11)
fill("hh128", (128, 5120), np.float32, 10)
fill("xB128", (128, 5120), np.float32, 9)
dev.synchronize()
def grab(nm, shape, dt_): return P.down(nm, shape, dt_).copy()
def V(nm, off, sz): return d[nm].offset(offset=off, size=sz)

gdn_blocks = [i for i in E.gdn_idx if ("fd", i) in W7][:6]
out_blocks = [i for i in E.gdn_idx if (not E.gdn_oq8[i]) and ("out", i) in W7][:6]
ffn_blocks = [i for i in E.gdn_idx if ("fg", i) in W7][:6]
attn_i3 = [i for i in E.qtypes if E.qtypes[i] != 14 and ("k", i) in W7][:3]
attn_q6 = [i for i in E.qtypes if E.qtypes[i] == 14 and ("k", i) in W7][:3]

def mk(nm):
  p = pr[nm]; o = old[nm]
  if nm.startswith("pfg3_ffn"):
    def b(pp, i): pp(W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx128"], d["gact128"], global_size=(1088,1,1), local_size=(128,1,1))
    return (lambda i: b(p, i), lambda i: b(o, i)), "gact128", (128, 17408), np.float16, ffn_blocks
  if nm.startswith("pfg3_iq3d"):
    def b(pp, i): pp(W7[("fd", i)], d["gridf"], d["gact128"], d["hh128"], d["xB128"], global_size=(160,1,1), local_size=LS)
    return (lambda i: b(p, i), lambda i: b(o, i)), "xB128", (128, 5120), np.float32, gdn_blocks
  if nm.startswith("pfg3_iq3o"):
    def b(pp, i): pp(W7[("out", i)], d["gridf"], d["z128"], d["attn_out128"], global_size=(160,1,1), local_size=LS)
    return (lambda i: b(p, i), lambda i: b(o, i)), "attn_out128", (128, 5120), np.float16, out_blocks
  if nm.startswith("pfg3m_gdnqg"):
    def b(pp, i): pp(W[("qkv", i)], W7[("gate", i)], d["gridf"], d["xh128"], d["qkv128"], d["gate128"], global_size=(512,1,1), local_size=LS)
    outs = []
    def gnew(i):
      bnew(i); outs.append((grab("qkv128", (128, 10240), np.float16), grab("gate128", (128, 6144), np.float16)))
    return (lambda i: b(p, i), lambda i: b(o, i)), None, None, None, gdn_blocks
  if nm.startswith("pfg3m_attnqkv"):
    ii = attn_i3 if nm.endswith("i3") else attn_q6
    def b(pp, i):
      qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
      pp(qw_, W7[("k", i)], W[("v", i)], d["gridf"], d["xh128"], d["qrow128"], d["krow128"], d["vrow128"], global_size=(448,1,1), local_size=LS)
    return (lambda i: b(p, i), lambda i: b(o, i)), "qrow128", (128, 12288), np.float16, ii
  raise ValueError(nm)

print("== X-uint4 A/B (old .prex4 vs rebuilt) ==", flush=True)
for nm in CLS:
  (fnew, fold), onm, oshape, odt, blocks = mk(nm)
  if onm is None:
    # gdnqg: dual-output compare
    fnew(blocks[0]); dev.synchronize()
    r1 = grab("qkv128", (128, 10240), np.float16); r2 = grab("gate128", (128, 6144), np.float16)
    fold(blocks[0]); dev.synchronize()
    m1 = grab("qkv128", (128, 10240), np.float16); m2 = grab("gate128", (128, 6144), np.float16)
    fnew(blocks[0]); dev.synchronize()
    n1 = grab("qkv128", (128, 10240), np.float16); n2 = grab("gate128", (128, 6144), np.float16)
    nz = int((m1 != r1).sum()) + int((m2 != r2).sum()) + int((n1 != m1).sum()) + int((n2 != m2).sum())
  else:
    fnew(blocks[0]); dev.synchronize()
    r1 = grab(onm, oshape, odt)
    fold(blocks[0]); dev.synchronize()
    m1 = grab(onm, oshape, odt)
    fnew(blocks[0]); dev.synchronize()
    n1 = grab(onm, oshape, odt)
    nz = int((m1 != r1).sum()) + int((n1 != m1).sum())
  def perf(fn):
    for _ in range(3): fn(blocks[0]); dev.synchronize()
    best = 1e9
    for _ in range(8):
      t0 = time.perf_counter()
      for i in blocks: fn(i)
      dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    return best / len(blocks)
  po, pn = perf(fold), perf(fnew)
  print(f"[x4] {nm:34s}: nz={nz} {'BID' if nz==0 else 'DIFF'} | old {po*1e6:8.1f}us new {pn*1e6:8.1f}us x{po/pn:.3f}", flush=True)
print("[x4] DONE")
