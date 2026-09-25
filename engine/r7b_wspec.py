# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7b rung-2b A/B: warp-spec pf_gemm3w vs base pf_gemm3 at TRUE M128 in-plan
shapes. Bit-identity (nz=0, det-x2) on real weights + perf min-of-8.
Env: NV_QMD_BARRIERS=16 NV_QMD_BARRIERS_NAMES=pfg3w (the fork QMD patch)."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("NV_QMD_BARRIERS", "16")
os.environ.setdefault("NV_QMD_BARRIERS_NAMES", "pfg3w")
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
NEW = ["pfg3w_ffn_r7_m64_ws6p2k64", "pfg3w_iq3d_r7_m64_ws10p2k64",
       "pfg3w_iq3o_r7_m64_ws10p2k64", "pfg3w_iq3d_r7_m64_ws11p3k64"]
for n in NEW:
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
dev.synchronize(); P._keep.clear()

W7, W = E._pf_W7, E.W
LS = (256, 1, 1)
def fill(nm, shape, dt_, seed):
  a = (np.random.default_rng(seed).standard_normal(shape) * 0.7)
  P.win_up(nm, 0, a.astype(dt_).reshape(-1))
fill("hhx128", (128, 5120), np.float16, 12)
fill("gact128", (128, 17408), np.float16, 11)
fill("z128", (128, 6144), np.float16, 13)
fill("hh128", (128, 5120), np.float32, 10)
fill("xB128", (128, 5120), np.float32, 9)
dev.synchronize()

def grab(nm, shape, dt_): return P.down(nm, shape, dt_).copy()

gdn_blocks = [i for i in E.gdn_idx if ("fd", i) in W7][:6]
out_blocks = [i for i in E.gdn_idx if (not E.gdn_oq8[i]) and ("out", i) in W7][:6]
ffn_blocks = [i for i in E.gdn_idx if ("fg", i) in W7][:6]

def ffn_b(p, i):  p(W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx128"], d["gact128"], global_size=(1088,1,1), local_size=(128,1,1))
def ffn_w(p, i):  p(W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx128"], d["gact128"], global_size=(1088,1,1), local_size=(192,1,1))
def fd_b(p, i):   p(W7[("fd", i)], d["gridf"], d["gact128"], d["hh128"], d["xB128"], global_size=(160,1,1), local_size=LS)
def fd_w(p, i):   p(W7[("fd", i)], d["gridf"], d["gact128"], d["hh128"], d["xB128"], global_size=(160,1,1), local_size=(320,1,1))
def fd_w3(p, i):  p(W7[("fd", i)], d["gridf"], d["gact128"], d["hh128"], d["xB128"], global_size=(160,1,1), local_size=(352,1,1))
def out_b(p, i):  p(W7[("out", i)], d["gridf"], d["z128"], d["attn_out128"], global_size=(160,1,1), local_size=LS)
def out_w(p, i):  p(W7[("out", i)], d["gridf"], d["z128"], d["attn_out128"], global_size=(160,1,1), local_size=(320,1,1))

FFN_B, FFN_W = pr["pfg3_ffn_r7_m64_nw4k128"], pr["pfg3w_ffn_r7_m64_ws6p2k64"]
FD_B, FD_W, FD_W3 = pr["pfg3_iq3d_r7_m64_nw8k128"], pr["pfg3w_iq3d_r7_m64_ws10p2k64"], pr["pfg3w_iq3d_r7_m64_ws11p3k64"]
OUT_B, OUT_W = pr["pfg3_iq3o_r7_m64_nw8k128"], pr["pfg3w_iq3o_r7_m64_ws10p2k64"]

def ab(name, base_fn, new_fn, newp, outnm, outshape, outdt, blocks):
  base_fn(blocks[0]); dev.synchronize()
  r1 = grab(outnm, outshape, outdt)
  new_fn(blocks[0]); dev.synchronize()
  m1 = grab(outnm, outshape, outdt)
  new_fn(blocks[0]); dev.synchronize()
  m2 = grab(outnm, outshape, outdt)
  nz = int((m1 != r1).sum()) + int((m2 != m1).sum())
  # perf
  def perf(fn):
    for _ in range(3): fn(blocks[0]); dev.synchronize()
    best = 1e9
    for _ in range(8):
      t0 = time.perf_counter()
      for i in blocks: fn(i)
      dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    return best / len(blocks)
  pb, pn = perf(base_fn), perf(new_fn)
  print(f"[ab] {name}: nz={nz} -> {'BIT-IDENTICAL det-x2' if nz==0 else 'DIFF'} | "
        f"base {pb*1e6:8.1f}us  wspec {pn*1e6:8.1f}us  x{pb/pn:.3f}", flush=True)
  return nz, pb, pn

print("== warp-spec A/B (TRUE shapes) ==", flush=True)
ab("ffn", lambda i: ffn_b(FFN_B, i), lambda i: ffn_w(FFN_W, i), FFN_W, "gact128", (128, 17408), np.float16, ffn_blocks)
ab("fd",  lambda i: fd_b(FD_B, i),   lambda i: fd_w(FD_W, i),   FD_W,  "xB128", (128, 5120), np.float32, gdn_blocks)
ab("fd3", lambda i: fd_b(FD_B, i),   lambda i: fd_w3(FD_W3, i), FD_W3, "xB128", (128, 5120), np.float32, gdn_blocks)
ab("out", lambda i: out_b(OUT_B, i), lambda i: out_w(OUT_W, i), OUT_W, "attn_out128", (128, 5120), np.float16, out_blocks)
print("[ab] DONE")
