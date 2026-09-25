# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2H v3: QMD field diff (hm vs padded-scalar control, both cfg-17) +
correct-thread-count direct launch test."""
import os, sys, json
os.environ["SKV"] = "1"; os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine, RBLK, CBLK, SLICE, CTXK
from engine0 import dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

SNAP = os.getenv("SNAPDIR", "~/snap100k")
BASE = "~/tinygrad-metal/engine0"
meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()

def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

E = MTPEngine(theta=1e7)   # loads kernels incl. HM a3
pad = prog("spk_g4nw32qh3ppad_100k")   # the in-graph-WORKING cfg-17 control
hm = E.pr["spk_a3"]
pre = E.pr["spk_pre3"]
print(f"[info] hm: regs {hm.regs_usage} shmem {hm.shmem_usage} max_threads {hm.max_threads}", flush=True)
print(f"[info] pad: regs {pad.regs_usage} shmem {pad.shmem_usage} max_threads {pad.max_threads}", flush=True)
print(f"[info] pre: regs {pre.regs_usage} shmem {pre.shmem_usage} max_threads {pre.max_threads}", flush=True)

from tinygrad.runtime.ops_nv import QMD
fields = sorted(QMD.fields[hm.qmd.pref].keys())
diffs = []
for k in fields:
  a, b = hm.qmd.read(k), pad.qmd.read(k)
  if a != b: diffs.append((k, a, b))
print(f"[qmd] {len(fields)} fields, {len(diffs)} diffs hm-vs-pad", flush=True)
for k, a, b in diffs: print(f"  QMDDIFF {k}: hm={a} pad={b}", flush=True)

# ---------- direct launch at CORRECT local size (1024), no graph needed ----------
d = E.P.d
E.P.poison("kv0", 2 * 4 * CTXK * 256, np.uint8, 100)
E.P.poison("sc0", 2 * 4 * CTXK * 8 * 2, np.float16, 0.01)
E.P.poison("qw16_3", 3 * 24 * 256 * 2, np.float16, 0.05)
E.P.poison("pm3", 4 * 256 * 18 * 4, np.float32, 7.7e31)
E.P.poison("ps3", 4 * 256 * 18 * 4, np.float32, 7.7e31)
E.P.poison("pA3", 4 * 256 * 18 * 256 * 4, np.float32, 7.7e31)
E.P.poison("pos_slot", 4, np.int32, -12345)
E.P.up("pos_slot", np.array([P0], dtype=np.int32))
E._flush(); dev.synchronize()
args = (d["kv0"], d["sc0"], d["qw16_3"], d["pos_slot"], d["pm3"], d["ps3"], d["pA3"])
G3 = 4 * 256
print(f"[direct] hm spk_g4nwhm3_100k grid={G3} local=(1024,1,1)", flush=True)
hm(*args, global_size=(G3, 1, 1), local_size=(1024, 1, 1), wait=True)
for nm, shape in (("pm3", (4 * 256 * 18,)), ("ps3", (4 * 256 * 18,)), ("pA3", (4 * 256 * 18 * 256,))):
  v = E.P.down(nm, shape, np.float32)
  npois = int((v == 7.7e31).sum())
  fin = v[np.isfinite(v) & (v != 7.7e31)]
  rng = f"min {fin.min():.5g} max {fin.max():.5g} med {np.median(fin):.5g}" if fin.size else "EMPTY"
  print(f"[hm1024] {nm}: poison {npois}/{v.size} finite {fin.size} | {rng}", flush=True)
# control: same launch shape with the padded scalar
E.P.poison("pm3", 4 * 256 * 18 * 4, np.float32, 7.7e31)
E.P.poison("ps3", 4 * 256 * 18 * 4, np.float32, 7.7e31)
E.P.poison("pA3", 4 * 256 * 18 * 256 * 4, np.float32, 7.7e31)
E._flush(); dev.synchronize()
print(f"[direct] pad spk_g4nw32qh3ppad_100k grid={G3} local=(1024,1,1)", flush=True)
pad(*args, global_size=(G3, 1, 1), local_size=(1024, 1, 1), wait=True)
for nm, shape in (("pm3", (4 * 256 * 18,)), ("ps3", (4 * 256 * 18,))):
  v = E.P.down(nm, shape, np.float32)
  npois = int((v == 7.7e31).sum())
  fin = v[np.isfinite(v) & (v != 7.7e31)]
  print(f"[pad1024] {nm}: poison {npois}/{v.size} finite {fin.size} | " + (f"min {fin.min():.5g} max {fin.max():.5g}" if fin.size else "EMPTY"), flush=True)
print("[dbg done]", flush=True)
