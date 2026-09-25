# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c: w128h standalone corr + bench (pf17_probe pattern). corr: ONE 128-row
w128h window at pos P == 8 concatenated t32 windows at P..P+112 (the causal
law). det x2, poison-first, warm-up pair. bench: w128h vs 2x w64h per
128-row chunk-equivalent @pos {2032, 48000, 100288}.
Run: DEV=NV ~/tg311/bin/python -u r2c_w128h_corr.py"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

BASE = "~/tinygrad-metal/engine0"
dev = Device["NV"]
P = Bufs()
CTXK = 100352
rng = np.random.default_rng(11)
S13, NHP3, R32 = 13, 3, 32

_pc = {}
def prog(cub, ent):
  if (cub, ent) in _pc: return _pc[(cub, ent)]
  lib = open(f"{BASE}/{cub}.cubin", "rb").read()
  _pc[(cub, ent)] = NVProgram(dev, TinyELF(lib=lib, name=ent, target=dev.renderer.target, signature=tuple()))
  return _pc[(cub, ent)]

kv = np.clip(rng.integers(0, 256, (2*4*CTXK*256,)).astype(np.uint8), 0, 255)
P.poison("kv", 2*4*CTXK*256, np.uint8, 200); P.up("kv", kv)
sc = (rng.uniform(0.001, 0.02, (2*4*CTXK*8))).astype(np.float16)
P.poison("sc", 2*4*CTXK*8*2, np.float16, np.float16(7.7)); P.up("sc", sc)
qw = (rng.standard_normal(128*24*256) * 0.5).astype(np.float16)
P.poison("qw128p", 128*24*256*2, np.float16, np.float16(7.7)); P.up("qw128p", qw)
qrow128 = (rng.standard_normal(128*12288) * 0.3).astype(np.float16)
P.poison("qrow128p", 128*12288*2, np.float16, np.float16(7.7)); P.up("qrow128p", qrow128)
P.poison("ao16", 16*6144*2, np.float16, np.float16(7.7))
P.poison("pos_slot", 4, np.int32, -1)
P.poison("pmR", 4*S13*NHP3*R32*4, np.float32, 7.7e31)
P.poison("psR", 4*S13*NHP3*R32*4, np.float32, 7.7e31)
P.poison("pAR", 4*S13*NHP3*R32*256*4, np.float32, 7.7e31)
P.poison("qrowR", 16*12288*2, np.float16, np.float16(7.7))
MAXSLOT = 4*26*6*128   # 79872 covers w128h (39936) and w64 2x (19936 each)
P.poison("pmW", MAXSLOT*4, np.float32, 7.7e31)
P.poison("psW", MAXSLOT*4, np.float32, 7.7e31)
P.poison("pAW", MAXSLOT*256*4, np.float32, 7.7e31)
P.poison("qrowW", 128*12288*2, np.float16, np.float16(7.7))
P.poison("aoW", 128*6144*2, np.float16, np.float16(7.7))
P._keep.clear(); dev.synchronize()

def set_pos(pos):
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()

# warm-up control pair (the bare-world first-pair fault law)
_wu = 100336
P.win_up("pos_slot", 0, np.array([_wu], dtype=np.int32)); dev.synchronize()
prog("pfa32c_t32_s13_100k", "pfa32ct")(P.d["kv"], P.d["sc"], P.d["qw128p"], P.d["pos_slot"],
  P.d["pmR"], P.d["psR"], P.d["pAR"], global_size=(156,1,1), local_size=(512,1,1))
dev.synchronize()
prog("pfc16t_s13", "pfc16t")(P.d["pmR"], P.d["psR"], P.d["pAR"], P.d["qrowR"], P.d["ao16"],
  global_size=(24,1,1), local_size=(256,1,1))
dev.synchronize(); print(f"[wu] warm-up pair ok @{_wu}", flush=True)

def run_t32_pair(pos, k):
  gs = 4*S13*NHP3
  P.win_up("pmR", 0, np.full(gs*R32, 7.7e31, np.float32))
  P.win_up("psR", 0, np.full(gs*R32, 7.7e31, np.float32))
  P.win_up("pAR", 0, np.full(gs*R32*256, 7.7e31, np.float32))
  P.win_up("qrowR", 0, qrow128[k*16:(k+1)*16].copy())
  P.win_up("ao16", 0, np.full(16*6144, 7.7, dtype=np.float16))
  set_pos(pos); dev.synchronize()
  qwb = P.d["qw128p"].offset(offset=k*16*12288, size=16*12288)
  prog("pfa32c_t32_s13_100k", "pfa32ct")(P.d["kv"], P.d["sc"], qwb, P.d["pos_slot"],
    P.d["pmR"], P.d["psR"], P.d["pAR"], global_size=(gs,1,1), local_size=(512,1,1))
  prog("pfc16t_s13", "pfc16t")(P.d["pmR"], P.d["psR"], P.d["pAR"], P.d["qrowR"], P.d["ao16"],
    global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize()
  o = P.down("ao16", (16, 6144), np.float16).astype(np.float32).copy(); P._keep.clear()
  return o

def run_w128h(pos, rows=128):
  nhp, s, rmax = 6, 13, 128
  gs = 4*s*nhp; slots = gs*rmax
  P.win_up("pmW", 0, np.full(slots, 7.7e31, np.float32))
  P.win_up("psW", 0, np.full(slots, 7.7e31, np.float32))
  P.win_up("pAW", 0, np.full(slots*256, 7.7e31, np.float32))
  P.win_up("aoW", 0, np.full(rows*6144, 7.7, dtype=np.float16))
  P.win_up("qrowW", 0, qrow128[:rows].copy())
  set_pos(pos); dev.synchronize()
  prog("pfaw_w128h_s13_100k", "pfaw128h")(P.d["kv"], P.d["sc"], P.d["qw128p"], P.d["pos_slot"],
    P.d["pmW"], P.d["psW"], P.d["pAW"], global_size=(gs,1,1), local_size=(512,1,1))
  prog("pfcw128h_s13", "pfcw128h")(P.d["pmW"], P.d["psW"], P.d["pAW"], P.d["qrowW"], P.d["aoW"],
    global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize()
  o = P.down("aoW", (rows, 6144), np.float16).astype(np.float32).copy(); P._keep.clear()
  return o

def run_w64h_pair(pos, half):
  nhp, s, rmax = 6, 13, 64
  gs = 4*s*nhp; slots = gs*rmax
  P.win_up("pmW", 0, np.full(slots, 7.7e31, np.float32))
  P.win_up("psW", 0, np.full(slots, 7.7e31, np.float32))
  P.win_up("pAW", 0, np.full(slots*256, 7.7e31, np.float32))
  P.win_up("aoW", 0, np.full(64*6144, 7.7, dtype=np.float16))
  P.win_up("qrowW", 0, qrow128[half*64:(half+1)*64].copy())
  set_pos(pos + 64*half); dev.synchronize()
  qwb = P.d["qw128p"].offset(offset=half*64*12288, size=64*12288)
  prog("pfaw_w64h_s13_100k", "pfaw64h")(P.d["kv"], P.d["sc"], qwb, P.d["pos_slot"],
    P.d["pmW"], P.d["psW"], P.d["pAW"], global_size=(gs,1,1), local_size=(512,1,1))
  prog("pfcw64h_s13", "pfcw64h")(P.d["pmW"], P.d["psW"], P.d["pAW"], P.d["qrowW"], P.d["aoW"],
    global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize()
  o = P.down("aoW", (64, 6144), np.float16).astype(np.float32).copy(); P._keep.clear()
  return o

mode = sys.argv[1] if len(sys.argv) > 1 else "corr"
if mode == "corr":
  pos0 = int(os.getenv("POS", "100224"))
  refs = [run_t32_pair(pos0 + 16*k, k) for k in range(8)]
  ref = np.concatenate(refs, axis=0)
  try:
    o1 = run_w128h(pos0); o2 = run_w128h(pos0)
    det = np.array_equal(o1, o2)
    nz = int((o1 != ref).sum())
    print(f"[corr] w128h @pos{pos0} rows=128: nz={nz}/{o1.size} maxabsdiff {float(np.abs(o1-ref).max()):.3e} "
          f"| det x2 {det} | poison {(o1 == 7.7).sum()}", flush=True)
  except Exception as e:
    print(f"[corr] w128h LAUNCH FAIL {type(e).__name__}: {e}", flush=True)
else:
  def bench(name, fn):
    for _ in range(2): fn()
    dev.synchronize()
    best = 1e9
    for _ in range(5):
      t0 = time.perf_counter(); fn(); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    print(f"[bench] {name}: {best*1e3:8.2f} ms", flush=True)
    return best*1e3
  for pos in (2032, 48000, 100288):
    print(f"---- pos {pos} (per 128 rows) ----", flush=True)
    set_pos(pos)
    bench("w128h", lambda: run_w128h(pos))
    bench("2x w64h", lambda: (run_w64h_pair(pos, 0), run_w64h_pair(pos, 1)))
print("[done]", flush=True)
