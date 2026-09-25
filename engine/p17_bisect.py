# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P17 bisect v2: ONE process, ladder of attn+combine pairs over transformed
worlds; stops at the first fault (print shows the step). Steps:
  1: pf10-bench-EXACT world (qw 32-row full ptr, pm/ps/pA 12288 slots, pos 100336)
  2: + qw buffer grown to 64 rows (full ptr)
  3: + pm/ps/pA at EXACT 4992 slots
  4: + the extra P17 world buffers (pmW/psW/pAW/qrow64/aoW)
  5: + qw as an offset VIEW + qrowR swap + pos 100224 (my exact corr shape)
"""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

BASE = "~/tinygrad-metal/engine0"
dev = Device["NV"]; P = Bufs()
CTXK = 100352
rng = np.random.default_rng(11)

def prog(cub, ent):
  lib = open(f"{BASE}/{cub}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=ent, target=dev.renderer.target, signature=tuple()))

kv = np.clip(rng.integers(0, 256, (2*4*CTXK*256,)).astype(np.uint8), 0, 255)
P.poison("kv", 2*4*CTXK*256, np.uint8, 200); P.up("kv", kv)
sc = (rng.uniform(0.001, 0.02, (2*4*CTXK*8))).astype(np.float16)
P.poison("sc", 2*4*CTXK*8*2, np.float16, np.float16(7.7)); P.up("sc", sc)
qw = (rng.standard_normal(32*24*256) * 0.5).astype(np.float16)
P.poison("qw", 32*24*256*2, np.float16, np.float16(7.7)); P.up("qw", qw)
qrow16 = (rng.standard_normal(16*12288) * 0.3).astype(np.float16)
P.poison("qrow16", 16*12288*2, np.float16, np.float16(7.7)); P.up("qrow16", qrow16)
P.poison("ao16", 16*6144*2, np.float16, np.float16(7.7))
P.poison("atc_ctr", 4, np.uint32, 0xFF); P.up("atc_ctr", np.zeros(1, dtype=np.uint32))
P.poison("pos_slot", 4, np.int32, -1)
P._keep.clear(); dev.synchronize()

pka = prog("pfa32c_t32_s13_100k", "pfa32ct")
pkc = prog("pfc16t_s13", "pfc16t")

def pair(tag, pos, qwptr, pmb, psb, pab, qrowptr):
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
  pka(P.d["kv"], P.d["sc"], qwptr, P.d["pos_slot"], pmb, psb, pab,
      global_size=(156,1,1), local_size=(512,1,1))
  dev.synchronize(); print(f"[{tag}] attn ok", flush=True)
  pkc(pmb, psb, pab, qrowptr, P.d["ao16"], global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize(); print(f"[{tag}] combine ok", flush=True)

# step 1: pf10-bench exact
P.poison("pm", 4*32*96*4, np.float32, 7.7e31); P.poison("ps", 4*32*96*4, np.float32, 7.7e31)
P.poison("pA", 4*32*96*256*4, np.float32, 7.7e31); dev.synchronize(); P._keep.clear()
pair("s1-pf10world", 100336, P.d["qw"], P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow16"])

# step 2: qw buffer grown to 64 rows (new alloc + full ptr)
qw64 = (rng.standard_normal(64*24*256) * 0.5).astype(np.float16)
P.poison("qw64", 64*24*256*2, np.float16, np.float16(7.7)); P.up("qw64", qw64)
dev.synchronize(); P._keep.clear()
pair("s2-qw64", 100336, P.d["qw64"], P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow16"])

# step 3: partials at EXACT 4992 slots
P.poison("pmR", 4992*4, np.float32, 7.7e31); P.poison("psR", 4992*4, np.float32, 7.7e31)
P.poison("pAR", 4992*256*4, np.float32, 7.7e31); dev.synchronize(); P._keep.clear()
pair("s3-exact-slots", 100336, P.d["qw64"], P.d["pmR"], P.d["psR"], P.d["pAR"], P.d["qrow16"])

# step 4: + the extra P17 world buffers
P.poison("qrow64", 64*12288*2, np.float16, np.float16(7.7))
P.up("qrow64", (rng.standard_normal(64*12288) * 0.3).astype(np.float16))
P.poison("aoW", 64*6144*2, np.float16, np.float16(7.7))
P.poison("pmW", 39936*4, np.float32, 7.7e31); P.poison("psW", 39936*4, np.float32, 7.7e31)
P.poison("pAW", 39936*256*4, np.float32, 7.7e31)
dev.synchronize(); P._keep.clear()
pair("s4-extrabufs", 100336, P.d["qw64"], P.d["pmR"], P.d["psR"], P.d["pAR"], P.d["qrow16"])

# step 5: qw as offset VIEW + qrowR + pos 100224 (my exact corr shape)
qwb = P.d["qw64"].offset(offset=0, size=16*12288)
P.up("qrowR", qrow16.copy())
dev.synchronize(); P._keep.clear()
pair("s5-view+pos", 100224, qwb, P.d["pmR"], P.d["psR"], P.d["pAR"], P.d["qrowR"])
print("[bisect] ALL STEPS PASSED", flush=True)

# step 6: + the 5 win_up poisons immediately before the launches (the probe's run_t32_pair shape)
P.win_up("pmR", 0, np.full(156*32, 7.7e31, np.float32))
P.win_up("psR", 0, np.full(156*32, 7.7e31, np.float32))
P.win_up("pAR", 0, np.full(156*32*256, 7.7e31, np.float32))
P.win_up("qrowR", 0, qrow16.copy())
P.win_up("ao16", 0, np.full(16*6144, 7.7, dtype=np.float16))
P.win_up("pos_slot", 0, np.array([100224], dtype=np.int32)); dev.synchronize()
pair("s6-winups", 100224, qwb, P.d["pmR"], P.d["psR"], P.d["pAR"], P.d["qrowR"])
print("[bisect] STEP6 PASSED", flush=True)

# ---- step 7+: the P17 corr ladder (refs + wide shapes) in this warmed process ----
qrow64np = (rng.standard_normal(64*12288) * 0.3).astype(np.float16)
P.up("qrow64b", qrow64np); dev.synchronize(); P._keep.clear()

def t32_ref(pos, k):
  P.win_up("pmR", 0, np.full(4992, 7.7e31, np.float32))
  P.win_up("psR", 0, np.full(4992, 7.7e31, np.float32))
  P.win_up("pAR", 0, np.full(4992*256, 7.7e31, np.float32))
  P.win_up("qrowR", 0, qrow64np[k*16:(k+1)*16].copy())
  P.win_up("ao16", 0, np.full(16*6144, 7.7, dtype=np.float16))
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
  qwbk = P.d["qw64"].offset(offset=k*16*12288, size=16*12288)
  pka(P.d["kv"], P.d["sc"], qwbk, P.d["pos_slot"], P.d["pmR"], P.d["psR"], P.d["pAR"],
      global_size=(156,1,1), local_size=(512,1,1))
  dev.synchronize()
  pkc(P.d["pmR"], P.d["psR"], P.d["pAR"], P.d["qrowR"], P.d["ao16"],
      global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize()
  o = P.down("ao16", (16, 6144), np.float16).astype(np.float32).copy()
  P._keep.clear()
  print(f"[ref] k={k} pos={pos} ok", flush=True)
  return o

def wide_run(nm):
  acub, aent, ccub, cent, rows, hrp, nw = {
    "w32":  ("pfaw_w32_s13_100k", "pfaw32", "pfcw32_s13", "pfcw32", 32, 2, 16),
    "w64":  ("pfaw_w64_s13_100k", "pfaw64", "pfcw64_s13", "pfcw64", 64, 2, 16),
    "w64q": ("pfaw_w64q_s13_100k", "pfaw64q", "pfcw64_s13", "pfcw64", 64, 2, 32),
    "w64h": ("pfaw_w64h_s13_100k", "pfaw64h", "pfcw64h_s13", "pfcw64h", 64, 1, 16)}[nm]
  nhp = 6 // hrp; gs = 4*13*nhp; rmax = hrp*rows; slots = gs*rmax
  P.win_up("pmW", 0, np.full(slots, 7.7e31, np.float32))
  P.win_up("psW", 0, np.full(slots, 7.7e31, np.float32))
  P.win_up("pAW", 0, np.full(slots*256, 7.7e31, np.float32))
  P.win_up("aoW", 0, np.full(rows*6144, 7.7, dtype=np.float16))
  P.up("qrowW", qrow64np[:rows].copy())
  P.win_up("pos_slot", 0, np.array([100224], dtype=np.int32)); dev.synchronize()
  pmb = P.d["pmW"].offset(offset=0, size=slots*4); psb = P.d["psW"].offset(offset=0, size=slots*4)
  pab = P.d["pAW"].offset(offset=0, size=slots*256*4); aob = P.d["aoW"].offset(offset=0, size=rows*6144*2)
  qrb = P.d["qrowW"].offset(offset=0, size=rows*12288*2)
  prog(acub, aent)(P.d["kv"], P.d["sc"], P.d["qw64"], P.d["pos_slot"], pmb, psb, pab,
    global_size=(gs,1,1), local_size=(nw*32,1,1))
  dev.synchronize(); print(f"[wide] {nm} attn ok", flush=True)
  prog(ccub, cent)(pmb, psb, pab, qrb, aob, global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize(); print(f"[wide] {nm} combine ok", flush=True)
  o = P.down("aoW", (rows, 6144), np.float16).astype(np.float32).copy()
  P._keep.clear()
  return o

pos0 = 100224
refs = [t32_ref(pos0 + 16*k, k) for k in range(4)]
ref = np.concatenate(refs, axis=0)
for nm in ("w32", "w64", "w64h", "w64q"):
  try:
    o1 = wide_run(nm); o2 = wide_run(nm)
  except Exception as e:
    print(f"[corr] {nm}: LAUNCH FAIL {type(e).__name__}: {e}", flush=True); break
  rows = {"w32": 32, "w64": 64, "w64h": 64, "w64q": 64}[nm]
  det = np.array_equal(o1, o2)
  nz = int((o1 != ref[:rows]).sum())
  mad = float(np.abs(o1 - ref[:rows]).max())
  e = np.abs(o1 - ref[:rows]) / np.maximum(np.abs(ref[:rows]), 1e-3)
  print(f"[corr] {nm} @pos{pos0} rows={rows}: BIT-IDENTICAL nz={nz}/{o1.size} maxabsdiff {mad:.3e} "
        f"relerr med {np.median(e):.3e} max {np.max(e):.3e} | det x2 {det} | poison {(o1==7.7).sum()}", flush=True)
print("[bisect] CORR LADDER DONE", flush=True)
