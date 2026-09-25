# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P13 isolation: why does the ref path fault with concatenated+offset weights?
Stage A: P7B-style — separate per-block buffers, one launch (the proven path).
Stage B: concatenated buffer, offset(0) slice.
Stage C: concatenated buffer, offset(b*WBLK) for b=1.
Stage D: 8 back-to-back offset launches (my harness's exact ref loop).
"""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
P7 = f"{BASE}/packed7"
KD, ND, NBLK = 5120, 17408, 8
NGRID, NGRP, NCH = 272, 8, 40
WBLK = NGRID * NGRP * NCH * 32 * 16
LS = (256, 1, 1)
P = Bufs()
lib = open(f"{BASE}/pfg3_ffn_r7_m32_nw8k128.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name="pfg3_ffn_r7_m32_nw8k128", target=dev.renderer.target, signature=tuple()))
P.up("gridf", iq3_grid_f32())
rng = np.random.default_rng(7)
P.up("x", (rng.standard_normal((32, KD)) * 0.8).astype(np.float16).reshape(-1))
P.poison("o", 32 * ND, np.float16, 7.7)
dev.synchronize()
print("[iso] buffers up", flush=True)

def launch(w1b, w2b, xb, ob):
  pr(w1b, w2b, P.d["gridf"], xb, ob, global_size=(NGRID, 1, 1), local_size=LS)

stage = sys.argv[1] if len(sys.argv) > 1 else "A"
if stage == "A":
  P.up("fg0", np.load(f"{P7}/fg0.npy")); P.up("fu0", np.load(f"{P7}/fu0.npy"))
  dev.synchronize(); print("[iso A] separate buffers, 1 launch...", flush=True)
  launch(P.d["fg0"], P.d["fu0"], P.d["x"], P.d["o"]); dev.synchronize()
  v = P.down("o", (32, ND), np.float16)
  print(f"[iso A] CLEAN outsum={float(np.abs(v.astype(np.float32)).sum()):.1f}", flush=True)
elif stage == "B":
  fg = np.concatenate([np.load(f"{P7}/fg{b}.npy") for b in range(NBLK)])
  fu = np.concatenate([np.load(f"{P7}/fu{b}.npy") for b in range(NBLK)])
  P.up("w1", fg); P.up("w2", fu); del fg, fu
  dev.synchronize(); print(f"[iso B] concat up, offset(0) launch...", flush=True)
  w1b = P.d["w1"].offset(offset=0, size=WBLK); w2b = P.d["w2"].offset(offset=0, size=WBLK)
  launch(w1b, w2b, P.d["x"], P.d["o"]); dev.synchronize()
  print("[iso B] CLEAN", flush=True)
elif stage == "C":
  fg = np.concatenate([np.load(f"{P7}/fg{b}.npy") for b in range(NBLK)])
  fu = np.concatenate([np.load(f"{P7}/fu{b}.npy") for b in range(NBLK)])
  P.up("w1", fg); P.up("w2", fu); del fg, fu
  dev.synchronize(); print("[iso C] concat up, offset(1*WBLK) launch...", flush=True)
  w1b = P.d["w1"].offset(offset=WBLK, size=WBLK); w2b = P.d["w2"].offset(offset=WBLK, size=WBLK)
  launch(w1b, w2b, P.d["x"], P.d["o"]); dev.synchronize()
  print("[iso C] CLEAN", flush=True)
elif stage == "D":
  fg = np.concatenate([np.load(f"{P7}/fg{b}.npy") for b in range(NBLK)])
  fu = np.concatenate([np.load(f"{P7}/fu{b}.npy") for b in range(NBLK)])
  P.up("w1", fg); P.up("w2", fu); del fg, fu
  dev.synchronize(); print("[iso D] concat up, 8 back-to-back offset launches...", flush=True)
  for b in range(NBLK):
    w1b = P.d["w1"].offset(offset=b * WBLK, size=WBLK)
    w2b = P.d["w2"].offset(offset=b * WBLK, size=WBLK)
    launch(w1b, w2b, P.d["x"], P.d["o"])
  dev.synchronize()
  print("[iso D] CLEAN", flush=True)
elif stage == "E":  # exact P7B replica: classic + P6 kernel prelude, then r7
  P.up("w_fg", np.load(f"{BASE}/packed/fg0.npy")); P.up("w_fu", np.load(f"{BASE}/packed/fu0.npy"))
  P.up("r_fg", np.load(f"{P7}/fg0.npy")); P.up("r_fu", np.load(f"{P7}/fu0.npy"))
  P.up("x64", (rng.standard_normal((64, KD)) * 0.8).astype(np.float16).reshape(-1))
  P.poison("ref64", 64 * ND, np.float16, 7.7); P.poison("mine64", 64 * ND, np.float16, 7.7)
  dev.synchronize(); print("[iso E] prelude launch (P6 classic)...", flush=True)
  lib6 = open(f"{BASE}/pfg_ffn_m32_hm_nw8k128.cubin", "rb").read()
  pr6 = NVProgram(dev, TinyELF(lib=lib6, name="pfg_ffn_m32_hm_nw8k128", target=dev.renderer.target, signature=tuple()))
  for i in range(2):
    x = P.d["x64"] if i == 0 else P.d["x64"].offset(offset=32*KD*2, size=32*KD*2)
    o = P.d["ref64"] if i == 0 else P.d["ref64"].offset(offset=32*ND*2, size=32*ND*2)
    pr6(P.d["w_fg"], P.d["w_fu"], P.d["gridf"], x, o, global_size=(NGRID,1,1), local_size=LS)
  dev.synchronize(); print("[iso E] P6 prelude CLEAN; now r7 (2 launches)...", flush=True)
  for i in range(2):
    x = P.d["x64"] if i == 0 else P.d["x64"].offset(offset=32*KD*2, size=32*KD*2)
    o = P.d["mine64"] if i == 0 else P.d["mine64"].offset(offset=32*ND*2, size=32*ND*2)
    pr(P.d["r_fg"], P.d["r_fu"], P.d["gridf"], x, o, global_size=(NGRID,1,1), local_size=LS)
  dev.synchronize()
  ref = P.down("ref64", (64, ND), np.float16); mine = P.down("mine64", (64, ND), np.float16)
  print(f"[iso E] CLEAN nz={int((mine != ref).sum())}", flush=True)
elif stage == "F":  # r7 only, P7B buffer shapes (64 rows), 2 launches, NO prelude
  P.up("r_fg", np.load(f"{P7}/fg0.npy")); P.up("r_fu", np.load(f"{P7}/fu0.npy"))
  P.up("x64", (rng.standard_normal((64, KD)) * 0.8).astype(np.float16).reshape(-1))
  P.poison("mine64", 64 * ND, np.float16, 7.7)
  dev.synchronize(); print("[iso F] r7 only, 64-row, 2 launches...", flush=True)
  for i in range(2):
    x = P.d["x64"] if i == 0 else P.d["x64"].offset(offset=32*KD*2, size=32*KD*2)
    o = P.d["mine64"] if i == 0 else P.d["mine64"].offset(offset=32*ND*2, size=32*ND*2)
    pr(P.d["r_fg"], P.d["r_fu"], P.d["gridf"], x, o, global_size=(NGRID,1,1), local_size=LS)
  dev.synchronize(); print("[iso F] CLEAN", flush=True)
elif stage == "G":  # r7, 32-row buffers, TWO back-to-back launches
  P.up("r_fg", np.load(f"{P7}/fg0.npy")); P.up("r_fu", np.load(f"{P7}/fu0.npy"))
  dev.synchronize(); print("[iso G] r7, 32-row, 2 launches...", flush=True)
  launch(P.d["r_fg"], P.d["r_fu"], P.d["x"], P.d["o"])
  launch(P.d["r_fg"], P.d["r_fu"], P.d["x"], P.d["o"])
  dev.synchronize(); print("[iso G] CLEAN", flush=True)
elif stage == "A2":  # P7B allocation ORDER: weights FIRST, then x, then poison
  P.up("fg0", np.load(f"{P7}/fg0.npy")); P.up("fu0", np.load(f"{P7}/fu0.npy"))
  P.up("x2", (rng.standard_normal((32, KD)) * 0.8).astype(np.float16).reshape(-1))
  P.poison("o2", 32 * ND, np.float16, 7.7)
  dev.synchronize(); print("[iso A2] weights-first order, 1 launch...", flush=True)
  launch(P.d["fg0"], P.d["fu0"], P.d["x2"], P.d["o2"]); dev.synchronize()
  print("[iso A2] CLEAN", flush=True)
elif stage == "A3":  # weights-first + TWO launches (P7B launch pattern)
  P.up("fg0", np.load(f"{P7}/fg0.npy")); P.up("fu0", np.load(f"{P7}/fu0.npy"))
  P.up("x2", (rng.standard_normal((32, KD)) * 0.8).astype(np.float16).reshape(-1))
  P.poison("o2", 32 * ND, np.float16, 7.7)
  dev.synchronize(); print("[iso A3] weights-first order, 2 launches...", flush=True)
  launch(P.d["fg0"], P.d["fu0"], P.d["x2"], P.d["o2"])
  launch(P.d["fg0"], P.d["fu0"], P.d["x2"], P.d["o2"])
  dev.synchronize(); print("[iso A3] CLEAN", flush=True)
