# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P17 iso: is the pfcw32 combine launchable at all? Warmed pf10-shape world,
one control pair, then pfcw32 on garbage partials (pmW-class, in-bounds),
then pfc16t control again."""
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
P.poison("pos_slot", 4, np.int32, -1)
P.poison("pm", 4*32*96*4, np.float32, 7.7e31)
P.poison("ps", 4*32*96*4, np.float32, 7.7e31)
P.poison("pA", 4*32*96*256*4, np.float32, 7.7e31)
P.poison("pmW", 39936*4, np.float32, 7.7e31)
P.poison("psW", 39936*4, np.float32, 7.7e31)
P.poison("pAW", 39936*256*4, np.float32, 7.7e31)
P.poison("aoW", 64*6144*2, np.float16, np.float16(7.7))
P.up("qrowW", (rng.standard_normal(32*12288) * 0.3).astype(np.float16))
P._keep.clear(); dev.synchronize()
print("[iso] world up", flush=True)

pka = prog("pfa32c_t32_s13_100k", "pfa32ct")
pkc = prog("pfc16t_s13", "pfc16t")
pcw = prog("pfcw32_s13", "pfcw32")

# control pair
P.win_up("pos_slot", 0, np.array([100336], dtype=np.int32)); dev.synchronize()
pka(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
    global_size=(156,1,1), local_size=(512,1,1))
dev.synchronize(); print("[iso] ctrl attn ok", flush=True)
pkc(P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow16"], P.d["ao16"], global_size=(24,1,1), local_size=(256,1,1))
dev.synchronize(); print("[iso] ctrl combine ok", flush=True)

# pfcw32 on garbage partials (in-bounds reads of pmW-class; math garbage is fine)
pcw(P.d["pmW"], P.d["psW"], P.d["pAW"], P.d["qrowW"], P.d["aoW"],
    global_size=(24,1,1), local_size=(256,1,1))
dev.synchronize(); print("[iso] pfcw32 RUNS (garbage math ok)", flush=True)
pkc(P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow16"], P.d["ao16"], global_size=(24,1,1), local_size=(256,1,1))
dev.synchronize(); print("[iso] ctrl combine again ok", flush=True)
o = P.down("aoW", (32, 6144), np.float16)
print(f"[iso] down ok nan={np.isnan(o.astype(np.float32)).sum()}", flush=True)
