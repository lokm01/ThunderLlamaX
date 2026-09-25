# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np, time
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
BASE = "~/tinygrad-metal/engine0"
CTXK, S, RM, LS32, POS = 100352, 256, 3, (1024,1,1), 97810
P = Bufs()
rng = np.random.default_rng(7)
kv16 = (rng.standard_normal((2,4,CTXK,256)).astype(np.float32)*0.5).astype(np.float16)
g = kv16.reshape(2,4,CTXK,8,32)
amax = np.abs(g).max(axis=-1)
sc = (np.maximum(amax,1e-8)*(1.0/127.0)).astype(np.float16)
q = (np.clip(np.rint(g.astype(np.float32)/sc.astype(np.float32)[...,None]),-127,127)+128).astype(np.uint8)
P.up("kv16", kv16); P.up("kv8", q.reshape(-1)); P.up("sc", sc.reshape(-1))
P.up("qw", rng.standard_normal((RM*24,256)).astype(np.float32)*0.05)
P.up("pos_slot", np.array([POS],dtype=np.int32))
P.poison("pm", 4*S*6*RM*4, np.float32, 7.7e31); P.poison("ps", 4*S*6*RM*4, np.float32, 7.7e31); P.poison("pA", 4*S*6*RM*256*4, np.float32, 7.7e31)
dev.synchronize()
del kv16, q
def bench(nm, is8, reps=10):
  lib = open(f"{BASE}/{nm}.cubin","rb").read()
  pr = NVProgram(dev, TinyELF(lib=lib, name=nm, target=dev.renderer.target, signature=tuple()))
  a8 = (P.d["kv8"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"])
  a16 = (P.d["kv16"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"])
  for r in range(2): pr(*(a8 if is8 else a16), global_size=(4*S,1,1), local_size=LS32); dev.synchronize()
  t0=time.perf_counter()
  for r in range(reps):
    pr(*(a8 if is8 else a16), global_size=(4*S,1,1), local_size=LS32); dev.synchronize()
  return (time.perf_counter()-t0)/reps
for nm, is8 in (("spk_g4nw32a3_100k",False),("spk_g4nw32qa3_100k",True),("d1g4q_100k",True),("d2g4q_100k",True)):
  print(f"[bench] {nm}: {bench(nm,is8)*1e3:.3f} ms", flush=True)
