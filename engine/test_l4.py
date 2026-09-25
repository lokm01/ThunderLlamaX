# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2F L4 standalone: k2s3v(+k2z3) vs k2s3 bit-identity + bench (random fixtures)."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)
P = Bufs()
pr = {}
for n in ("k2s3", "k2s3v", "k2z3"):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
rng = np.random.default_rng(21)
CBLK, RBLK, NVH = 3*10240, 48*128*128, 48
P.up("conv", (rng.standard_normal(5*CBLK)*0.1).astype(np.float32))
P.up("rec", (rng.standard_normal(5*RBLK)*0.05).astype(np.float32))
P.up("qkv3", (rng.standard_normal(3*10240)*0.3).astype(np.float16))
P.up("gate3", (rng.standard_normal(3*6144)*0.3).astype(np.float16))
P.up("convw", (rng.standard_normal(10240*4)*0.05).astype(np.float32))
P.up("dtb", (rng.standard_normal(48)*0.1).astype(np.float32))
P.up("ssma", (rng.standard_normal(48)*0.02).astype(np.float32))
P.up("a3", (rng.standard_normal(3*48)*0.3).astype(np.float32))
P.up("b3", (rng.standard_normal(3*48)*0.3).astype(np.float32))
P.up("snw", (1.0 + 0.05*rng.standard_normal(128)).astype(np.float32))
for sfx in "AB":
  P.poison(f"q_{sfx}", 48*128*4, np.float32, 7.7e31)
  P.poison(f"k_{sfx}", 48*128*4, np.float32, 7.7e31)
  P.poison(f"v_{sfx}", 48*128*4, np.float32, 7.7e31)
  P.poison(f"core_{sfx}", 48*128*4, np.float32, 7.7e31)
  P.poison(f"z3_{sfx}", 3*6144*2, np.float16, 7.7)
  P.poison(f"convw_{sfx}", 5*CBLK*4, np.float32, 7.7e31)
  P.poison(f"recw_{sfx}", 5*RBLK*4, np.float32, 7.7e31)
dev.synchronize()
d = P.d
def outs(sfx):
  return [P.down(f"{nm}_{sfx}", sh, dt) for nm, sh, dt in
          (("q",(48*128,),np.float32),("k",(48*128,),np.float32),("v",(48*128,),np.float32),
           ("core",(48*128,),np.float32),("z3",(3*6144,),np.float16),("convw",(5*CBLK,),np.float32),("recw",(5*RBLK,),np.float32))]
ARGS = lambda sfx: (d[f"convw_{sfx}"], d[f"recw_{sfx}"], d["qkv3"], d["gate3"], d["convw"], d["dtb"], d["ssma"],
                    d["a3"], d["b3"], d[f"q_{sfx}"], d[f"k_{sfx}"], d[f"v_{sfx}"], d[f"core_{sfx}"], d["snw"], d[f"z3_{sfx}"])
pr["k2s3"](*ARGS("A"), global_size=(48,1,1), local_size=LS); dev.synchronize()
pr["k2s3v"](*ARGS("B"), global_size=(384,1,1), local_size=LS)
pr["k2z3"](d["core_B"], d["gate3"], d["snw"], d["z3_B"], global_size=(48,1,1), local_size=(32,1,1))
dev.synchronize()
for nm, a, b in zip(("q","k","v","core","z3","conv","rec"), outs("A"), outs("B")):
  print(f"[l4] {nm} bit-identical: {bool((a==b).all())} (nonzero {int((a!=0).sum())}/{a.size})", flush=True)
def bench():
  t0 = time.perf_counter()
  for r in range(30): pr["k2s3"](*ARGS("A"), global_size=(48,1,1), local_size=LS)
  dev.synchronize(); dt0 = (time.perf_counter()-t0)/30
  t0 = time.perf_counter()
  for r in range(30):
    pr["k2s3v"](*ARGS("B"), global_size=(384,1,1), local_size=LS)
    pr["k2z3"](d["core_B"], d["gate3"], d["snw"], d["z3_B"], global_size=(48,1,1), local_size=(32,1,1))
  dev.synchronize(); dt1 = (time.perf_counter()-t0)/30
  return dt0, dt1
dt0, dt1 = bench()
print(f"[l4] k2s3 {dt0*1e3:.3f} ms -> k2s3v+k2z3 {dt1*1e3:.3f} ms  delta {(dt0-dt1)*1e3:+.3f} (x48 blocks/probe)", flush=True)
print("[l4 done]", flush=True)
