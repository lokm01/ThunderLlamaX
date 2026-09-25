# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Bare-world p8q8x probe (the p8_imma_time pattern — no engine, Bufs only).
Bisects the in-world launch fault: if this faults too, it's the kernel/launch;
arms: [a] full kernel, and if it faults, edit PF8Q8_ARM below for reduced
variants. Back-to-back + per-launch synced."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev, Bufs
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

P = Bufs()
NM = "p8q8x_nw8k128"
lib = open(f"~/tinygrad-metal/engine0/{NM}.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name="p8q8x", target=dev.renderer.target, signature=tuple()))
M, KD, NCH = 128, 5120, 40
rng = np.random.default_rng(3)
xf = (rng.standard_normal((M, KD)) * 0.7).astype(np.float16)
xq = np.zeros((M, KD), dtype=np.int8); sx = np.zeros((M, NCH), dtype=np.float32)
rs = np.zeros((M, NCH), dtype=np.int32)
for c in range(NCH):
  seg = xf[:, c*128:(c+1)*128].astype(np.float32)
  am = np.maximum(np.abs(seg).max(axis=1), 1e-6)
  s = am * (1.0 / 127.0)
  sx[:, c] = s
  q = np.clip(np.rint(seg / s[:, None]), -127, 127).astype(np.int8)
  xq[:, c*128:(c+1)*128] = q
  rs[:, c] = q.sum(axis=1).astype(np.int32)
P.up("t_hhx", xf)
P.poison("t_xqk", M*KD, np.int8, -19)
P.poison("t_sxk", M*NCH*4, np.float32, 7.7e31)
P.poison("t_rsk", M*NCH*4, np.int32, 0x5a5a5a5a)
dev.synchronize(); P._keep.clear()
d = P.d
def L(wait=True):
  pr(d["t_hhx"], d["t_xqk"], d["t_sxk"], d["t_rsk"], d["t_hhx"], d["t_hhx"],
     global_size=(M, 1, 1), local_size=(256, 1, 1), wait=wait)
print("[probe] first launch (synced)...", flush=True)
L(); dev.synchronize()
print("[probe] first launch OK", flush=True)
gxq = P.down("t_xqk", (M, KD), np.int8)
gsx = P.down("t_sxk", (M, NCH), np.float32)
grs = P.down("t_rsk", (M, NCH), np.int32)
print(f"[probe] xq identical: {bool((gxq == xq).all())}", flush=True)
print(f"[probe] sx  identical: {bool((gsx == sx).all())}", flush=True)
print(f"[probe] rs  identical: {bool((grs == rs).all())}", flush=True)
ts = []
for _ in range(8):
  t0 = time.perf_counter(); L(); dev.synchronize(); ts.append(time.perf_counter() - t0)
print(f"[probe] per-launch min {min(ts)*1e6:.1f} us", flush=True)
print("[probe] DONE", flush=True)
