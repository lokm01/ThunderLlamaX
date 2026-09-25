# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
P = Bufs()
eh = NVProgram(dev, TinyELF(lib=open("ehproj.cubin","rb").read(), name="ehproj", target=dev.renderer.target, signature=tuple()))
P.up("w_eh", np.load("draft_pack/d_eh.npy").reshape(-1))
P.up("zed", np.zeros(5120, dtype=np.float32))
P.poison("o", 5120*4, np.float32, 7.7e31)
dev.synchronize(); d = P.d
def run(cat_np):
  P.up("x", cat_np.astype(np.float16)); dev.synchronize()
  eh(d["w_eh"], d["x"], d["zed"], d["o"], global_size=(640,1,1), local_size=(256,1,1), wait=True)
  return P.down("o", (5120,))
def deq():
  w = np.load("draft_pack/d_eh.npy"); nout, rowb = w.shape; ngrp = rowb // 144
  qs = w[:, :ngrp*128].reshape(nout, ngrp*8, 16)
  dd = np.frombuffer(w[:, ngrp*128:].tobytes(), dtype="<f2").reshape(nout, ngrp*8).astype(np.float32)
  out = np.zeros((nout, ngrp*256), np.float32)
  for s in range(ngrp*8):
    b = qs[:, s, :]
    nib = np.zeros((nout, 32), np.float32)
    for j in range(16):
      nib[:, 2*j] = (b[:, j] & 0xF) - 8
      nib[:, 2*j+1] = (b[:, j] >> 4) - 8
    out[:, s*32:(s+1)*32] = nib * dd[:, s:s+1]
  return out
W = deq()


W = deq()
row0 = W[0].copy()
weff = np.zeros(10240, np.float32)
x = np.zeros(10240, np.float32)
for E in range(10240):
    x[:] = 0; x[E] = 1.0
    g = run(x)
    weff[E] = g[0]
np.save("/tmp/weff.npy", weff)
print("[bf] weff[:12] ", np.round(weff[:12], 4).tolist())
print("[bf] W[0][:12] ", np.round(row0[:12], 4).tolist())
# find permutation: for each E, which column C has W[0][C] == weff[E]
match = {}
for E in range(0, 64):
    cs = np.where(np.abs(row0 - weff[E]) < 1e-4)[0]
    match[E] = cs[:4].tolist()
print("[bf] E->matching cols (first 32):")
for E in range(0, 32, 8):
    print("[bf] ", E, "->", match[E], " .. ", E+7, "->", match[E+7])
