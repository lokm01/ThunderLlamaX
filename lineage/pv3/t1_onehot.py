# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
dev = Device["NV"]
D = "~/tinygrad-metal/pv3"
L = 100352; PS = 100349; S = 24; SP = L - 500
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(3)
P = np.zeros(24 * PS, np.float32)
KVn = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
G = (rng.standard_normal(12288) * 0.5).astype(np.float32)
# one-hot: head 7 (=g1,h6=1), position K1POS
K1POS = 5023
P[7 * PS + K1POS] = 1.0
bP, bKV, bG = up(P), up(KVn), up(G)
wst = Tensor.zeros(S*4*6*256).contiguous().realize(); _KEEP.append(wst)
k1 = mk(f"{D}/k1t1.cubin", "k1t1")
k1(bP, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
ws = wst.numpy().reshape(S, 4, 6, 256)
# which slot is nonzero?
nz = np.abs(ws).sum(axis=(1, 2, 3))
s_expect = K1POS // ((L + S - 1)//S)
print("nonzero splits:", np.nonzero(nz > 1e-6)[0], "expected split:", s_expect)
nzs = int(np.argmax(nz))
w = ws[nzs]
gh = np.unravel_index(np.abs(w).argmax(), (4, 6, 256))[:2]
print("nonzero (g,h6):", gh, "expected (1,1)")
row = w[gh[0], gh[1], :]
V = KVn.reshape(2, 4, L, 256)[1].astype(np.float32)
# find which position's V row this equals (head 7 -> group 1)
target_g = 7 // 6
best, bestk = 1e9, -1
for k in range(max(0, K1POS-60000), min(SP, K1POS+60000), 1):
    d = np.abs(row - V[target_g, k]).max()
    if d < best: best, bestk = d, k
print(f"closest V row: k={bestk} dist={best:.4f} (one-hot was at K={K1POS}, dist-to-correct={np.abs(row - V[target_g, K1POS]).max():.4f})")
