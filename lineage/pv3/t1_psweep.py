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
KVn = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
bKV = up(KVn)
k1 = mk(f"{D}/k1t1.cubin", "k1t1")
V = KVn.reshape(2, 4, L, 256)[1].astype(np.float32)
for K1P in (0, 3, 7):
    P = np.zeros(24 * PS, np.float32)
    P[0 * PS + K1P] = 1.0     # head 0
    bP = up(P)
    wst = Tensor.zeros(S*4*6*256).contiguous().realize(); _KEEP.append(wst)
    k1(bP, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
    ws = wst.numpy().reshape(S, 4, 6, 256)
    row = ws[0, 0, 0, :]     # split 0, group 0, head 0
    nz = int((np.abs(row) > 1e-9).sum())
    # which V position does this row equal?
    hits = [k for k in range(0, 40) if np.abs(row - V[0, k]).max() < 1e-5]
    # also check other groups/splits just in case
    allhits = []
    tot = np.abs(ws).sum()
    if tot < 1e-6:
        print(f"one-hot p={K1P}: ws EMPTY")
        continue
    print(f"one-hot p={K1P}: row nz={nz}/256, equals V[0,{hits}] (want [{K1P}])")
