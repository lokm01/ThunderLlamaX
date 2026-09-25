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
P = np.zeros(24 * PS, np.float32)
P[0 * PS + 3] = 1.0
bP = up(P)
wst = Tensor.zeros(S*4*6*256).contiguous().realize(); _KEEP.append(wst)
k1(bP, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
ws = wst.numpy().reshape(S, 4, 6, 256)
row = ws[0, 0, 0, :]
nzdims = np.nonzero(np.abs(row) > 1e-9)[0]
print("nz dims count:", len(nzdims), "min:", nzdims.min() if len(nzdims) else None, "max:", nzdims.max() if len(nzdims) else None)
print("contiguous [0,192)?", len(nzdims) == 192 and nzdims.min() == 0 and nzdims.max() == 191)
for span in (192, 256):
    seg = row[:span]; exp = V[0, 3][:span]
    print(f"dims [0,{span}): relerr vs V[0,3] = {np.abs(seg - exp).max() / max(np.abs(exp).max(), 1e-9):.4f}")
# also: which OTHER ws slots are nonzero?
nzs = np.abs(ws).sum(axis=3)
print("nonzero (s,g,h6):", list(zip(*np.nonzero(nzs > 1e-6))))
