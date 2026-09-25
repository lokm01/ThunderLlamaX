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
L = 100352; SP = L - 500
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(3)
KVn = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
P = np.zeros(24 * 100349, np.float32)
bP, bKV = up(P), up(KVn)
wst = Tensor.zeros(24*4*6*256).contiguous().realize(); _KEEP.append(wst)
dbg = mk(f"{D}/k1dbg.cubin", "k1dbg")
dbg(bP, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
ws = wst.numpy().reshape(24, 4, 6, 256)
V = KVn.reshape(2, 4, L, 256)[1].astype(np.float32)
for g in range(4):
    for pi in range(8):
        img = ws[0, g if pi < 6 else min(g+1,3), pi if pi < 6 else pi-6, :] if pi < 6 else ws[0, min(g+1,3), pi-6, :]
        # NOTE: the g+1 aliasing for pi>=6 only valid for g<3; skip confusing cases
        if pi >= 6 and g == 3: continue
        exp_row = V[g, pi]
        ok = np.abs(img - exp_row).max() < 1e-5
        if not ok:
            # find what it actually is: compare against a few candidate rows
            cands = {}
            for k in (pi, (pi+1)%8, 7-pi, 0):
                cands[f"V[{k}]"] = np.abs(img - V[g, k]).max()
            print(f"g={g} row={pi}: MISMATCH exp-max={np.abs(img-exp_row).max():.4f} candidates={cands} nz={int((np.abs(img)>0).sum())}")
        else:
            print(f"g={g} row={pi}: EXACT match V[{g},{pi}]")

# K-side check: are the image rows actually K rows?
K = KVn.reshape(2, 4, L, 256)[0].astype(np.float32)
for g in (0,):
    for pi in range(6):
        img = ws[0, g, pi, :]
        for side in (0, 1):
            Vx = KVn.reshape(2, 4, L, 256)[side].astype(np.float32)
            for gg in range(4):
                d = np.abs(img - Vx[gg, pi]).max()
                if d < 1e-5:
                    print(f"g={g} row={pi}: image == side{side}({K if side==0 else V})[{gg},{pi}]")
