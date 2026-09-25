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
bP = up(np.zeros(24 * 100349, np.float32)); bKV = up(KVn)
wst = Tensor.zeros(24*4*6*256).contiguous().realize(); _KEEP.append(wst)
mk(f"{D}/k1dbg.cubin", "k1dbg")(bP, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
ws = wst.numpy().reshape(24, 4, 6, 256)
KV2 = KVn.reshape(2, 4, L, 256).astype(np.float32)
for pi in range(6):
    img = ws[0, 0, pi, :]
    hits = []
    for side in range(2):
        for gg in range(4):
            for kk in range(0, 16):
                if np.abs(img - KV2[side, gg, kk]).max() < 1e-5:
                    hits.append((side, gg, kk))
    print("row", pi, "matches:", hits)
