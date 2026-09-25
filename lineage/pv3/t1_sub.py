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
P = np.zeros(24 * PS, np.float32)
P[0 * PS + 3] = 1.0
P[5] = 0.5
bP, bKV = up(P), up(KVn)
wst = Tensor.zeros(S*4*6*256).contiguous().realize(); _KEEP.append(wst)
sub = mk(f"{D}/k1sub.cubin", "k1sub")
sub(bP, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
ws = wst.numpy()
print("ws[0:8] =", ws[:8])
print("expect  [0,0,0,1,0,0.5,0,0]:", "PASS" if abs(ws[3]-1.0)<1e-6 and abs(ws[5]-0.5)<1e-6 else "FAIL")
