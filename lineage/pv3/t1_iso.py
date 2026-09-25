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
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
k2 = mk(f"{D}/k2t1.cubin", "k2t1")
rng = np.random.default_rng(5)
bP = up((rng.random(24*100349)*0.001).astype(np.float32))
bKV = up((rng.standard_normal(100352*2048)*0.3).astype(np.float16))
bG = up((rng.standard_normal(12288)*0.5).astype(np.float32))
bws = up((rng.random(24*4*6*256)*0.01).astype(np.float32))   # host-filled, NO k1
o = Tensor.full((6144,), -3e38).contiguous().realize(); _KEEP.append(o)
k2(o.uop.buf_uop.buffer._bufs["NV"], bP, bKV, bG, bws, global_size=(24,1,1), local_size=(256,1,1), vals=(0,), wait=True)
r = o.numpy()
print("k2t1 ALONE:", "CLEAN" if np.isfinite(r).all() and (r > -1e37).any() else "BAD", r[:3])
