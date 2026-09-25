# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Oracle: P=0 (all), V=1, gates=+10 -> out must be exactly sigmoid(10) everywhere."""
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
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]
bP = up(np.zeros(196608, np.float32))
bm = up(np.zeros(24, np.float32))
bs = up(np.full(24, 8192.0, np.float32))
bV = up(np.ones(16777216, np.float16))
bG = up(np.full(36864, 10.0, np.float32))
for nm, cubin, grid, thr in (
    ("stock", "~/tinygrad-metal/splitkv/pv_stock.cubin", (16,24,16), (16,1,1)),
    ("new",   "~/tinygrad-metal/splitkv/splitkv_pv.cubin", (3,24,1), (256,1,1))):
    prog = NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name="x", target=dev.renderer.target, signature=tuple()))
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    prog(ob, bP, bm, bs, bV, bP, bm, bs, bP, bm, bs, bG, global_size=grid, local_size=thr, wait=True)
    o = outT.numpy()
    print(f"[{nm}] min={o.min():.6f} max={o.max():.6f} (expect ~{1/(1+np.exp(-10)):.6f})", flush=True)
