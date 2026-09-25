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
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]
bufs = [up(np.full(8, float(i+1), np.float32)) for i in range(12)]
outT = Tensor.zeros(18432).contiguous().realize()
ob = outT.uop.buf_uop.buffer._bufs["NV"]
p = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv_min12.cubin","rb").read(),
      name="pv_min12", target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
import sys as _s
grid = tuple(int(x) for x in _s.argv[1:4]) if len(_s.argv) > 3 else (3,24,2)
p(ob, *bufs[1:], global_size=grid, local_size=(128,1,1), vals=(42,), wait=True)
o = outT.numpy()
print(f"[min12 grid={grid}] o[:3]={o[:3]}", flush=True)
