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
outT = Tensor.zeros(256).contiguous().realize()
ob = outT.uop.buf_uop.buffer._bufs["NV"]
p = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv_min12.cubin","rb").read(),
      name="pv_min12", target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
p(ob, *bufs[1:], global_size=(1,1,2), local_size=(128,1,1), vals=(42,), wait=True)
print("[pv_min12]", outT.numpy()[:3], "(expect 44=42+1+1)", flush=True)
