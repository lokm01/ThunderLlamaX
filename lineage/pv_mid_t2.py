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
sizes = {1:196608, 5:196608, 8:196608, 4:16777216, 11:36864}
bufs = [up(np.ones(sizes.get(i, 24), np.float16 if i==4 else np.float32)) for i in range(1,12)]
outT = Tensor.zeros(18432).contiguous().realize()
ob = outT.uop.buf_uop.buffer._bufs["NV"]
p = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv_mid.cubin","rb").read(),
      name="pv_mid", target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
p(ob, *bufs, global_size=(3,24,2), local_size=(128,1,1), vals=(0,), wait=True)
o = outT.numpy()
print(f"[pv_mid] o[:2]={o[:2]} max={o.max():.4f} (expect ~64)", flush=True)
