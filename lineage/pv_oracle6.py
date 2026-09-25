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
# DISTINCT buffers for every arg (no aliasing with __restrict__)
bP0 = up(np.zeros(196608, np.float32))
bP1 = up(np.zeros(196608, np.float32))
bP2 = up(np.zeros(196608, np.float32))
bm0, bm1, bm2 = up(np.zeros(24, np.float32)), up(np.zeros(24, np.float32)), up(np.zeros(24, np.float32))
bs0, bs1, bs2 = (up(np.full(24, 8192.0, np.float32)) for _ in range(3))
bV = up(np.ones(16777216, np.float16))
bG = up(np.full(36864, 10.0, np.float32))
outT = Tensor.zeros(18432).contiguous().realize()
ob = outT.uop.buf_uop.buffer._bufs["NV"]
prog = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pvc3.cubin","rb").read(),
      name="pv_coalesced", target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
# arg order: out, P2, m2, s2, V, P1, m1, s1, P0, m0, s0, gates
prog(ob, bP2, bm2, bs2, bV, bP1, bm1, bs1, bP0, bm0, bs0, bG,
     global_size=(3,24,2), local_size=(128,1,1), vals=(0,), wait=True)
o = outT.numpy()
print(f"[pv-distinct] min={o.min():.6f} max={o.max():.6f} expect~{1/(1+np.exp(-10)):.6f}", flush=True)
