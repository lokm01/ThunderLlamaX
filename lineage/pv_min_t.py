# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
dev = Device["NV"]
outT = Tensor.zeros(256).contiguous().realize()
ob = outT.uop.buf_uop.buffer._bufs["NV"]
p = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv_min.cubin","rb").read(),
      name="pv_min", target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
p(ob, global_size=(1,1,2), local_size=(128,1,1), vals=(42,), wait=True)
print("[pv_min]", outT.numpy()[:3], "(expect 42)", flush=True)
