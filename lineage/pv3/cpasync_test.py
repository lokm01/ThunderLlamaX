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
_KEEP = []
t = Tensor(np.arange(16, dtype=np.float32)).contiguous().realize(); _KEEP.append(t)
o = Tensor.full((16,), -3e38).contiguous().realize(); _KEEP.append(o)
p = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/pv3/cpasync_min.cubin","rb").read(), name="cpasync_min", target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
p(o.uop.buf_uop.buffer._bufs["NV"], t.uop.buf_uop.buffer._bufs["NV"], global_size=(1,1,1), local_size=(32,1,1), vals=(0,), wait=True)
r = o.numpy()
ok = np.allclose(r, np.arange(16) + 1.0)
print("cpasync result:", r[:4], "... OK" if ok else "FAIL")
print("VERDICT: cp.async WORKS on dext" if ok else "VERDICT: cp.async broken/faulted")
