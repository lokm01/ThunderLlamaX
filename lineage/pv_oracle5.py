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
bP = up(np.zeros(196608, np.float32))
bm = up(np.zeros(24, np.float32))
bs = up(np.full(24, 8192.0, np.float32))
bV = up(np.ones(16777216, np.float16))
bG = up(np.full(36864, 10.0, np.float32))
outT = Tensor.zeros(18432).contiguous().realize()
ob = outT.uop.buf_uop.buffer._bufs["NV"]
# CONTROL: reuse the PROVEN K1 cubin+signature style — actually simplest control:
# load pv cubin and print its ELF symbols the same way ops_nv does
lib = open("~/tinygrad-metal/splitkv/splitkv_pv.cubin","rb").read()
print("cubin len:", len(lib), flush=True)
import subprocess
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin")+":/opt/homebrew/bin:/usr/bin:/bin")
r = subprocess.run(["bash","-c","strings ~/tinygrad-metal/splitkv/splitkv_pv.cubin | grep -i pv | head -5"],capture_output=True,text=True,env=env)
print("symbols:", r.stdout, flush=True)
