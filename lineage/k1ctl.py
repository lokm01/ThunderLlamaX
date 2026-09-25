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
# CONTROL: launch K1 (the proven kernel) with signature=tuple() and NO vals
# signature currently in splitkv_test is 4 ints. Try empty.
Hkv, Rep, T, KD, VD = 8, 2, 3, 128, 128
L, POS, S = 512, 100, 1
rng = np.random.default_rng(3)
q = rng.normal(0,.3,(48,KD)).astype(np.float16)
kc = rng.normal(0,.3,(L,Hkv,KD)).astype(np.float16)
vc = rng.normal(0,.3,(L,Hkv,VD)).astype(np.float16)
slots = S*4
oacc = Tensor.zeros(slots,16,T,VD).contiguous().realize()
lse = Tensor.full((slots,16,T), float("-inf")).contiguous().realize()
q_b, k_b, v_b = up(q), up(kc), up(vc)
ob = oacc.uop.buf_uop.buffer._bufs["NV"]
lb = lse.uop.buf_uop.buffer._bufs["NV"]
n_chunk = POS + T
k1 = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/splitkv_k1.cubin","rb").read(),
      name="splitkv_k1", target=dev.renderer.target, signature=tuple()))
k1(q_b,k_b,v_b,ob,lb, global_size=(Hkv*S,1,1), local_size=(128,1,1), wait=True)
print("[k1-empty-sig] lse[0,0,0] =", lse.numpy()[0,0,0], "(nonzero == runs)", flush=True)
