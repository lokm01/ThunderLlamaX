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
L = 100352; S = 20; SP = 90000
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(cubin, name):
    return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
k1 = mk(f"{D}/k1v2.cubin", "k1v2"); k2 = mk(f"{D}/k2v2.cubin", "k2v2")
rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bQ, bKV, bG = up(Q), up(KV), up(G)
bP = up(np.zeros(72*L, np.float32))
bws_t = Tensor.zeros(S*12*6*258).contiguous().realize(); bws = bws_t.uop.buf_uop.buffer._bufs["NV"]
o1 = Tensor.full((18432,), -3e38).contiguous().realize(); ob1 = o1.uop.buf_uop.buffer._bufs["NV"]
k1(bP, bQ, bKV, bws, global_size=(S*12,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
k2(ob1, bP, up(np.zeros(72, np.float32)), up(np.ones(72, np.float32)), bKV, bG, bws, global_size=(72,1,1), local_size=(256,1,1), vals=(0,), wait=True)
got = o1.numpy()
pt = Tensor.zeros(72*L).contiguous().realize()
wsn = bws_t.numpy().reshape(S*12*6, 258)
print("ws m2 -inf count:", (wsn[:,256] == float("-inf")).sum(), "of", S*12*6)
print("ws l==0 count:", (wsn[:,257] == 0).sum())
print("ws o nan:", np.isnan(wsn[:,:256]).sum())
print("got nan:", np.isnan(got).sum(), "inf:", np.isinf(got).sum())
P_t = Tensor.zeros(72*L).contiguous().realize()
k1(P_t.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, bws, global_size=(S*12,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
Pn = P_t.numpy().reshape(72, L)
print("P nan:", np.isnan(Pn).sum(), "P -inf:", np.isneginf(Pn).sum(), "expect", 72*(L-SP-1))
print("P row0 valid-range max:", Pn[0,:SP+1].max(), "min:", Pn[0,:SP+1].min())
