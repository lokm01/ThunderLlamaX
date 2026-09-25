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
L = 100352; S = 41; SP = 90000
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
k1 = mk(f"{D}/k1rs_S{S}.cubin", "k1rs")
rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
bQ, bKV = up(Q), up(KV)
pt = Tensor.zeros(72*L).contiguous().realize(); _KEEP.append(pt)
wst = Tensor.zeros(S*4*18*258).contiguous().realize(); _KEEP.append(wst)
k1(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
Pn = pt.numpy().reshape(72, L)
wsn = wst.numpy().reshape(S*4*18, 258)
print("P nan:", np.isnan(Pn).sum(), "P -inf:", np.isneginf(Pn).sum(), "expect-inf", 24*((L-SP-1)+(L-SP-2)+(L-SP-3)))
print("ws nan:", np.isnan(wsn).sum(), "ws m=-inf:", (wsn[:,256]==float("-inf")).sum(), "ws l=0:", (wsn[:,257]==0).sum(), "of", S*4*18)
bad = np.isnan(wsn[:, :256]).any(axis=1).nonzero()[0]
print("ws bad rows:", bad[:10])
s, gr, row = (bad[0]//(4*18)), (bad[0]//18)%4, bad[0]%18 if len(bad) else (0,0,0)
print("first bad: s,gr,row =", s, gr, row)
