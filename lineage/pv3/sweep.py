# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, time
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
L = 100352; SP = 90000
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(cubin, name):
    return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bQ, bKV, bG = up(Q), up(KV), up(G)
pt = Tensor.zeros(72*L).contiguous().realize(); _KEEP.append(pt)
bP = pt.uop.buf_uop.buffer._bufs["NV"]
o1 = Tensor.full((18432,), -3e38).contiguous().realize(); _KEEP.append(o1)
ob1 = o1.uop.buf_uop.buffer._bufs["NV"]
def bench(p, args, grid, thr, vals, n=3, inner=32):
    for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=vals)
        p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3
for S in (20, 40, 64):
    k1 = mk(f"{D}/k1v2_S{S}.cubin" if S != 20 else f"{D}/k1v2.cubin", "k1v2")
    k2 = mk(f"{D}/k2v2_S{S}.cubin" if S != 20 else f"{D}/k2v2.cubin", "k2v2")
    wst = Tensor.zeros(S*12*6*8*258).contiguous().realize(); _KEEP.append(wst)
    bws = wst.uop.buf_uop.buffer._bufs["NV"]
    k1(bP, bQ, bKV, bws, global_size=(S*12,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
    k2(ob1, bP, up(np.zeros(72, np.float32)), up(np.ones(72, np.float32)), bKV, bG, bws, global_size=(72,1,1), local_size=(256,1,1), vals=(0,), wait=True)
    t1 = bench(k1, (bP, bQ, bKV, bws), (S*12,1,1), (256,1,1), (SP,))
    t2 = bench(k2, (ob1, bP, up(np.zeros(72, np.float32)), up(np.ones(72, np.float32)), bKV, bG, bws), (72,1,1), (256,1,1), (0,))
    print(f"S={S}: K1={t1:.3f} K2={t2:.3f} pair={t1+t2:.3f} proj16={(t1+t2)*16:.1f}ms", flush=True)
