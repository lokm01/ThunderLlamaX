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
L = 100352
S = int(sys.argv[1]) if len(sys.argv) > 1 else 41
SP = 90000
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
k1 = mk(f"{D}/k1rs_S{S}.cubin", "k1rs")
k2 = mk(f"{D}/k2rs_S{S}.cubin", "k2rs")
rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bQ, bKV, bG = up(Q), up(KV), up(G)
pt = Tensor.zeros(72*L).contiguous().realize(); _KEEP.append(pt)
bws = up(np.zeros(S*4*18*258, np.float32))
bmx = up(np.zeros(72, np.float32)); bsm = up(np.ones(72, np.float32))
k1(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, bws, global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
o1 = Tensor.full((18432,), -3e38).contiguous().realize(); _KEEP.append(o1)
ob1 = o1.uop.buf_uop.buffer._bufs["NV"]
k2(ob1, pt.uop.buf_uop.buffer._bufs["NV"], bmx, bsm, bKV, bG, bws, global_size=(72,1,1), local_size=(256,1,1), vals=(0,), wait=True)
got = o1.numpy()
ws = up.__self__ if False else None
wst = [t for t in _KEEP if t.shape == (S*4*18*258,)][0]
wsn = wst.numpy().reshape(S, 72, 258)
ref = np.zeros(18432, np.float64)
for h in range(24):
    g, h6 = h//6, h%6
    for r in range(3):
        sl = g*18 + h6*3 + r
        M = wsn[:, sl, 256].max()
        wt = 2.0**(wsn[:, sl, 256] - M)
        Lr = (wt * wsn[:, sl, 257]).sum()
        O = (wt[:, None] * wsn[:, sl, :256]).sum(axis=0)
        gt = G[256 + h*512 + r*12288: 256 + h*512 + r*12288 + 256]
        ref[r*6144 + h*256: r*6144 + h*256 + 256] = (O/Lr) * (1/(1+2.0**(-gt*1.4426950216293334)))
rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
def bench(p, args, grid, thr, vals, n=3, inner=32):
    for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=vals)
        p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3
t1 = bench(k1, (pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, bws), (S*4,1,1), (256,1,1), (SP,))
t2 = bench(k2, (ob1, pt.uop.buf_uop.buffer._bufs["NV"], bmx, bsm, bKV, bG, bws), (72,1,1), (256,1,1), (0,))
print(f"RS2 S={S}: relerr={rel:.2e} {'OK' if rel < 1e-4 else 'FAIL'}  K1={t1:.3f}ms K2={t2:.3f}ms pair16={(t1+t2)*16:.1f}ms", flush=True)
