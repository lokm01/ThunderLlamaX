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
PS = 100349
S = 24
SP = L - 500        # draft runs near the end; exercise the sp bound
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
stock = mk(f"{D}/draftpv_stock.cubin", "r_12_16_16_2_28mtp_sp2B129")
k1 = mk(f"{D}/k1t1.cubin", "k1t1")
k2 = mk(f"{D}/k2t1.cubin", "k2t1")
rng = np.random.default_rng(11)
Pn = (rng.random(24 * PS) * 0.001).astype(np.float32)   # normalized probs (small, sum~1/head)
KV = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
G = (rng.standard_normal(12288) * 0.5).astype(np.float32)
bP, bKV, bG = up(Pn), up(KV), up(G)
bws = up(np.zeros(S * 4 * 6 * 8 * 256, np.float32))
def outb():
    o = Tensor.full((6144,), -3e38).contiguous().realize(); _KEEP.append(o)
    return o, o.uop.buf_uop.buffer._bufs["NV"]
o0, ob0 = outb()
stock(ob0, bP, bKV, bG, global_size=(16,12,1), local_size=(16,1,1), vals=(SP,), wait=True)
ref = o0.numpy()
o1, ob1 = outb()
k1(bP, bKV, bws, global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
k2(ob1, bP, bKV, bG, bws, global_size=(24,1,1), local_size=(256,1,1), vals=(0,), wait=True)
got = o1.numpy()
rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
print(f"T1 relerr={rel:.2e} {'OK' if rel < 1e-4 else 'FAIL'}")
if rel >= 1e-4 and np.isfinite(got).all():
    bad = np.abs(got - ref).argmax()
    print("worst idx", bad, "ref", ref[bad], "got", got[bad])
def bench(p, args, grid, thr, vals, n=3, inner=32):
    for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=vals)
        p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3
t_s = bench(stock, (ob0, bP, bKV, bG), (16,12,1), (16,1,1), (SP,))
t_1 = bench(k1, (bP, bKV, bws), (S*4,1,1), (256,1,1), (SP,))
t_2 = bench(k2, (ob1, bP, bKV, bG, bws), (24,1,1), (256,1,1), (0,))
print(f"stock={t_s:.3f}ms  K1={t_1:.3f} K2={t_2:.3f} pair={t_1+t_2:.3f}  x2-draft-steps save={(t_s*2-(t_1+t_2)*2):.1f}ms/cycle")
