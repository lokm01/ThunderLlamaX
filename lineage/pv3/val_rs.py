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
def mk(cubin, name):
    return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
qk    = mk(f"{D}/qk100352.cubin", "r_3_25088_4_4_3_4_2_16_4")
stock = mk(f"{D}/stock100352.cubin", "r_3_24_16_16_12544_3136_12544_3136_12544_12544_12544_12544_4_4")
k1    = mk(f"{D}/k1rs_S{S}.cubin", "k1rs")
k2    = mk(f"{D}/k2rs_S{S}.cubin", "k2rs")
rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bQ, bKV, bG = up(Q), up(KV), up(G)
pt = Tensor.zeros(72*L).contiguous().realize(); _KEEP.append(pt)
bws = up(np.zeros(S*4*18*258, np.float32))
def outb():
    o = Tensor.full((18432,), -3e38).contiguous().realize(); _KEEP.append(o)
    return o, o.uop.buf_uop.buffer._bufs["NV"]
qk(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, global_size=(25088,3,1), local_size=(4,4,3), vals=(SP,), wait=True)
Pn = pt.numpy().reshape(72, L)
mx = Pn.max(axis=1).astype(np.float32)
sm = np.exp2((Pn - mx[:,None]) * np.float32(1.4426950216293335)).sum(axis=1).astype(np.float32)
bmx, bsm = up(mx), up(sm)
o0, ob0 = outb()
stock(ob0, pt.uop.buf_uop.buffer._bufs["NV"], bmx, bsm, bKV, bG, global_size=(16,24,3), local_size=(16,1,1), vals=(0,), wait=True)
ref = o0.numpy()
o1, ob1 = outb()
k1(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, bws, global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
k2(ob1, pt.uop.buf_uop.buffer._bufs["NV"], bmx, bsm, bKV, bG, bws, global_size=(72,1,1), local_size=(256,1,1), vals=(0,), wait=True)
got = o1.numpy()
print("ref nan:", np.isnan(ref).sum(), "got nan:", np.isnan(got).sum(), flush=True)
if np.isnan(got).any():
    bi = np.isnan(got).nonzero()[0][:5]
    print("got nan idx:", bi, "ref there:", ref[bi])
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
print(f"RS S={S}: relerr={rel:.2e} {'OK' if rel < 1e-4 else 'FAIL'}  K1={t1:.3f}ms K2={t2:.3f}ms pair16={(t1+t2)*16:.1f}ms (vs pf 3.28/52.6)", flush=True)
