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
S = 20
SP = 90000

_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]

def mk(cubin, name):
    return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))

qk    = mk(f"{D}/qk100352.cubin", "r_3_25088_4_4_3_4_2_16_4")
stock = mk(f"{D}/stock100352.cubin", "r_3_24_16_16_12544_3136_12544_3136_12544_12544_12544_12544_4_4")
k1    = mk(f"{D}/k1pf_S41.cubin", "k1pf")
k2    = mk(f"{D}/k2v2.cubin", "k2v2")

rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bQ, bKV, bG = up(Q), up(KV), up(G)
bP = up(np.zeros(72*L, np.float32))
bws = up(np.zeros(S*12*6*8*258, np.float32))  # kept via _KEEP
def outb():
    o = Tensor.full((18432,), -3e38).contiguous().realize()   # poison
    _KEEP.append(o)
    return o, o.uop.buf_uop.buffer._bufs["NV"]

# ---- stock chain: QK -> numpy mx/sm -> stock PV ----
qk(bP, bQ, bKV, global_size=(25088,3,1), local_size=(4,4,3), vals=(SP,), wait=True)
P = Tensor.zeros(72*L).contiguous().realize(); P.uop.assign(...) if False else None
# read P back via a realized tensor sharing the buffer is messy; re-run into a Tensor instead
import tinygrad
# simpler: copy P buffer content through a numpy roundtrip using a realized tensor
from tinygrad.tensor import Tensor as T
pt = T.zeros(72*L, dtype=dtypes.float32).contiguous().realize(); _KEEP.append(pt)
# run QK writing into pt buffer
qk(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, global_size=(25088,3,1), local_size=(4,4,3), vals=(SP,), wait=True)
Pn = pt.numpy().reshape(72, L)
mx = Pn.max(axis=1).astype(np.float32)
sm = np.exp2((Pn - mx[:,None]).astype(np.float32) * 1.4426950216293335).sum(axis=1).astype(np.float32)
bmx, bsm = up(mx), up(sm)
o0, ob0 = outb()
stock(ob0, pt.uop.buf_uop.buffer._bufs["NV"], bmx, bsm, bKV, bG, global_size=(16,24,3), local_size=(16,1,1), vals=(0,), wait=True)
ref = o0.numpy()

# ---- new chain: K1 -> K2 ----
o1, ob1 = outb()
k1(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, bws, global_size=(S*12,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
k2(ob1, pt.uop.buf_uop.buffer._bufs["NV"], bmx, bsm, bKV, bG, bws, global_size=(72,1,1), local_size=(256,1,1), vals=(0,), wait=True)
got = o1.numpy()
print("Pn[0,:3]=", Pn[0,:3], " Pn[0,SP-2:SP+3]=", Pn[0,SP-2:SP+3], " Pn[0,-3:]=", Pn[0,-3:], flush=True)
print("mx[:3]=", mx[:3], " sm[:3]=", sm[:3], flush=True)
print("ref: max=", ref.max(), " min=", ref.min(), " nzero=", (ref==0).sum(), flush=True)
print("got: max=", got.max(), " min=", got.min(), flush=True)
rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
print(f"[chain] relerr={rel:.2e} {'OK' if rel < 1e-4 else 'FAIL'}", flush=True)
if rel >= 1e-4:
    bad = np.unravel_index(np.abs(got-ref).argmax(), got.shape)
    print("worst at", bad, "ref", ref[bad], "got", got[bad])
    print("row means ref", ref.mean(), "got", got.mean())

# ---- timings (amortized, 32 inner) ----
def bench(p, args, grid, thr, vals, n=3, inner=32):
    for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=vals)
        p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3
Pb = pt.uop.buf_uop.buffer._bufs["NV"]
t_qk = bench(qk, (Pb, bQ, bKV), (25088,3,1), (4,4,3), (SP,))
t_pv = bench(stock, (ob0, Pb, bmx, bsm, bKV, bG), (16,24,3), (16,1,1), (0,))
t_k1 = bench(k1, (Pb, bQ, bKV, bws), (S*12,1,1), (256,1,1), (SP,))
t_k2 = bench(k2, (ob1, Pb, bmx, bsm, bKV, bG, bws), (72,1,1), (256,1,1), (0,))
print(f"[t] QK={t_qk:.3f} PV={t_pv:.3f} | stock_pair={t_qk+t_pv:.3f}", flush=True)
print(f"[t] K1={t_k1:.3f} K2={t_k2:.3f} | new_pair={t_k1+t_k2:.3f}  speedup={(t_qk+t_pv)/(t_k1+t_k2):.2f}x", flush=True)
print(f"[proj16] stock={(t_qk+t_pv)*16:.1f}ms new={(t_k1+t_k2)*16:.1f}ms  save={(t_qk+t_pv-t_k1-t_k2)*16:.1f}ms/probe", flush=True)
