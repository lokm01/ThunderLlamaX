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

def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]

def mk(cubin, name):
    return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))

stock8 = mk(f"{D}/../splitkv/pv_stock_d.cubin", "r_24_16_16_3_1024_256_1024_256_256_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_4_4_4")
pv8    = mk(f"{D}/../splitkv/pv8kh.cubin", "pv8kh")
emu8   = mk(f"{D}/emu8192.cubin", "emu")
pv100  = mk(f"{D}/pvL100352.cubin", "pvL")
emu100 = mk(f"{D}/emu100352.cubin", "emu")

def data(L):
    rng = np.random.default_rng(21)
    Pn = L * 24
    Ps = [(rng.standard_normal(Pn)*0.5).astype(np.float32) for _ in range(3)]
    for P in Ps:
        for g in range(24): P[g*L:(g+1)*L] -= P[g*L:(g+1)*L].max()
    ms = [np.zeros(24, np.float32) for _ in range(3)]
    ss = [np.stack([np.exp2(P[g*L:(g+1)*L]).sum() for g in range(24)]).astype(np.float32) for P in Ps]
    V = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
    G = (rng.standard_normal(36864)*0.5).astype(np.float32)
    bufs = [up(V), up(G)]
    bufs += [up(x) for x in (Ps[2], ms[2], ss[2], Ps[1], ms[1], ss[1], Ps[0], ms[0], ss[0])]
    return bufs  # [V, G, P2,m2,s2, P1,m1,s1, P0,m0,s0]

def run(p, bufs, grid, thr, out=None):
    outT = out if out is not None else Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    # order: out, P2, m2, s2, V, P1, m1, s1, P0, m0, s0, G
    args = (ob, bufs[2], bufs[3], bufs[4], bufs[0], bufs[5], bufs[6], bufs[7], bufs[8], bufs[9], bufs[10], bufs[1])
    p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    return outT

def bench(p, bufs, grid, thr, n=8, inner=16):
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    args = (ob, bufs[2], bufs[3], bufs[4], bufs[0], bufs[5], bufs[6], bufs[7], bufs[8], bufs[9], bufs[10], bufs[1])
    for _ in range(3): p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=(0,))
        p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3

# ---- L=8192 ----
b8 = data(8192)
a = run(stock8, b8, (16,24,16), (16,1,1)).numpy()
e = run(emu8, b8, (16,24,16), (16,1,1)).numpy()
m = run(pv8, b8, (3,24,2), (128,1,1)).numpy()
rel_emu = np.abs(a-e).max()/max(np.abs(a).max(),1e-9)
rel_pv  = np.abs(a-m).max()/max(np.abs(a).max(),1e-9)
print(f"[8192] emu-vs-stock relerr={rel_emu:.2e}  pv-vs-stock relerr={rel_pv:.2e}", flush=True)
t_stock = bench(stock8, b8, (16,24,16), (16,1,1))
t_emu   = bench(emu8, b8, (16,24,16), (16,1,1))
t_pv    = bench(pv8, b8, (3,24,2), (128,1,1))
print(f"[8192] stock={t_stock:.3f}ms emu={t_emu:.3f}ms pv={t_pv:.3f}ms  speedup={t_stock/t_pv:.2f}x", flush=True)

# ---- L=100352 ----
b100 = data(100352)
a2 = run(emu100, b100, (16,24,16), (16,1,1)).numpy()
m2 = run(pv100, b100, (3,24,2), (128,1,1)).numpy()
rel2 = np.abs(a2-m2).max()/max(np.abs(a2).max(),1e-9)
print(f"[100352] pv-vs-emu relerr={rel2:.2e}", flush=True)
t_emu100 = bench(emu100, b100, (16,24,16), (16,1,1), n=4, inner=8)
t_pv100  = bench(pv100, b100, (3,24,2), (128,1,1), n=4, inner=8)
print(f"[100352] emu={t_emu100:.3f}ms pv={t_pv100:.3f}ms  speedup={t_emu100/t_pv100:.2f}x", flush=True)
print(f"[proj] per-probe (x48 layers): emu={t_emu100*48:.1f}ms pv={t_pv100*48:.1f}ms", flush=True)
