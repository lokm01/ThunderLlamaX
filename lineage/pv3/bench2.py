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

def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]

def mk(cubin, name):
    return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))

stock = mk(f"{D}/stock100352.cubin", "r_3_24_16_16_12544_3136_12544_3136_12544_12544_12544_12544_4_4")
mine  = mk(f"{D}/pv3_100352.cubin", "pv3")

rng = np.random.default_rng(21)
P = (rng.standard_normal(72*L)*0.5).astype(np.float32)
sm = np.zeros(72, np.float32)
for i in range(72):
    s = P[i*L:(i+1)*L]; s -= s.max()
    P[i*L:(i+1)*L] = s
    sm[i] = np.exp2(s).sum()
mx = np.zeros(72, np.float32)
V = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bP, bmx, bsm, bV, bG = up(P), up(mx), up(sm), up(V), up(G)

def runonce(p, grid, thr):
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    p(ob, bP, bmx, bsm, bV, bG, global_size=grid, local_size=thr, vals=(0,), wait=True)
    return outT.numpy()

def bench(p, grid, thr, n=6, inner=8):
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    args = (ob, bP, bmx, bsm, bV, bG)
    for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=(0,))
        p(*args, global_size=grid, local_size=thr, vals=(0,), wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3

a = runonce(stock, (16,24,3), (16,1,1))
b = runonce(mine,  (3,24,2),  (128,1,1))
rel = np.abs(a-b).max()/max(np.abs(a).max(),1e-9)
print(f"[100352-real] relerr={rel:.2e} {'OK' if rel < 1e-3 else 'FAIL'}", flush=True)
t_s = bench(stock, (16,24,3), (16,1,1))
t_m = bench(mine,  (3,24,2),  (128,1,1))
print(f"[100352-real] stock={t_s:.3f}ms pv3={t_m:.3f}ms speedup={t_s/t_m:.2f}x", flush=True)
print(f"[proj x16] stock={t_s*16:.1f}ms pv3={t_m*16:.1f}ms per probe", flush=True)
