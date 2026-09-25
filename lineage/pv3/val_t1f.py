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
L = 100352; PS = 100349; S = 24; SP = L - 500
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
stock = mk(f"{D}/draftpv_stock.cubin", "r_12_16_16_2_28mtp_sp2B129")
fused = mk(f"{D}/k1t1f.cubin", "k1t1f")
rng = np.random.default_rng(11)
Pn = (rng.random(24 * PS) * 0.001).astype(np.float32)
KV = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
G = (rng.standard_normal(12288) * 0.5).astype(np.float32)
bP, bKV, bG = up(Pn), up(KV), up(G)
bws = up(np.zeros(S*4*6*8*256 + 1, np.float32))   # +1 = the atomic counter (starts 0)
def outb():
    o = Tensor.full((6144,), -3e38).contiguous().realize(); _KEEP.append(o)
    return o, o.uop.buf_uop.buffer._bufs["NV"]
o0, ob0 = outb()
stock(ob0, bP, bKV, bG, global_size=(16,12,1), local_size=(16,1,1), vals=(SP,), wait=True)
ref = o0.numpy()
o1, ob1 = outb()
# TWO launches back-to-back to prove the counter self-resets
fused(ob1, bP, bKV, bG, bws, global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
got1 = o1.numpy()
o2, ob2 = outb()
fused(ob2, bP, bKV, bG, bws, global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
got2 = o2.numpy()
r1 = np.abs(got1 - ref).max() / max(np.abs(ref).max(), 1e-9)
r2 = np.abs(got2 - ref).max() / max(np.abs(ref).max(), 1e-9)
print(f"launch1 relerr={r1:.2e}  launch2 relerr={r2:.2e} (counter self-reset check)")
def bench(p, args, grid, thr, vals, n=3, inner=32):
    for _ in range(2): p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    t0 = time.perf_counter()
    for _ in range(n):
        for _ in range(inner): p(*args, global_size=grid, local_size=thr, vals=vals)
        p(*args, global_size=grid, local_size=thr, vals=vals, wait=True)
    return (time.perf_counter()-t0)/n/inner*1e3
t_f = bench(fused, (ob1, bP, bKV, bG, bws), (S*4,1,1), (256,1,1), (SP,))
t_s = bench(stock, (ob0, bP, bKV, bG), (16,12,1), (16,1,1), (SP,))
print(f"fused={t_f:.3f}ms stock={t_s:.3f}ms speedup={t_s/t_f:.2f}x  x2-draft-steps save={(t_s-t_f)*2:.1f}ms/cycle")
