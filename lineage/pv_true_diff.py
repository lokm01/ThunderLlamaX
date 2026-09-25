# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys, subprocess
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.device import TinyELF
dev = Device["NV"]
rng = np.random.default_rng(21)
P2, P1, P0 = ((rng.standard_normal(196608)*0.5).astype(np.float32) for _ in range(3))
for P in (P0, P1, P2):
    for g in range(24): P[g*8192:(g+1)*8192] -= P[g*8192:(g+1)*8192].max()
m0 = np.zeros(24, np.float32); m1 = np.zeros(24, np.float32); m2 = np.zeros(24, np.float32)
s0 = np.stack([np.exp2(P0[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
s1 = np.stack([np.exp2(P1[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
s2 = np.stack([np.exp2(P2[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
V = (rng.standard_normal(16777216)*0.3).astype(np.float16)
gates = (rng.standard_normal(36864)*0.5).astype(np.float32)
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]
bP2, bP1, bP0 = up(P2), up(P1), up(P0)
bm0, bm1, bm2 = up(m0), up(m1), up(m2)
bs0, bs1, bs2 = up(s0), up(s1), up(s2)
bV, bG = up(V), up(gates)
_pi = ("v", 0, dtypes.int32, ())
stock = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv_stock_d.cubin","rb").read(),
      name="r_24_16_16_3_1024_256_1024_256_256_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_4_4_4", target=dev.renderer.target, signature=(_pi,)))
new = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv8kh.cubin","rb").read(),
      name="pv8kh", target=dev.renderer.target, signature=(_pi,)))
ARGS = (bP2, bm2, bs2, bV, bP1, bm1, bs1, bP0, bm0, bs0, bG)
def run(prog, grid, thr):
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    prog(ob, *ARGS, global_size=grid, local_size=thr, vals=(0,), wait=True)
    return outT.numpy()
a = run(stock, (16,24,16), (16,1,1))
b = run(new, (3,24,2), (128,1,1))
rel = np.abs(a-b).max()/max(np.abs(a).max(),1e-9)
print(f"[TRUE-DIFF] stock max={np.abs(a).max():.4f} relerr={rel:.2e} {'OK' if rel<1e-3 else 'FAIL'}", flush=True)
