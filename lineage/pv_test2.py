# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""PV differential: stock vs pv_coalesced on random contract inputs."""
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
rng = np.random.default_rng(9)
P2, P1, P0 = ((np.random.randn(196608)*0.5).astype(np.float32) for _ in range(3))
for P in (P0, P1, P2):
    for g in range(24): P[g*8192:(g+1)*8192] -= P[g*8192:(g+1)*8192].max()
m = np.zeros(24, np.float32)
s0 = np.stack([np.exp2(P0[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
s1 = np.stack([np.exp2(P1[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
s2 = np.stack([np.exp2(P2[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
V = (np.random.randn(16777216)*0.3).astype(np.float16)
gates = (np.random.randn(36864)*0.5).astype(np.float32)

def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]
bP2, bP1, bP0 = up(P2), up(P1), up(P0)
bm, bs0, bs1, bs2 = up(m), up(s0), up(s1), up(s2)
bV, bG = up(V), up(gates)

stock = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv_stock.cubin","rb").read(),
      name="r_24_16_16_3_1024_256_1024_256_256_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_1024_4_4_4",
      target=dev.renderer.target, signature=tuple()))
new = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/splitkv_pv.cubin","rb").read(),
      name="pv_coalesced", target=dev.renderer.target, signature=tuple()))
print("[progs loaded]", flush=True)

def run(prog, grid, thr):
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    prog(ob, bP2, bm, bm, bV, bP1, bm, bm, bP0, bm, bm, bG,
         global_size=grid, local_size=thr, wait=True)
    return outT.numpy()

# NOTE stock arg order: data0,data1(P2),data2(max2),data3(sum2),data4(V),data5(P1),
# data6(max1),data7(sum1),data8(P0),data9(max0),data10(sum0),data11(gates)
def run_stock():
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    stock(ob, bP2, bm, bs2, bV, bP1, bm, bs1, bP0, bm, bs0, bG,
          global_size=(16,24,16), local_size=(16,1,1), wait=True)
    return outT.numpy()
def run_new():
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    new(ob, bP2, bm, bs2, bV, bP1, bm, bs1, bP0, bm, bs0, bG,
        global_size=(3,24,1), local_size=(256,1,1), wait=True)
    return outT.numpy()

a = run_stock(); b = run_new()
rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-9)
print(f"[pv] relerr={rel:.2e} {'OK' if rel < 1e-3 else 'FAIL'}", flush=True)
print("[pv] stock[0,:4]=", a[:4], " new=", b[:4], flush=True)
