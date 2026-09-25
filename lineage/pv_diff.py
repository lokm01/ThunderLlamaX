# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Final differential: stock PV kernel vs pv8k on random contract inputs."""
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
m = np.zeros(24, np.float32)
s0 = np.stack([np.exp2(P0[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
s1 = np.stack([np.exp2(P1[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
s2 = np.stack([np.exp2(P2[g*8192:(g+1)*8192]).sum() for g in range(24)]).astype(np.float32)
V = (rng.standard_normal(16777216)*0.3).astype(np.float16)
gates = (rng.standard_normal(36864)*0.5).astype(np.float32)
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
    return t.uop.buf_uop.buffer._bufs["NV"]
bP2, bP1, bP0 = up(P2), up(P1), up(P0)
bm, bs0, bs1, bs2 = up(m), up(s0), up(s1), up(s2)
bV, bG = up(V), up(gates)
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin")+":/opt/homebrew/bin:/usr/bin:/bin")
name = subprocess.run(["bash","-c","strings ~/tinygrad-metal/splitkv/pv_stock.cubin | grep -m1 '^r_24_16'"],capture_output=True,text=True,env=env).stdout.strip()
stock = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv_stock.cubin","rb").read(), name=name, target=dev.renderer.target, signature=tuple()))
new = NVProgram(dev, TinyELF(lib=open("~/tinygrad-metal/splitkv/pv8k.cubin","rb").read(), name="pv8k", target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
ARGS = (bP2, bm, bs2, bV, bP1, bm, bs1, bP0, bm, bs0, bG)
def run(prog, grid, thr, vals=()):
    outT = Tensor.zeros(18432).contiguous().realize()
    ob = outT.uop.buf_uop.buffer._bufs["NV"]
    prog(ob, *ARGS, global_size=grid, local_size=thr, vals=vals, wait=True)
    return outT.numpy()
a = run(stock, (16,24,16), (16,1,1))
b = run(new, (3,24,2), (128,1,1), vals=(0,))
rel = np.abs(a-b).max()/max(np.abs(a).max(),1e-9)
print(f"[pv-diff] relerr={rel:.2e} {'OK' if rel<1e-3 else 'FAIL'}", flush=True)
print(f"[pv-diff] stock[:3]={a[:3]} new={b[:3]}", flush=True)
