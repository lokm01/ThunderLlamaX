# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
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
D = "~/tinygrad-metal/pv3"
L = 100352; S = 20; SP = 90000
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(cubin, name):
    return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
        target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
k1 = mk(f"{D}/k1v2.cubin", "k1v2")
rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
bQ, bKV = up(Q), up(KV)
pt = Tensor.zeros(72*L).contiguous().realize(); _KEEP.append(pt)
wst = Tensor.zeros(S*12*6*258).contiguous().realize(); _KEEP.append(wst)
k1(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, wst.uop.buf_uop.buffer._bufs["NV"],
   global_size=(S*12,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
ws = wst.numpy().reshape(S*12, 6, 258)
Qh = Q.reshape(24, 3, 256)
KVg = KV.reshape(2, 4, L, 256)
LOG2E = np.float32(1.4426950216293335)
C = (L + S - 1)//S
# check (h=0,r=0) i.e. gr=0,g=0 across splits
h, r, g = 0, 0, 0
K = KVg[0, g].astype(np.float32); V = KVg[1, g].astype(np.float32)
for s in (0, 5, 17, 19):
    a, b = s*C, min((s+1)*C, L)
    end = min(b, SP + r + 1)
    w = ws[s, 0, :]   # gr=0, h6=0
    if a >= end:
        print(f"s={s} EMPTY k1(m={w[256]:.3f},l={w[257]:.4f}) expect m=-inf l=0")
        continue
    raw = (K[a:end] @ Qh[h, r]) * 0.0625
    s2 = (raw * LOG2E).astype(np.float32)
    m2 = s2.max()
    e = np.exp2(s2 - m2)
    l_true = np.float32(e.sum())
    o_true = e @ V[a:end]
    relm = abs(w[256]-m2)/max(abs(m2),1e-9); rell = abs(w[257]-l_true)/max(l_true,1e-9)
    relo = np.abs(w[:256]-o_true).max()/max(np.abs(o_true).max(),1e-9)
    print(f"s={s} range[{a},{end}) m: {w[256]:.5f} vs {m2:.5f} (rel {relm:.1e})  l: {w[257]:.3f} vs {l_true:.3f} (rel {rell:.1e})  o rel: {relo:.1e}")
