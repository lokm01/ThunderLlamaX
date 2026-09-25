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
L = 100352; PS = 100349; SP = L - 500
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(3)
KVn = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
G = (rng.standard_normal(12288) * 0.5).astype(np.float32)
P = np.zeros(24 * PS, np.float32)
H, K1P = 1, 5000
P[H * PS + K1P] = 1.0
bP, bKV, bG = up(P), up(KVn), up(G)
o = Tensor.full((6144,), -3e38).contiguous().realize(); _KEEP.append(o)
stock = mk(f"{D}/draftpv_stock.cubin", "r_12_16_16_2_28mtp_sp2B129")
stock(o.uop.buf_uop.buffer._bufs["NV"], bP, bKV, bG, global_size=(16,12,1), local_size=(16,1,1), vals=(SP,), wait=True)
r = o.numpy()
V = KVn.reshape(2, 4, L, 256)[1].astype(np.float32)
g = H // 6
sig = lambda x: 1.0 / (1.0 + np.exp2(-x * 1.4426950216293334))
exp_h1 = V[g, K1P] * sig(G[256 + H*512 : 256 + H*512 + 256])
nz = np.nonzero(np.abs(r) > 1e-9)[0]
print("stock nonzero outputs:", len(nz), "of 6144")
if len(nz):
    h_lit = nz[0] // 256
    row = r[H*256:(H+1)*256]
    print(f"head {H}: row-vs-expected relerr = {np.abs(row - exp_h1).max()/max(np.abs(exp_h1).max(),1e-9):.3f}")
    print("  expected = V[g,K]*sigmoid(gate):", exp_h1[:3])
    print("  stock gave:", row[:3])
