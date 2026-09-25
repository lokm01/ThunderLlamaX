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
L = 100352; PS = 100349; S = 24; SP = L - 500
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(11)
Pn = (rng.random(24 * PS) * 0.001).astype(np.float32)
KVn = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
G = (rng.standard_normal(12288) * 0.5).astype(np.float32)
bP, bKV, bG = up(Pn), up(KVn), up(G)
wst = Tensor.zeros(S*4*6*8*256).contiguous().realize(); _KEEP.append(wst)
bws = wst.uop.buf_uop.buffer._bufs["NV"]
k1 = mk(f"{D}/k1t1.cubin", "k1t1"); k2 = mk(f"{D}/k2t1.cubin", "k2t1")
k1(bP, bKV, bws, global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
o = Tensor.full((6144,), -3e38).contiguous().realize(); _KEEP.append(o)
ob = o.uop.buf_uop.buffer._bufs["NV"]
k2(ob, bP, bKV, bG, bws, global_size=(24,1,1), local_size=(256,1,1), vals=(0,), wait=True)
got = o.numpy()
ws = wst.numpy().reshape(S, 4, 6, 8, 256)
sig = lambda x: 1.0/(1.0+np.exp2(-x*1.4426950216293334))
ref = np.zeros(6144, np.float32)
for h in range(24):
    g, h6 = h//6, h%6
    acc = ws[:, g, h6, :, :].sum(axis=(0,1))
    ref[h*256:(h+1)*256] = acc * sig(G[256 + h*512: 256 + h*512 + 256])
rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
print(f"K2-vs-numpy-combine relerr={rel:.2e}", "OK" if rel < 1e-4 else "FAIL")
bad = np.abs(got - ref).argmax()
print("worst idx", bad, "h=", bad//256, "d=", bad%256)
