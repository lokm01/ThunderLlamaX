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
L = 100352; S = 41; SP = 90000
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
k1 = mk(f"{D}/k1rs_S{S}.cubin", "k1rs"); k2 = mk(f"{D}/k2rs_S{S}.cubin", "k2rs")
rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bQ, bKV, bG = up(Q), up(KV), up(G)
pt = Tensor.zeros(72*L).contiguous().realize(); _KEEP.append(pt)
wst = Tensor.zeros(S*4*18*258).contiguous().realize(); _KEEP.append(wst)
bws = wst.uop.buf_uop.buffer._bufs["NV"]
# stock chain first (bisect: does the stock PV between break the K1/K2 pair?)
sqk = mk(f"{D}/qk100352.cubin", "r_3_25088_4_4_3_4_2_16_4")
spv = mk(f"{D}/stock100352.cubin", "r_3_24_16_16_12544_3136_12544_3136_12544_12544_12544_12544_4_4")
sqk(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, global_size=(25088,3,1), local_size=(4,4,3), vals=(SP,), wait=True)
Pn0 = pt.numpy().reshape(72, L)
mx0 = Pn0.max(axis=1).astype(np.float32)
sm0 = np.exp2((Pn0 - mx0[:,None]) * np.float32(1.4426950216293335)).sum(axis=1).astype(np.float32)
bmx0, bsm0 = up(mx0), up(sm0)
oref = Tensor.full((18432,), -3e38).contiguous().realize(); _KEEP.append(oref)
spv(oref.uop.buf_uop.buffer._bufs["NV"], pt.uop.buf_uop.buffer._bufs["NV"], bmx0, bsm0, bKV, bG, global_size=(16,24,3), local_size=(16,1,1), vals=(0,), wait=True)
refsv = oref.numpy()
print("stock chain done, ref nan:", np.isnan(refsv).sum())
k1(pt.uop.buf_uop.buffer._bufs["NV"], bQ, bKV, bws, global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
o1 = Tensor.full((18432,), -3e38).contiguous().realize(); _KEEP.append(o1)
k2(o1.uop.buf_uop.buffer._bufs["NV"], pt.uop.buf_uop.buffer._bufs["NV"], up(np.zeros(72, np.float32)), up(np.ones(72, np.float32)), bKV, bG, bws, global_size=(72,1,1), local_size=(256,1,1), vals=(0,), wait=True)
got = o1.numpy()
ws = wst.numpy().reshape(S, 72, 258)
# numpy combine
ref = np.zeros(18432, np.float64)
for h in range(24):
    g, h6 = h//6, h%6
    for r in range(3):
        M = -np.inf
        for s_ in range(S): M = max(M, ws[s_, g*18+h6*3+r, 256])
        Lr, O = 0.0, np.zeros(256)
        for s_ in range(S):
            w = ws[s_, g*18+h6*3+r]
            wt = 2.0**(w[256]-M)
            Lr += wt*w[257]; O += wt*w[:256]
        gt = G[256 + h*512 + r*12288: 256 + h*512 + r*12288 + 256]
        ref[r*6144 + h*256: r*6144 + h*256 + 256] = (O/Lr) * (1/(1+2.0**(-gt*1.4426950216293334)))
print("got nan:", np.isnan(got).sum())
ok = np.isfinite(got) & np.isfinite(ref)
rel = np.abs(got[ok]-ref[ok]).max()/max(np.abs(ref[ok]).max(),1e-9)
print(f"K2rs-vs-numpy relerr(finite)={rel:.2e}  finite={ok.sum()}/18432")
