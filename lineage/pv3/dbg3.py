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
k1 = mk(f"{D}/k1v2.cubin", "k1v2"); k2 = mk(f"{D}/k2v2.cubin", "k2v2")
rng = np.random.default_rng(7)
Q = (rng.standard_normal(18432)*0.5).astype(np.float32)
KV = (rng.standard_normal(L*2048)*0.3).astype(np.float16)
G = (rng.standard_normal(36864)*0.5).astype(np.float32)
bQ, bKV, bG = up(Q), up(KV), up(G)
pt = Tensor.zeros(72*L).contiguous().realize(); _KEEP.append(pt)
bP = pt.uop.buf_uop.buffer._bufs["NV"]
wst = Tensor.zeros(S*12*6*258).contiguous().realize(); _KEEP.append(wst)
bws = wst.uop.buf_uop.buffer._bufs["NV"]
o1 = Tensor.full((18432,), -3e38).contiguous().realize(); _KEEP.append(o1)
ob1 = o1.uop.buf_uop.buffer._bufs["NV"]
k1(bP, bQ, bKV, bws, global_size=(S*12,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
k2(ob1, bP, up(np.zeros(72, np.float32)), up(np.ones(72, np.float32)), bKV, bG, bws, global_size=(72,1,1), local_size=(256,1,1), vals=(0,), wait=True)
got = o1.numpy()
wsn = wst.numpy().reshape(S*12*6, 258)
Pn = pt.numpy().reshape(72, L)

# ---- numpy full software reference ----
Qh = Q.reshape(24, 3, 256)                     # [h][r][d]
KVg = KV.reshape(2, 4, L, 256)                  # [K/V half][group][p][d] (slab-major)
nout = np.zeros((3, 24, 256), np.float64)
nl   = np.zeros((3, 24))
for r in range(3):
    end = SP + r + 1
    for h in range(24):
        g = h // 6
        K = KVg[0, g, :end].astype(np.float32)
        V = KVg[1, g, :end].astype(np.float32)
        s = (K @ Qh[h, r]) * 0.0625
        e = np.exp2((s - s.max()) * np.float32(1.4426950216293335))
        nout[r, h] = e @ V / e.sum() * (1/(1+np.exp2(-G[256 + h*512 + r*12288: 256 + h*512 + r*12288 + 256] * 1.4426950216293334)))
        nl[r, h] = e.sum()
ref_np = nout.reshape(3, 6144).T.reshape(-1)   # careful: out layout [r][h*256+d]
ref_flat = np.zeros(18432, np.float64)
for r in range(3):
    for h in range(24):
        ref_flat[r*6144 + h*256: r*6144 + h*256 + 256] = nout[r, h]
relg = np.abs(got - ref_flat).max() / np.abs(ref_flat).max()
# also compare P scores vs numpy scores for row0 head0
g0 = 0
K = KVg[0, 0, :SP+1].astype(np.float32)
s_np = (K @ Qh[0, 0]) * 0.0625
s_k1 = Pn[0*3*0 + 0, :SP+1]  # P[h*3L + r*L + p]: h=0,r=0 -> row 0
relp = np.abs(s_k1 - s_np).max() / np.abs(s_np).max()
print(f"K2-vs-numpy relerr={relg:.2e}", flush=True)
print(f"K1-scores-vs-numpy relerr (h0,r0)={relp:.2e}", flush=True)
if relp > 1e-4:
    i = np.abs(s_k1 - s_np).argmax()
    print("worst p:", i, "k1:", s_k1[i], "np:", s_np[i])
    print("first 5 k1:", s_k1[:5], "np:", s_np[:5])
