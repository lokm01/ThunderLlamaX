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
C = (L + S - 1)//S
_KEEP = []
def up(a):
    t = Tensor(np.ascontiguousarray(a)).contiguous().realize(); _KEEP.append(t)
    return t.uop.buf_uop.buffer._bufs["NV"]
def mk(c, n):
    return NVProgram(dev, TinyELF(lib=open(c,"rb").read(), name=n, target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))
rng = np.random.default_rng(11)
Pn = (rng.random(24 * PS) * 0.001).astype(np.float32)
KVn = (rng.standard_normal(L * 2048) * 0.3).astype(np.float16)
bP, bKV = up(Pn), up(KVn)
wst = Tensor.zeros(S*4*6*8*256).contiguous().realize(); _KEEP.append(wst)
k1 = mk(f"{D}/k1t1.cubin", "k1t1")
k1(bP, bKV, wst.uop.buf_uop.buffer._bufs["NV"], global_size=(S*4,1,1), local_size=(256,1,1), vals=(SP,), wait=True)
ws = wst.numpy().reshape(S, 4, 6, 8, 256)
V = KVn.reshape(2, 4, L, 256)[1].astype(np.float32)

def expect(s, g, h6, warp):
    # warp handles positions p = start + 8*t + warp for all tiles t
    a = s * C
    b = min((s+1)*C, SP+1)
    ps = np.arange(a + warp, b, 8)
    h = g*6 + h6
    out = np.zeros(256, np.float32)
    if len(ps): out = (Pn[h*PS + ps][:, None] * V[g, ps]).sum(axis=0)
    return out

# check a grid of slots: which (s,g,h6,warp) match?
res = {}
for (s, g, h6, warp) in ((0,0,0,0),(0,0,0,3),(0,0,1,1),(0,1,2,5),(1,0,0,2),(1,3,5,7),(5,2,3,4)):
    e = expect(s, g, h6, warp)
    got = ws[s, g, h6, warp, :]
    rel = np.abs(got - e).max() / max(np.abs(e).max(), 1e-9)
    print(f"(s{s},g{g},h6{h6},w{warp}): rel={rel:.3f}")
    # if mismatch: search for what it actually is
    if rel > 1e-3:
        found = None
        for g2 in range(4):
            for h62 in range(6):
                for w2 in range(8):
                    e2 = expect(s, g2, h62, w2)
                    if np.abs(got - e2).max() < 1e-5:
                        found = (g2, h62, w2); break
                if found: break
            if found: break
        print(f"   actually equals expect(s{s},g{found[0]},h6{found[1]},w{found[2]})" if found else "   matches NOTHING")
